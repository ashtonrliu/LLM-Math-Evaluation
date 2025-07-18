import argparse
from datetime import datetime
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import json
import jsonlines
from tqdm import tqdm
import evaluate
from qwen.qwen_math_parser import extract_answer
from qwen.math_grade import grade_answer
from grade_functions import grade_custom_math_problem
import multiprocessing as mp
import torch

import os
# 禁用并行化，避免子进程警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def truncate_prompt(prompt, ratio=0.8):
    """按比例截断字符串，保留完整单词"""
    if ratio >= 1.0:
        return prompt
    
    cutoff = int(len(prompt) * ratio)
    if cutoff >= len(prompt):
        return prompt
    while cutoff < len(prompt) and prompt[cutoff] not in " \n,.!?)]}":
        cutoff += 1
    return prompt[:cutoff].rstrip()

def get_truncate_fn(ratio):
    def inner(example):
        example["origin_prompt"] = example["prompt"]
        example["truncate_prompt"] = truncate_prompt(example["prompt"], ratio)
        example["prompt"] = example["truncate_prompt"]
        return example
    return inner

def cal_rouge(rouge, origin_problem, prompt, completion):
    target = origin_problem[len(prompt):].strip()
    prediction = completion.strip()
    prediction = " ".join(prediction.split()[:len(target.split())])
    rouge_result = rouge.compute(predictions=[prediction], references=[target], use_stemmer=True)
    exact_match = bool(round(rouge_result["rougeL"], 4) == 1.0)
    return rouge_result, exact_match


def worker_loop(task_queue, result_queue, grade_method):
    pid = os.getpid()

    grade_func = None
    if grade_method == "normal":
        print("[worker_loop] choose grade_answer")
        grade_func = grade_answer
    elif grade_method == "absolute_diff":
        print("[worker_loop] choose grade_custom_math_problem")
        grade_func = grade_custom_math_problem
    
    if grade_func is None:
        print(f"[worker_loop {pid}] unknown grade_method.")
        return
    print(f"[worker_loop {pid}] grade_method is {grade_method}, grade_func is {grade_func}")
    
    while True:
        try:
            extractd, answer = task_queue.get()
            # 处理任务并返回

            is_correct = grade_func(extractd, answer)

            result_queue.put((is_correct))
        except Exception as e:
            print(f"[{pid}] 子进程出现异常：{e}")

def create_worker(grade_method):
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    worker = mp.Process(target=worker_loop, args=(task_queue, result_queue, grade_method))
    worker.start()
    print(f"✅ 已创建用于处理任务的子进程: {worker.pid}, {grade_method}")
    return worker, task_queue, result_queue


def load_model(model_path, gen_kwargs=None):
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
    generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

    if gen_kwargs:
        print(f"🔧 Applying custom generation config from JSON:")
        for k, v in gen_kwargs.items():
            if hasattr(generation_config, k):
                setattr(generation_config, k, v)
                print(f"  ✅ {k} = {v}")
            else:
                print(f"  ⚠️ generation_config has no attribute: {k}, ignoring.")

    model.generation_config = generation_config

    print(model)
    print(model.generation_config)
    return model, tokenizer


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # 获取 end-of-text token ID
    eos_token_id = tokenizer.eos_token_id
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        # 截取生成部分的 token
        gen_tokens = tokens[raw_text_len:]
        # 如果包含 eos_token_id，就截断到它之前
        if eos_token_id in gen_tokens:
            gen_tokens = gen_tokens[:gen_tokens.index(eos_token_id)]
        # 解码
        sent = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, input_txt):
    input_ids = tokenizer.encode(input_txt)
    raw_text_len = len(input_ids)
    context_enc = torch.tensor([input_ids]).to(model.device)
    print(f"Input text: {input_txt}\n")
    with torch.no_grad():
        outputs = model.generate(context_enc)
        output_text = decode(outputs, tokenizer, raw_text_len)[0]
        print(f"\nOutput text: {output_text}\n")
        return output_text

def generate_using_hf_model(model_path, prompts, gen_config):
    model, tokenizer = load_model(model_path, gen_config)
    results = []
    for prompt in prompts:
        results.append(generate_sample(model, tokenizer, prompt))
    return results


gen_mode_to_params_dict = {
    # 用于贪婪生成的采样参数
    "greedy": {"do_sample":False,"temperature":1.0,"top_p":1.0,"max_new_tokens":4096},
    # 用于采样的生成参数
    "sample": SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=4096),
}

if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description="Distributed Prompt Inference")
    parser.add_argument("--model_path",         type=str, required=True)
    parser.add_argument("--input_file",         type=str, required=True)
    parser.add_argument("--output_file",        type=str, required=True)
    parser.add_argument("--task_id",            type=int, required=True)
    parser.add_argument("--task_count",         type=int, required=True)
    parser.add_argument("--prompt_ratio",       type=int, required=True)
    parser.add_argument("--num_rollout",        type=int, required=True)
    parser.add_argument("--gen_mode",           type=str, required=True)
    parser.add_argument("--apply_template",     type=int, required=True)
    parser.add_argument("--grade_method",       type=str, default="normal")
    args = parser.parse_args()

    print("Arguments (JSON format):")
    print(json.dumps(vars(args), indent=4, ensure_ascii=False))

    if not (10 <= args.prompt_ratio <= 100):
        raise ValueError("prompt_ratio 必须在 10 到 100 之间")

    ratio_float = args.prompt_ratio / 100.0
    output_file = f"{args.output_file}_{args.task_id}.jsonl"
    print(f"output_file={output_file}")

    sampling_params = gen_mode_to_params_dict.get(args.gen_mode)
    if sampling_params is None:
        raise ValueError("Bad Generation Mode.")
    print(sampling_params)

    dataset = load_dataset("json", data_files=args.input_file, split="train")
    dataset = dataset.shard(num_shards=args.task_count, index=args.task_id, contiguous=False)
    dataset = dataset.map(get_truncate_fn(ratio_float), desc="Processing Prompt")

    rouge = evaluate.load("rouge")

    if args.apply_template == 1:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        def apply_template(item):
            msgs = {"role": "user", "content": item["prompt"]}
            item["prompt"] = tokenizer.apply_chat_template(
                [msgs], tokenize=False, add_generation_prompt=True
            )
            return item

        dataset = dataset.map(apply_template, desc="Applying Chat Template")

    dataset_list = list(dataset)
    all_prompts = []
    for data in dataset_list:
        all_prompts.extend([data["prompt"]] * args.num_rollout)

    print(f"Total prompts to generate: {len(all_prompts)}")

    model_outputs = []
    if args.gen_mode == "greedy":
        """
        贪婪生成模式走HuggingFace接口
        """
        print("use hf model for inference.")
        model_outputs = generate_using_hf_model(args.model_path, all_prompts, sampling_params)
    else:
        """
        采样模式走vLLM，多轮速度更快
        """
        print("use vllm for inference.")
        llm = LLM(model=args.model_path)

        all_generations = llm.generate(all_prompts, sampling_params=sampling_params)
        assert len(all_generations) == len(all_prompts), "llm.generate 输出数量不一致"
        for g in all_generations:
            for output in g.outputs:
                model_outputs.append(output.text)

    evaluate_start = datetime.now()
    # 创建工作者线程
    worker, task_queue, result_queue = create_worker(args.grade_method)
    results = []
    for i, data in enumerate(tqdm(dataset_list, desc="Evaluating Model Output")):
        prompt = data["prompt"]
        answer = data["answer"]
        model_generation = []
        accuracy_avg = 0.0
        pass_k = False
        rougel_avg = 0.0
        exact_match_rate = 0.0

        offset = i * args.num_rollout
        generations = model_outputs[offset:offset + args.num_rollout]

        for model_output in generations:
            details = {}
            details["model_output"] = model_output

            if args.prompt_ratio != 100:
                rouge_result, exact_match = cal_rouge(
                    rouge, data["origin_prompt"], data["truncate_prompt"], model_output
                )
                details["exact_match"] = exact_match
                details["rouge_result"] = rouge_result
                rougel_avg += rouge_result["rougeL"]
                exact_match_rate += 1.0 if exact_match else 0.0

            extractd = extract_answer(model_output, "math")

            # 评估过程可能由于表达式太复杂而卡死，因此采用异步方式
            task_queue.put((extractd, answer))
            try:
                # 最多等待30s，超时就返回
                is_correct = result_queue.get(timeout=30)
            except Exception as e:
                print(f"评估答案时出现异常：{extractd} ---> {answer}")
                print(f"检查答案过程超时未响应，正在终止并重启子进程")
                worker.terminate()
                worker.join()
                worker, task_queue, result_queue = create_worker(args.grade_method)
                is_correct = False

            if float(is_correct) == 1.0:
                    pass_k = True

            details["extract_answer"] = extractd
            details["is_correct"] = is_correct

            accuracy_avg += float(is_correct)
            model_generation.append(details)

        accuracy_avg /= args.num_rollout
        rougel_avg /= args.num_rollout
        exact_match_rate /= args.num_rollout

        result = dict(data)
        result["rollout"] = args.num_rollout
        result["accuracy-avg"] = accuracy_avg
        result["accuracy-pass"] = float(pass_k)
        result["rougel-avg"] = rougel_avg
        result["exact_match_rate"] = exact_match_rate
        result["model_generation"] = model_generation
        results.append(result)

    worker.terminate()
    worker.join()
    print("子进程已停止")

    evaluate_end = datetime.now()

    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(results)

    end_time = datetime.now()
    print(f"\nStart Time:   {start_time}")
    print(f"End Time:       {end_time}")
    print(f"Elapsed:        {end_time - start_time}")
    print(f"Evaluate Time:  {evaluate_end - evaluate_start}")
