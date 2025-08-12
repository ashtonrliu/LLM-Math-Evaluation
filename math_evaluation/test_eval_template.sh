echo '{"prompt":"How many positive whole-number divisors does 196 have?","answer":"9"}' > /tmp/smoke.jsonl

# Greedy (HF)
python inference_vllm.py \
  --model_path /home/aliu/dev/models/Qwen2.5/Qwen2.5-Math-7B \
  --input_file /tmp/smoke.jsonl \
  --output_file /tmp/out \
  --task_id 0 --task_count 1 \
  --prompt_ratio 100 --num_rollout 1 \
  --gen_mode greedy --apply_template 0 --grade_method normal