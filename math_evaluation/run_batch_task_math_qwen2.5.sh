###########################################
#
# 2025.07.14 测试Qwen模型在数学数据集上的能力
#
###########################################

#set -x

START_TIME=$(date +%s)

# 数据集名称列表
DATASET_NAMES=(
    aime2024_30
    aime2025_30
    amc_83
    livemathbench_100
    math_500
    minervamath_272
)

# 数据集路径列表，需要和DATASET_NAMES一一对应
DATASET_PATHS=(
    dataset/AIME2024/train.json
    dataset/AIME2025/train.json
    dataset/AMC/train.json
    dataset/LiveMathBench/livemathbench_2504_v2.json
    dataset/MATH-500/train.json
    dataset/MinervaMath/minervamath.json
)

# 模型名称列表
MODEL_NAMES=(
    vllm_qwen_2.5_math_7b
    vllm_qwen_2.5_math_7b_instruct
    vllm_qwen_2.5_7b
    vllm_qwen_2.5_7b_instruct
)

# 模型路径列表，需要和MODEL_NAMES一一对应
MODEL_PATHS=(
    your_path_to/Qwen2.5/Qwen2.5-Math-7B
    your_path_to/Qwen2.5/Qwen2.5-Math-7B-Instruct
    your_path_to/Qwen2.5/Qwen2.5-7B
    your_path_to/Qwen2.5/Qwen2.5-7B-Instruct
)

# 打分方法
GRADE_METHOD=normal     # 正常评分

for ((i = 0; i < ${#MODEL_NAMES[@]}; i++)); do
    MODEL_NAME=${MODEL_NAMES[i]}
    MODEL_PATH=${MODEL_PATHS[i]}
    
    for ((j = 0; j < ${#DATASET_NAMES[@]}; j++)); do
        DATASET_NAME=${DATASET_NAMES[j]}
        DATASET_PATH=${DATASET_PATHS[j]}
        
        RATIOS=(100 80 60 40)
        for ((k = 0; k < ${#RATIOS[@]}; k++)); do
            RATIO=${RATIOS[k]}
            # 不应用模板且不采样
            NUM_ROLLOUT=01; GEN_MODE="greedy";  APPLY_TEMPLATE="0";     TASK_INFO="greedy_only"
            bash run_eval_template_vllm.sh --model_name ${MODEL_NAME} --model_path ${MODEL_PATH}  --prompt_ratio ${RATIO} --num_rollout ${NUM_ROLLOUT} --dataset_name ${DATASET_NAME} --dataset_path ${DATASET_PATH} --gen_mode ${GEN_MODE} --apply_template ${APPLY_TEMPLATE} --grade_method ${GRADE_METHOD} --task_info ${TASK_INFO}

            # 不应用模板但采样
            NUM_ROLLOUT=16; GEN_MODE="sample";  APPLY_TEMPLATE="0";     TASK_INFO="sample_only"
            bash run_eval_template_vllm.sh --model_name ${MODEL_NAME} --model_path ${MODEL_PATH}  --prompt_ratio ${RATIO} --num_rollout ${NUM_ROLLOUT} --dataset_name ${DATASET_NAME} --dataset_path ${DATASET_PATH} --gen_mode ${GEN_MODE} --apply_template ${APPLY_TEMPLATE} --grade_method ${GRADE_METHOD} --task_info ${TASK_INFO}
            
            # 应用模板但不采样
            NUM_ROLLOUT=01; GEN_MODE="greedy";  APPLY_TEMPLATE="1";     TASK_INFO="template_greedy"
            bash run_eval_template_vllm.sh --model_name ${MODEL_NAME} --model_path ${MODEL_PATH}  --prompt_ratio ${RATIO} --num_rollout ${NUM_ROLLOUT} --dataset_name ${DATASET_NAME} --dataset_path ${DATASET_PATH} --gen_mode ${GEN_MODE} --apply_template ${APPLY_TEMPLATE} --grade_method ${GRADE_METHOD} --task_info ${TASK_INFO}
            
            # 应用模板并采样
            NUM_ROLLOUT=16; GEN_MODE="sample";  APPLY_TEMPLATE="1";     TASK_INFO="template_sample"
            bash run_eval_template_vllm.sh --model_name ${MODEL_NAME} --model_path ${MODEL_PATH}  --prompt_ratio ${RATIO} --num_rollout ${NUM_ROLLOUT} --dataset_name ${DATASET_NAME} --dataset_path ${DATASET_PATH} --gen_mode ${GEN_MODE} --apply_template ${APPLY_TEMPLATE} --grade_method ${GRADE_METHOD} --task_info ${TASK_INFO}

        done
    done
done


END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "运行结束，总耗时：${HOURS}小时${MINUTES}分钟${SECONDS}秒"

