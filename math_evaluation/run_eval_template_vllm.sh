################################################################################
#
# 日期：2025.06.01 19:33
# 用于推理的模板文件
# 
################################################################################
set -x
START_TIME=$(date +%s)
export HF_ENDPOINT=https://hf-mirror.com

# ------------------------------
# CUDA environment configuration
# ------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# For example, only use the three 4090s (adjust based on nvidia-smi output)
export CUDA_VISIBLE_DEVICES=1,2,3
# ------------------------------
export HF_ENDPOINT=https://hf-mirror.com



# 默认参数
GRADE_METHOD=normal

# 参数解析
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --prompt_ratio)
      PROMPT_RATIO="$2"
      shift 2
      ;;
    --num_rollout)
      NUM_ROLLOUT="$2"
      shift 2
      ;;
    --dataset_name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --gen_mode)
      GEN_MODE="$2"
      shift 2
      ;;
    --apply_template)
      APPLY_TEMPLATE="$2"
      shift 2
      ;;
    --grade_method)
      GRADE_METHOD="$2"
      shift 2
      ;;
    --task_info)
      TASK_INFO="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 检查必填参数是否为空
if [[ -z "$MODEL_NAME" || -z "$MODEL_PATH" || -z "$PROMPT_RATIO" || -z "$NUM_ROLLOUT" || -z "$DATASET_NAME" || -z "$DATASET_PATH" || -z "$GEN_MODE" || -z "$APPLY_TEMPLATE" || -z "$TASK_INFO" ]]; then
  echo "错误：缺少必要参数。"
  exit 1
fi

# 获取当前日期时间作为文件名标识，例如 20250530_113015
DATE_STR=$(date +%Y%m%d_%H%M%S)
FOLDER_NAME="${MODEL_NAME}_${DATASET_NAME}_ratio_${PROMPT_RATIO}_rollout_${NUM_ROLLOUT}_at_${DATE_STR}_${TASK_INFO}"
OUTPUT_DIR="output/${FOLDER_NAME}"
mkdir -p "$OUTPUT_DIR"
echo "已创建输出目录: $OUTPUT_DIR"
echo "所有输出重定向至: ${OUTPUT_DIR}/run.log"

# 重定向 stdout 和 stderr 到日志文件
exec >> "${OUTPUT_DIR}/run.log" 2>&1

# 随机选择一个端口
START_VLLM_PORT=40197
# 启动并行推理任务，使用全部GPU
NUM_TASKS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
for ((i=0; i<NUM_TASKS; i++)); do
    RESULT_FILE="${OUTPUT_DIR}/model_infer_result"
    LOG_FILE="${OUTPUT_DIR}/model_infer_result_${i}.log"

    echo "启动任务 $i"

    VLLM_PORT=$START_VLLM_PORT CUDA_VISIBLE_DEVICES=$i python -u inference_vllm.py \
        --model_path  $MODEL_PATH \
        --input_file "$DATASET_PATH" \
        --output_file "$RESULT_FILE" \
        --task_id "$i" \
        --task_count "$NUM_TASKS" \
        --prompt_ratio "$PROMPT_RATIO" \
        --num_rollout "$NUM_ROLLOUT" \
        --gen_mode "$GEN_MODE" \
        --apply_template "$APPLY_TEMPLATE" \
        --grade_method "$GRADE_METHOD" > "$LOG_FILE" 2>&1 &
        
    # 防止端口冲突
    # sleep 4
    ((START_VLLM_PORT += 100))
done

# 等待所有后台任务完成
wait
echo "所有推理任务已完成"

# 合并所有 JSONL 文件
MERGED_RESULT_FILE="${OUTPUT_DIR}/model_infer_result_merged.jsonl"
> "$MERGED_RESULT_FILE"  # 创建空文件

for ((i=0; i<NUM_TASKS; i++)); do
    RESULT_FILE="${OUTPUT_DIR}/model_infer_result_${i}.jsonl"
    cat "$RESULT_FILE" >> "$MERGED_RESULT_FILE"
done

echo "所有结果已合并到：$MERGED_RESULT_FILE"

##################################################
#
# 这里需要调用统计指标的脚本
#
##################################################
python calculate_metrics.py \
    --input_file $MERGED_RESULT_FILE \
    --output_file "${OUTPUT_DIR}/${FOLDER_NAME}" \
    --metrics_file "${OUTPUT_DIR}/metrics.json"

# 复制当前脚本文件到输出目录中备份
SCRIPT_BACKUP_PATH="$OUTPUT_DIR/src"
mkdir -p $SCRIPT_BACKUP_PATH

SCRIPT_PATH="$(realpath "$0")"
cp "$SCRIPT_PATH" "$SCRIPT_BACKUP_PATH/$(basename "$SCRIPT_PATH")"
cp "inference_vllm.py" "$SCRIPT_BACKUP_PATH/inference_vllm.py"
cp "calculate_metrics.py" "$SCRIPT_BACKUP_PATH/calculate_metrics.py"
echo "已将相关脚本复制到输出目录"


END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "运行结束，总耗时：${HOURS}小时${MINUTES}分钟${SECONDS}秒"
