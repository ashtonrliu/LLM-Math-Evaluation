import os
import re
import json
import csv
from collections import defaultdict

# 六个固定前缀
KNOWN_PREFIXES = [
    "vllm_qwen_2.5_math_7b",
    "vllm_qwen_2.5_math_7b_instruct",
    "vllm_qwen_2.5_7b",
    "vllm_qwen_2.5_7b_instruct",
    "vllm_llama_3.1_8b",
    "vllm_llama_3.1_8b_instruct",
]

RATIOS = ["100", "80", "60", "40"]
METRICS = ["accuracy-avg", "accuracy-pass", "rougel-avg", "exact_match_rate"]

# 正则表达式用于解析目录名
PREFIX_PATTERNS = {
    prefix: re.compile(
        re.escape(prefix) +
        r"_(?P<dataset>[a-zA-Z0-9]+)_" +
        r"(?P<count>\d+)_ratio_" +
        r"(?P<ratio>\d+)_rollout_" +
        r"(?P<rollout>\d+)_at_" +
        r"(?P<timestamp>\d{8}_\d{6})_" +
        r"(?P<mode>[a-zA-Z0-9_]+)$"
    )
    for prefix in KNOWN_PREFIXES
}

def get_subdirectories(root):
    return [name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))]

def match_prefix(name):
    matches = [p for p in KNOWN_PREFIXES if name.startswith(p)]
    return max(matches, key=len) if matches else None

def parse_dirname(dirname, prefix):
    pattern = PREFIX_PATTERNS.get(prefix)
    if not pattern:
        return None
    match = pattern.match(dirname)
    return match.groupdict() if match else None

def read_metrics_json(path):
    metrics_path = os.path.join(path, "metrics.json")
    if not os.path.isfile(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

if __name__ == "__main__":
    root_dir = "output"
    grouped_by_mode = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # grouped_by_mode[mode][prefix][dataset][ratio][metric] = value

    for dirname in get_subdirectories(root_dir):
        prefix = match_prefix(dirname)
        if not prefix:
            continue
        fields = parse_dirname(dirname, prefix)
        if not fields or fields["ratio"] not in RATIOS:
            continue

        metrics = read_metrics_json(os.path.join(root_dir, dirname))
        if not metrics:
            continue

        dataset = fields["dataset"]
        ratio = fields["ratio"]
        mode = fields["mode"]

        for metric_name in METRICS:
            if metric_name in metrics:
                try:
                    value = float(metrics[metric_name].get("average", 0.0)) * 100
                    formatted = f"{value:.2f}"
                except Exception:
                    formatted = ""
                grouped_by_mode[mode][prefix][dataset].setdefault(ratio, {})[metric_name] = formatted


    # 写入多个 CSV 文件
    for mode, mode_data in grouped_by_mode.items():
        output_csv = f"metrics_wide_summary_{mode}.csv"
        header = ["prefix", "dataset"]
        for ratio in RATIOS:
            for metric in METRICS:
                header.append(f"{ratio}%-{metric}")

        with open(output_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for prefix in KNOWN_PREFIXES:
                datasets = ["math", "amc", "aime2024", "aime2025", "minervamath", "livemathbench"]
                for dataset in datasets:
                    row = [prefix, dataset]
                    for ratio in RATIOS:
                        metric_dict = mode_data[prefix][dataset].get(ratio, {})
                        for metric in METRICS:
                            row.append(metric_dict.get(metric, ""))
                    writer.writerow(row)

        print(f"✅ 已写入：{output_csv}")
