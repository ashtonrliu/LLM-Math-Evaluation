import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算指标")
    parser.add_argument("--input_file", type=str, help="输入 JSONL 文件路径")
    parser.add_argument("--output_file", type=str, help="输出 JSON 文件路径前缀")
    parser.add_argument("--metrics_file", type=str, help="输出指标 JSON 文件路径")
    args = parser.parse_args()

    # 读取 JSONL 文件
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # 初始化累计值和计数器
    stats = {
        "accuracy-avg": {"sum": 0.0, "count": 0},
        "accuracy-pass": {"sum": 0.0, "count": 0},
        "rougel-avg": {"sum": 0.0, "count": 0},
        "exact_match_rate": {"sum": 0.0, "count": 0}
    }

    for item in data:
        for key in stats:
            if key in item and isinstance(item[key], (int, float)):
                stats[key]["sum"] += item[key]
                stats[key]["count"] += 1

    # 计算平均值并构造输出
    averages = {}
    for key in stats:
        count = stats[key]["count"]
        avg = stats[key]["sum"] / count if count > 0 else None
        averages[key] = {
            "sum": stats[key]["sum"],
            "count": count,
            "average": round(avg, 6) if avg is not None else None
        }

    # 打印结果
    print(f"样本总数: {len(data)}")
    for key, val in averages.items():
        if val["average"] is not None:
            print(f"平均 {key}: {val['average']:.4f} (共 {val['count']} 条记录)")
        else:
            print(f"{key} 字段缺失或无有效值")

    # 保存 JSON 数据本体
    json_output_path = args.output_file + ".json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"已成功将 {args.input_file} 转换为 {json_output_path}")

    # 保存指标为 metrics_file
    if args.metrics_file:
        with open(args.metrics_file, "w", encoding="utf-8") as f:
            json.dump(averages, f, ensure_ascii=False, indent=4)
        print(f"已将指标保存到 {args.metrics_file}")
