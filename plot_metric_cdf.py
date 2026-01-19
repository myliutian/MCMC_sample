import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import argparse

# -------------------- 指标定义 --------------------
METRICS = ["f1", "recall", "precision", "iou", "dcc"]

# True = 越大越好（CDF）
# False = 越小越好（反向CDF）
METRIC_DIRECTION = {
    "f1": True,
    "recall": True,
    "precision": True,
    "iou": True,
    "dcc": False,
}


# -------------------- 数据读取 --------------------
def load_values_and_success(json_path):
    """
    返回：
        metrics_values: dict {metric_name: np.array(values)}
        success_ratio: float, success_6A 为 True 的比例
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    metrics_values = {m: [] for m in METRICS}
    success_true = 0
    total = 0

    for _, content in data.items():
        metrics = content.get("metrics", {})
        total += 1
        if metrics.get("success_6A", False):
            success_true += 1
        for m in METRICS:
            val = metrics.get(m)
            if val is not None:
                metrics_values[m].append(val)

    # 转 np.array
    for m in METRICS:
        metrics_values[m] = np.array(metrics_values[m], dtype=float)

    success_ratio = success_true / total if total > 0 else 0.0

    return metrics_values, success_ratio


# -------------------- CDF 计算 --------------------
def compute_cdf(values, reverse=False):
    values = np.sort(values)
    n = len(values)
    cdf = np.arange(1, n + 1) / n
    if reverse:
        cdf = 1.0 - cdf + 1.0 / n
    return values, cdf


# -------------------- 主函数 --------------------
def main():
    parser = argparse.ArgumentParser("Metrics CDF + CSV pipeline")
    parser.add_argument("--scheme", nargs=4, required=True,
                        help="Paths to four JSON files, in order of schemes")
    parser.add_argument("--scheme_names", nargs=4, default=None,
                        help="Optional names for the four schemes")
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory to save figures and CSV")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 默认方案名
    scheme_names = args.scheme_names if args.scheme_names else [f"scheme{i+1}" for i in range(4)]

    # ---------- 读取所有方案数据 ----------
    all_scheme_data = []
    for json_path in args.scheme:
        metrics_values, success_ratio = load_values_and_success(Path(json_path))
        all_scheme_data.append({
            "metrics_values": metrics_values,
            "success_ratio": success_ratio
        })

    # ---------- 生成每个指标的 CDF 图 ----------
    for metric in METRICS:
        reverse_cdf = not METRIC_DIRECTION[metric]
        plt.figure(figsize=(8, 6))
        all_values = []

        for name, scheme_data in zip(scheme_names, all_scheme_data):
            values = scheme_data["metrics_values"][metric]
            if len(values) == 0:
                continue
            x, y = compute_cdf(values, reverse=reverse_cdf)
            plt.plot(x, y, label=name)
            all_values.append(values)

        # 全局统计（所有方案合并）
        all_values_concat = np.concatenate(all_values)
        p50 = np.percentile(all_values_concat, 50)
        p90 = np.percentile(all_values_concat, 90)
        p99 = np.percentile(all_values_concat, 99)
        avg = np.mean(all_values_concat)

        # 水平线
        plt.axhline(0.5, linestyle="--", color="gray", alpha=0.6)
        plt.axhline(0.9, linestyle="--", color="gray", alpha=0.6)
        plt.axhline(0.99, linestyle="--", color="gray", alpha=0.6)

        # 注释
        plt.text(0.98, 0.52, f"P50 = {p50:.4f}", transform=plt.gca().transAxes,
                 ha="right", va="bottom")
        plt.text(0.98, 0.92, f"P90 = {p90:.4f}", transform=plt.gca().transAxes,
                 ha="right", va="bottom")
        plt.text(0.98, 0.995, f"P99 = {p99:.4f}", transform=plt.gca().transAxes,
                 ha="right", va="bottom")
        plt.text(0.98, 0.05, f"Avg = {avg:.4f}", transform=plt.gca().transAxes,
                 ha="right", va="bottom")

        plt.xlabel(metric)
        if reverse_cdf:
            plt.ylabel("Reverse CDF (P[X ≥ x])")
            plt.title(f"Reverse CDF of {metric} (lower is better)")
        else:
            plt.ylabel("CDF (P[X ≤ x])")
            plt.title(f"CDF of {metric} (higher is better)")

        plt.grid(True)
        plt.legend()
        fig_path = output_dir / f"{metric}_cdf.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fig_path}")

    # ---------- 生成 CSV ----------
    csv_path = output_dir / "metrics_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = METRICS + ["success_6A_ratio"]
        writer = csv.DictWriter(f, fieldnames=["scheme"] + fieldnames)
        writer.writeheader()
        for name, scheme_data in zip(scheme_names, all_scheme_data):
            row = {"scheme": name}
            for m in METRICS:
                row[m] = np.mean(scheme_data["metrics_values"][m])
            row["success_6A_ratio"] = scheme_data["success_ratio"]
            writer.writerow(row)

    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
