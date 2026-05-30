import re
import os
import matplotlib.pyplot as plt

# 尝试导入 docx 库
try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("提示：未安装 python-docx，将只支持 .txt 文件。建议运行: pip install python-docx")


def read_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)


def read_log(file_path):
    """读取日志文件（支持 .txt 或 .docx），返回 test_mse 列表"""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if file_path.lower().endswith('.docx'):
        if not HAS_DOCX:
            raise RuntimeError("需要安装 python-docx 来读取 .docx 文件")

        text = read_text_from_docx(file_path)

    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

    pattern = r'test_mse\s*=\s*([0-9.]+)'
    matches = re.findall(pattern, text)

    values = [float(v) for v in matches]

    print(f"从 {os.path.basename(file_path)} 中提取到 {len(values)} 个 test_mse 值")

    if len(values) == 0:
        raise ValueError("未找到任何 test_mse 值")

    return values


def main():

    # =========================
    # 文件名
    # =========================
    file1 = "Random Sampling.docx"
    file2 = "Facility Location.docx"
    file3 = "K-Means Representative.docx"

    ourwork_file = "K-Means Farthest.docx"
    kfar_file = "K-Means Nearest.docx"
    result_file = "Temporal Variation .docx"

    # =========================
    # 读取数据
    # =========================
    mse1 = read_log(file1)
    mse2 = read_log(file2)
    mse3 = read_log(file3)

    try:
        mse4 = read_log(ourwork_file)
    except FileNotFoundError:
        print(f"警告：文件 {ourwork_file} 不存在，将跳过。")
        mse4 = []

    try:
        mse5 = read_log(kfar_file)
    except FileNotFoundError:
        print(f"警告：文件 {kfar_file} 不存在，将跳过。")
        mse5 = []

    try:
        mse6 = read_log(result_file)
    except FileNotFoundError:
        print(f"警告：文件 {result_file} 不存在，将跳过。")
        mse6 = []

    # =========================
    # ⭐ 方法名称修改区（已按要求更新）
    # =========================
    names = [
        "Random Sampling",
        "Facility Location Coreset",
        "K-Means Cluster Representative Selection"
    ]

    ms_list = [mse1, mse2, mse3]

    if mse4:
        names.append("K-Means Farthest-to-Center Selection")
        ms_list.append(mse4)

    if mse5:
        names.append("K-Means Nearest-to-Center Selection")
        ms_list.append(mse5)

    if mse6:
        names.append("Temporal Variation")
        ms_list.append(mse6)

    # =========================
    # 统计 & 柱状图数据收集
    # =========================
    print("\n========== 统计结果 ==========")

    start_epoch = 200
    end_epoch = 500

    bar_names = []
    bar_means = []
    bar_mins = []
    bar_maxs = []

    random_all = {}
    random_200 = {}

    for name, mse in zip(names, ms_list):

        if not mse:
            continue

        mean_val = sum(mse) / len(mse)
        best_val = min(mse)
        worst_val = max(mse)

        if len(mse) >= start_epoch:
            mse_200_500 = mse[start_epoch - 1 : min(end_epoch, len(mse))]
            mean_200_500 = sum(mse_200_500) / len(mse_200_500)
            best_200_500 = min(mse_200_500)
            worst_200_500 = max(mse_200_500)
        else:
            mse_200_500 = []

        if name == "Random Sampling":
            random_all = {"mean": mean_val, "best": best_val, "worst": worst_val}
            if mse_200_500:
                random_200 = {"mean": mean_200_500, "best": best_200_500, "worst": worst_200_500}

        def pct_str(val, ref):
            if ref is not None:
                return f" ({((val - ref) / ref) * 100:+.1f}%)"
            return ""

        print(f"\n{name}:")
        print(f"  总轮数: {len(mse)}")

        print("  [全部轮次]")
        print(f"    平均 test MSE: {mean_val:.6f}{pct_str(mean_val, random_all.get('mean'))}")
        print(f"    最佳 test MSE: {best_val:.6f}{pct_str(best_val, random_all.get('best'))}")
        print(f"    最差 test MSE: {worst_val:.6f}{pct_str(worst_val, random_all.get('worst'))}")

        if mse_200_500:
            print("  [200-500轮]")
            print(f"    轮次数: {len(mse_200_500)}")
            print(f"    平均 test MSE: {mean_200_500:.6f}{pct_str(mean_200_500, random_200.get('mean'))}")
            print(f"    最佳 test MSE: {best_200_500:.6f}{pct_str(best_200_500, random_200.get('best'))}")
            print(f"    最差 test MSE: {worst_200_500:.6f}{pct_str(worst_200_500, random_200.get('worst'))}")

            if name in ("Random Sampling", "Facility Location Coreset", "K-Means Cluster Representative Selection"):
                bar_display_name = name.replace("K-Means Cluster Representative Selection",
                                                 "K-Means Cluster\nRepresentative Selection")
                bar_names.append(bar_display_name)
                bar_means.append(mean_200_500)
                bar_mins.append(best_200_500)
                bar_maxs.append(worst_200_500)
        else:
            print("  [200-500轮] 数据不足")

    # =========================
    # 绘图
    # =========================
    lengths = [len(m) for m in ms_list if m]

    if not lengths:
        print("没有有效数据可绘图")
        return

    min_len = min(lengths)
    epochs = range(1, min_len + 1)

    # 全局字体设置：加粗 + 加大
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titleweight': 'bold',
    })

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', '*', 'x']
    linestyles = ['-', '-', '-', '-', '-', '-']

    # =========================
    # 折线图 1: 除 Temporal Variation 外的 5 条线
    # =========================
    plt.figure(figsize=(12, 6))

    for idx, (name, mse) in enumerate(zip(names, ms_list)):

        if not mse or name == "Temporal Variation":
            continue

        mse_plot = mse[:min_len]

        plt.plot(
            epochs,
            mse_plot,
            label=name,
            marker=markers[idx],
            markersize=3,
            linewidth=1,
            color=colors[idx],
            linestyle=linestyles[idx]
        )

    plt.xlabel('Epoch')
    plt.ylabel('Test MSE')
    plt.title('Test MSE Comparison')
    leg = plt.legend()
    for handle in leg.legend_handles:
        handle.set_linewidth(2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('line_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =========================
    # 折线图 2: 仅 Temporal Variation
    # =========================
    for idx, (name, mse) in enumerate(zip(names, ms_list)):
        if not mse or name != "Temporal Variation":
            continue

        plt.figure(figsize=(12, 6))
        mse_plot = mse[:min_len]

        plt.plot(
            epochs,
            mse_plot,
            label=name,
            marker=markers[idx],
            markersize=3,
            linewidth=1,
            color=colors[idx],
            linestyle=linestyles[idx]
        )

        plt.xlabel('Epoch')
        plt.ylabel('Test MSE')
        plt.title(name)
        leg = plt.legend()
        for handle in leg.legend_handles:
            handle.set_linewidth(2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('line_chart_temporal.png', dpi=300, bbox_inches='tight')
        plt.show()

    # =========================
    # 柱状图 (200-500轮均值 + 上下界)
    # =========================
    if bar_names:
        fig, ax = plt.subplots(figsize=(12, 7))

        x = range(len(bar_names))
        bar_colors = colors[:len(bar_names)]

        bars = ax.bar(x, bar_means, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.8)

        lower_err = [bar_means[i] - bar_mins[i] for i in range(len(bar_names))]
        upper_err = [bar_maxs[i] - bar_means[i] for i in range(len(bar_names))]
        yerr = [lower_err, upper_err]

        ax.errorbar(x, bar_means, yerr=yerr, fmt='none', ecolor='black',
                     capsize=8, capthick=1.5, elinewidth=1.5)

        for i, (mean_val, min_val, max_val) in enumerate(zip(bar_means, bar_mins, bar_maxs)):
            ax.text(i, max_val + 0.0002, f'Max={max_val:.4f}',
                    ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
            ax.text(i, min_val - 0.0002, f'Min={min_val:.4f}',
                    ha='center', va='top', fontsize=10, color='red', fontweight='bold')
            ax.text(i, mean_val, f'{mean_val:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(bar_names, rotation=0, ha='center', fontsize=12)
        ax.set_ylabel('Test MSE (200-500 Epochs)')
        ax.set_title('Mean Test MSE (Epochs 200-500) with Min-Max Range')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    main()