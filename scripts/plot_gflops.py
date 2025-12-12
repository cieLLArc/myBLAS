# plot_gflops.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 设置科研风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', 'build', 'matrix_gflops.csv')

# 检查文件是否存在
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV 文件未找到: {csv_path}")

df = pd.read_csv(csv_path)

# 确保列名正确
expected_columns = ['Label', 'N', 'GFLOPS']
if not all(col in df.columns for col in expected_columns):
    raise ValueError(f"CSV 列名应为: {expected_columns}, 实际为: {list(df.columns)}")

# 按 Label 分组绘图
fig, ax = plt.subplots(figsize=(10, 6))

# 获取所有唯一标签
labels = df['Label'].unique()

for label in labels:
    subset = df[df['Label'] == label]
    sizes = subset['N'].values
    gflops = subset['GFLOPS'].values

    ax.plot(sizes, gflops, 'o-', linewidth=2, markersize=8, label=label)

# 设置坐标轴
ax.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
ax.set_ylabel('GFLOPS', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')

# 设置刻度和网格
ax.set_xticks(sorted(df['N'].unique()))  # 确保 X 轴刻度是实际测试尺寸
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# 图例
ax.legend(loc='upper left', fontsize=10)

# 优化布局
plt.tight_layout()

# 设置保存路径：benchmark/plot/
save_dir = os.path.join(script_dir, '..', 'benchmark', 'plot')
os.makedirs(save_dir, exist_ok=True)

# 生成精确到分钟的时间戳
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
base_name = f"matrix_comp_gflops_{timestamp}"

png_path = os.path.join(save_dir, f"{base_name}.png")

# 保存图表
plt.savefig(png_path, dpi=300, bbox_inches='tight')

print(f"Files save to:\n   {png_path}\n")

# 显示图表（可选）
plt.show()
