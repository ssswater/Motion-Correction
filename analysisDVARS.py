import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_rel

# 原始数据
cases = ["case2", "case4", "case6", "case7", "case8", "case9", "case10"]
before = np.array([19.0482, 21.5206, 27.9183, 23.1200, 27.9019, 33.1302, 46.7868])
after = np.array([15.7722, 13.1720, 20.0521, 10.6827, 18.5466, 24.2280, 36.0987])

# 计算均值和标准差
mean_before, std_before = np.mean(before), np.std(before, ddof=1)
mean_after, std_after = np.mean(after), np.std(after, ddof=1)

# 计算配对 t 检验的 p 值
t_stat, p_value = ttest_rel(before, after)

# 打印结果
print(f"Before correction: Mean = {mean_before:.2f}, Std = {std_before:.2f}")
print(f"After correction: Mean = {mean_after:.2f}, Std = {std_after:.2f}")
print(f"Paired t-test t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

# 准备数据用于绘图
df = pd.DataFrame({"case": cases, "before": before, "after": after})
df_melted = df.melt(id_vars=["case"], var_name="Condition", value_name="DVARS")

# 绘制箱线图
plt.figure(figsize=(5, 6))
sns.boxplot(x="Condition", y="DVARS", data=df_melted, width=0.4, 
            boxprops=dict(facecolor="none", edgecolor="blue", linewidth=1.5),  # 空心箱体
            medianprops=dict(color="red"), 
            whiskerprops=dict(color="black",linestyle="--" ),  # 须线
            showfliers=False, whis=3.0)
sns.stripplot(x="Condition", y="DVARS", data=df_melted, color="black", jitter=True)


# # 添加显著性标记
# y_max = max(np.max(before), np.max(after))  # 找到最大值以放置显著性标记
# plt.plot([0, 0, 1, 1], [y_max + 2, y_max + 4, y_max + 4, y_max + 2], color="black")
# plt.text(0.5, y_max + 5, "*" if p_value < 0.05 else "n.s.", ha="center", fontsize=14)

# 设置标题和坐标轴
plt.ylabel("DVARS")
plt.xlabel("")
plt.xticks([0, 1], ["Before motion correction", "After motion correction"])

# 显示图形
plt.savefig('./DVARS.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
