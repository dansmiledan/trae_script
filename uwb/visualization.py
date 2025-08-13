import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from itertools import count
import random
import matplotlib.font_manager as fm

# 设置中文字体支持
try:
    # 尝试使用微软雅黑字体（Windows系统）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告：无法设置中文字体，图表中的中文可能无法正确显示")

# 创建图形和子图布局
plt.style.use('ggplot')
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(4, 2, figure=fig)

# 第一行占用1*2的空间（第一个子图）
ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('动态显示两组持续生成的数据')
ax1.set_xlabel('X轴')
ax1.set_ylabel('Y轴')

# 中间两行，每行两个子图
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('3组32个固定数据')
ax2.set_xlabel('索引')
ax2.set_ylabel('值')

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('第二个图的子集（16个点）')
ax3.set_xlabel('索引')
ax3.set_ylabel('值')

ax4 = fig.add_subplot(gs[2, 0])
ax4.set_title('3组32个固定数据（散点图）')
ax4.set_xlabel('X轴')
ax4.set_ylabel('Y轴')

ax5 = fig.add_subplot(gs[2, 1])
ax5.set_title('第四个图的子集（16个点）')
ax5.set_xlabel('X轴')
ax5.set_ylabel('Y轴')

# 最后一行占用1*2的空间（第六个子图）
ax6 = fig.add_subplot(gs[3, :])
ax6.set_title('动态显示两组持续生成的数据（不同来源）')
ax6.set_xlabel('X轴')
ax6.set_ylabel('Y轴')

# 调整布局
plt.tight_layout()

# 初始化数据
index = np.arange(32)

# 为第二个和第三个子图生成固定数据
data_2_1 = np.sin(np.linspace(0, 4*np.pi, 32)) * 3 + np.random.normal(0, 0.5, 32)
data_2_2 = np.cos(np.linspace(0, 4*np.pi, 32)) * 2 + np.random.normal(0, 0.5, 32)
data_2_3 = np.sin(np.linspace(0, 2*np.pi, 32)) * 4 + np.random.normal(0, 0.5, 32)

# 为第四个和第五个子图生成固定数据
data_4_x1 = np.random.rand(32) * 10
data_4_y1 = np.random.rand(32) * 10
data_4_x2 = np.random.rand(32) * 10
data_4_y2 = np.random.rand(32) * 10
data_4_x3 = np.random.rand(32) * 10
data_4_y3 = np.random.rand(32) * 10

# 选择子集（16个点）
subset_indices = sorted(random.sample(range(32), 16))

# 第一个子图的数据
x1_data, y1_data_1, y1_data_2 = [], [], []

# 第六个子图的数据
x6_data, y6_data_1, y6_data_2 = [], [], []

# 计数器
counter = count()

# 更新函数 - 用于动画
def update(frame):
    # 清除所有子图
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    
    # 设置标题
    ax1.set_title('动态显示两组持续生成的数据')
    ax2.set_title('3组32个固定数据')
    ax3.set_title('第二个图的子集（16个点）')
    ax4.set_title('3组32个固定数据（散点图）')
    ax5.set_title('第四个图的子集（16个点）')
    ax6.set_title('动态显示两组持续生成的数据（不同来源）')
    
    # 更新第一个子图数据（动态生成）
    x = next(counter)
    x1_data.append(x)
    y1_data_1.append(np.sin(x * 0.1) * 3 + np.random.normal(0, 0.5))
    y1_data_2.append(np.cos(x * 0.1) * 2 + np.random.normal(0, 0.5))
    
    # 限制数据点数量，保持动画流畅
    if len(x1_data) > 100:
        x1_data.pop(0)
        y1_data_1.pop(0)
        y1_data_2.pop(0)
    
    # 绘制第一个子图
    ax1.plot(x1_data, y1_data_1, 'r-', label='数据集1')
    ax1.plot(x1_data, y1_data_2, 'b-', label='数据集2')
    ax1.legend(loc='upper left')
    ax1.set_xlim(max(0, x - 50), x + 10)
    
    # 更新第二个子图数据（每次刷新所有32个点）
    new_data_2_1 = data_2_1 + np.random.normal(0, 0.2, 32)
    new_data_2_2 = data_2_2 + np.random.normal(0, 0.2, 32)
    new_data_2_3 = data_2_3 + np.random.normal(0, 0.2, 32)
    
    # 绘制第二个子图
    ax2.plot(index, new_data_2_1, 'r-', label='数据集1')
    ax2.plot(index, new_data_2_2, 'g-', label='数据集2')
    ax2.plot(index, new_data_2_3, 'b-', label='数据集3')
    ax2.legend(loc='upper right')
    
    # 绘制第三个子图（第二个的子集）
    ax3.plot(subset_indices, new_data_2_1[subset_indices], 'ro-', label='数据集1子集')
    ax3.plot(subset_indices, new_data_2_2[subset_indices], 'go-', label='数据集2子集')
    ax3.plot(subset_indices, new_data_2_3[subset_indices], 'bo-', label='数据集3子集')
    ax3.legend(loc='upper right')
    
    # 更新第四个子图数据（散点图）
    new_data_4_x1 = data_4_x1 + np.random.normal(0, 0.2, 32)
    new_data_4_y1 = data_4_y1 + np.random.normal(0, 0.2, 32)
    new_data_4_x2 = data_4_x2 + np.random.normal(0, 0.2, 32)
    new_data_4_y2 = data_4_y2 + np.random.normal(0, 0.2, 32)
    new_data_4_x3 = data_4_x3 + np.random.normal(0, 0.2, 32)
    new_data_4_y3 = data_4_y3 + np.random.normal(0, 0.2, 32)
    
    # 绘制第四个子图
    ax4.scatter(new_data_4_x1, new_data_4_y1, c='r', label='数据集1')
    ax4.scatter(new_data_4_x2, new_data_4_y2, c='g', label='数据集2')
    ax4.scatter(new_data_4_x3, new_data_4_y3, c='b', label='数据集3')
    ax4.legend(loc='upper right')
    ax4.set_xlim(0, 11)
    ax4.set_ylim(0, 11)
    
    # 绘制第五个子图（第四个的子集）
    ax5.scatter(new_data_4_x1[subset_indices], new_data_4_y1[subset_indices], c='r', label='数据集1子集')
    ax5.scatter(new_data_4_x2[subset_indices], new_data_4_y2[subset_indices], c='g', label='数据集2子集')
    ax5.scatter(new_data_4_x3[subset_indices], new_data_4_y3[subset_indices], c='b', label='数据集3子集')
    ax5.legend(loc='upper right')
    ax5.set_xlim(0, 11)
    ax5.set_ylim(0, 11)
    
    # 更新第六个子图数据（动态生成，不同来源）
    x6_data.append(x)
    y6_data_1.append(np.sin(x * 0.05) * 2 + np.cos(x * 0.1) * 1.5 + np.random.normal(0, 0.5))
    y6_data_2.append(np.cos(x * 0.05) * 3 + np.sin(x * 0.1) * 1.2 + np.random.normal(0, 0.5))
    
    # 限制数据点数量
    if len(x6_data) > 100:
        x6_data.pop(0)
        y6_data_1.pop(0)
        y6_data_2.pop(0)
    
    # 绘制第六个子图
    ax6.plot(x6_data, y6_data_1, 'g-', label='数据集1')
    ax6.plot(x6_data, y6_data_2, 'm-', label='数据集2')
    ax6.legend(loc='upper left')
    ax6.set_xlim(max(0, x - 50), x + 10)

# 创建动画
ani = animation.FuncAnimation(fig, update, interval=100)

# 显示图形
plt.show()