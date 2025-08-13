import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.font_manager as fm

# 设置中文字体支持
try:
    # 尝试使用微软雅黑字体（Windows系统）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告：无法设置中文字体，图表中的中文可能无法正确显示")

# 生成理想的CIR数据
def generate_ideal_cir(num_samples=1000, snr_db=20):
    """
    生成理想的UWB CIR复数数据
    
    参数:
    num_samples: 样本数量
    snr_db: 信噪比，单位为dB
    
    返回:
    ideal_cir: 理想的CIR复数数据
    """
    # 创建空的CIR数据
    ideal_cir = np.zeros(num_samples, dtype=complex)
    
    # 主路径的时间延迟
    main_path_delay = int(num_samples * 0.3)
    
    # 生成主路径脉冲
    pulse_width = int(num_samples * 0.05)  # 脉冲宽度
    pulse = signal.gausspulse(np.linspace(-1, 1, pulse_width), fc=0.5)
    
    # 将脉冲放置在正确的延迟位置
    if main_path_delay + pulse_width <= num_samples:
        ideal_cir[main_path_delay:main_path_delay+pulse_width] = pulse * np.exp(1j * 0)  # 相位为0
    
    # 添加多径效应（较弱的反射信号）
    for j in range(2):  # 添加两个多径分量
        multipath_delay = main_path_delay + int(num_samples * 0.1 * (j+1))  # 多径延迟
        multipath_amplitude = 0.3 / (j+1)  # 多径幅度
        multipath_phase = np.random.uniform(-np.pi/4, np.pi/4)  # 多径相位
        
        if multipath_delay + pulse_width <= num_samples:
            ideal_cir[multipath_delay:multipath_delay+pulse_width] += multipath_amplitude * pulse * np.exp(1j * multipath_phase)
    
    # 计算信号功率
    signal_power = np.mean(np.abs(ideal_cir)**2)
    
    # 根据SNR计算噪声功率
    noise_power = signal_power / (10**(snr_db/10))
    noise_std = np.sqrt(noise_power/2)  # 复数噪声的标准差
    
    # 添加噪声
    noise_real = np.random.normal(0, noise_std, num_samples)
    noise_imag = np.random.normal(0, noise_std, num_samples)
    noisy_cir = ideal_cir + noise_real + 1j * noise_imag
    
    return noisy_cir

# 生成多个不同SNR的CIR数据
def generate_multiple_cir_samples(num_samples=1000, snr_levels=[30, 20, 10, 0]):
    """
    生成多个不同信噪比的CIR数据样本
    
    参数:
    num_samples: 每个CIR的样本数量
    snr_levels: 不同的信噪比水平列表，单位为dB
    
    返回:
    cir_samples: 包含多个CIR样本的列表
    """
    cir_samples = []
    for snr in snr_levels:
        cir = generate_ideal_cir(num_samples, snr)
        cir_samples.append(cir)
    
    return cir_samples

# 主函数
def main():
    # 生成不同SNR的CIR样本
    num_samples = 1000
    snr_levels = [30, 20, 10, 0]
    cir_samples = generate_multiple_cir_samples(num_samples, snr_levels)
    
    # 创建图形
    plt.figure(figsize=(15, 12))
    
    # 1. 时域图 - 显示不同SNR的CIR幅度
    plt.subplot(3, 2, 1)
    for i, (cir, snr) in enumerate(zip(cir_samples, snr_levels)):
        plt.plot(np.abs(cir), label=f'SNR = {snr} dB')
    plt.title('CIR幅度 - 时域')
    plt.xlabel('样本索引')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True)
    
    # 2. 时域图 - 显示不同SNR的CIR相位
    plt.subplot(3, 2, 2)
    for i, (cir, snr) in enumerate(zip(cir_samples, snr_levels)):
        plt.plot(np.angle(cir), label=f'SNR = {snr} dB')
    plt.title('CIR相位 - 时域')
    plt.xlabel('样本索引')
    plt.ylabel('相位 (弧度)')
    plt.legend()
    plt.grid(True)
    
    # 3. 散点图 - 显示高SNR的CIR复数数据
    plt.subplot(3, 2, 3)
    high_snr_cir = cir_samples[0]  # SNR = 30dB
    plt.scatter(high_snr_cir.real, high_snr_cir.imag, c=np.arange(len(high_snr_cir)), cmap='viridis', alpha=0.7)
    plt.title('CIR复数数据散点图 (SNR = 30dB)')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.grid(True)
    plt.axis('equal')  # 保持坐标轴比例相同
    plt.colorbar(label='样本索引')
    
    # 4. 散点图 - 显示低SNR的CIR复数数据
    plt.subplot(3, 2, 4)
    low_snr_cir = cir_samples[-1]  # SNR = 0dB
    plt.scatter(low_snr_cir.real, low_snr_cir.imag, c=np.arange(len(low_snr_cir)), cmap='viridis', alpha=0.7)
    plt.title('CIR复数数据散点图 (SNR = 0dB)')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.grid(True)
    plt.axis('equal')  # 保持坐标轴比例相同
    plt.colorbar(label='样本索引')
    
    # 5. 放大显示主路径和多径的散点图
    plt.subplot(3, 2, 5)
    # 找到主路径的位置
    main_path_idx = np.argmax(np.abs(high_snr_cir))
    # 选择主路径附近的样本
    window_size = 100
    start_idx = max(0, main_path_idx - window_size//2)
    end_idx = min(len(high_snr_cir), main_path_idx + window_size//2)
    
    # 提取主路径附近的数据
    path_data = high_snr_cir[start_idx:end_idx]
    path_indices = np.arange(start_idx, end_idx)
    
    # 绘制散点图
    scatter = plt.scatter(path_data.real, path_data.imag, c=path_indices, cmap='viridis', alpha=0.7, s=50)
    plt.title('主路径和多径的CIR散点图 (放大视图)')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.grid(True)
    plt.axis('equal')  # 保持坐标轴比例相同
    plt.colorbar(scatter, label='样本索引')
    
    # 6. 理论解释图
    plt.subplot(3, 2, 6)
    # 创建一个简单的示意图，说明CIR散点图的理论形态
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 绘制单位圆（理想情况下的主路径）
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    plt.plot(circle_x, circle_y, 'b--', label='理想主路径轨迹')
    
    # 绘制主路径点
    plt.scatter([1], [0], c='r', s=100, label='主路径峰值')
    
    # 绘制多径点
    plt.scatter([0.3, 0.15], [0.2, -0.1], c='g', s=80, label='多径分量')
    
    # 绘制噪声区域
    noise_x = np.random.normal(0, 0.05, 100)
    noise_y = np.random.normal(0, 0.05, 100)
    plt.scatter(noise_x, noise_y, c='gray', alpha=0.3, s=20, label='噪声')
    
    plt.title('CIR散点图的理论形态')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.grid(True)
    plt.axis('equal')  # 保持坐标轴比例相同
    plt.legend()
    
    # 添加总体说明文本
    plt.figtext(0.5, 0.01, """
    CIR散点图理论解释：
    1. 理想情况下，主路径信号在复平面上形成一个圆形轨迹，其半径代表信号强度
    2. 主路径峰值通常具有最大幅度，在高SNR条件下表现为复平面上的一个明确点
    3. 多径分量表现为较小幅度的点，通常分散在主路径周围
    4. 噪声在复平面上呈现为围绕原点的随机分布点
    5. 随着SNR降低，信号点与噪声点的区分变得更加困难
    6. 在实际应用中，CIR的峰值位置和相位信息用于定位和到达角计算
    """, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # 为底部的文本留出空间
    plt.show()

# 如果直接运行此脚本，则执行主函数
if __name__ == "__main__":
    main()