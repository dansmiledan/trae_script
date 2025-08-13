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

# 生成模拟数据
def generate_sample_data(num_samples=1000):
    """
    生成模拟的UWB CIR复数数据和通信相位数据
    
    参数:
    num_samples: 样本数量
    
    返回:
    cir_data: 三个天线的CIR复数数据，形状为(3, num_samples)
    comm_phase: 三个天线的通信相位数据，形状为(3,)
    true_aoa: 真实的到达角（用于验证），单位为度
    """
    # 设置真实的到达角（方位角和仰角）
    true_azimuth = 30  # 方位角，单位：度
    true_elevation = 15  # 仰角，单位：度
    true_aoa = (true_azimuth, true_elevation)
    
    # 天线阵列参数
    wavelength = 0.0461  # UWB波长（对应6.5GHz频率），单位：米
    antenna_spacing = wavelength / 2  # 天线间距，通常为半波长
    
    # 天线位置（假设三角形阵列）
    antenna_positions = np.array([
        [0, 0, 0],  # 天线1（参考天线）
        [antenna_spacing, 0, 0],  # 天线2
        [antenna_spacing/2, antenna_spacing*np.sqrt(3)/2, 0]  # 天线3
    ])
    
    # 将角度转换为弧度
    azimuth_rad = np.radians(true_azimuth)
    elevation_rad = np.radians(true_elevation)
    
    # 计算入射信号的方向向量
    direction_vector = np.array([
        np.cos(elevation_rad) * np.cos(azimuth_rad),
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.sin(elevation_rad)
    ])
    
    # 计算理论相位差
    phase_differences = 2 * np.pi * np.dot(antenna_positions, direction_vector) / wavelength
    
    # 生成通信相位（加入一些随机噪声）
    comm_phase = phase_differences + np.random.normal(0, 0.1, 3)
    
    # 生成CIR数据
    cir_data = np.zeros((3, num_samples), dtype=complex)
    
    # 为每个天线生成CIR
    for i in range(3):
        # 主路径的时间延迟（基于天线位置和信号方向）
        main_path_delay = int(num_samples * 0.3) + int(phase_differences[i] * num_samples / (2 * np.pi))
        
        # 信号幅度（随距离衰减）
        amplitude = 1.0 - 0.1 * i
        
        # 生成主路径脉冲
        pulse_width = int(num_samples * 0.05)  # 脉冲宽度
        pulse = signal.gausspulse(np.linspace(-1, 1, pulse_width), fc=0.5)
        
        # 将脉冲放置在正确的延迟位置
        if main_path_delay + pulse_width <= num_samples:
            cir_data[i, main_path_delay:main_path_delay+pulse_width] = amplitude * pulse * np.exp(1j * comm_phase[i])
        
        # 添加多径效应（较弱的反射信号）
        for j in range(2):  # 添加两个多径分量
            multipath_delay = main_path_delay + int(num_samples * 0.1 * (j+1))  # 多径延迟
            multipath_amplitude = amplitude * 0.3 / (j+1)  # 多径幅度
            multipath_phase = comm_phase[i] + np.random.uniform(-np.pi/4, np.pi/4)  # 多径相位
            
            if multipath_delay + pulse_width <= num_samples:
                cir_data[i, multipath_delay:multipath_delay+pulse_width] += multipath_amplitude * pulse * np.exp(1j * multipath_phase)
        
        # 添加噪声
        noise_real = np.random.normal(0, 0.05, num_samples)
        noise_imag = np.random.normal(0, 0.05, num_samples)
        cir_data[i, :] += noise_real + 1j * noise_imag
    
    return cir_data, comm_phase, true_aoa

# 计算到达角（AOA）
def calculate_aoa(cir_data, comm_phase):
    """
    使用CIR数据和通信相位计算到达角
    
    参数:
    cir_data: 三个天线的CIR复数数据，形状为(3, num_samples)
    comm_phase: 三个天线的通信相位数据，形状为(3,)
    
    返回:
    estimated_azimuth: 估计的方位角，单位为度
    estimated_elevation: 估计的仰角，单位为度
    """
    # 天线阵列参数
    wavelength = 0.0461  # UWB波长（对应6.5GHz频率），单位：米
    antenna_spacing = wavelength / 2  # 天线间距
    
    # 天线位置（假设三角形阵列）
    antenna_positions = np.array([
        [0, 0, 0],  # 天线1（参考天线）
        [antenna_spacing, 0, 0],  # 天线2
        [antenna_spacing/2, antenna_spacing*np.sqrt(3)/2, 0]  # 天线3
    ])
    
    # 找到每个天线CIR的峰值位置和相位
    peak_phases = np.zeros(3)
    for i in range(3):
        # 找到CIR幅度的峰值位置
        peak_idx = np.argmax(np.abs(cir_data[i, :]))
        # 提取峰值处的相位
        peak_phases[i] = np.angle(cir_data[i, peak_idx])
    
    # 计算相位差（相对于参考天线）
    phase_differences = peak_phases - peak_phases[0]
    phase_differences = np.mod(phase_differences + np.pi, 2 * np.pi) - np.pi  # 将相位差限制在[-π, π]范围内
    
    # 结合通信相位进行校正
    corrected_phases = phase_differences + (comm_phase - comm_phase[0])
    corrected_phases = np.mod(corrected_phases + np.pi, 2 * np.pi) - np.pi
    
    # 使用最小二乘法估计方向向量
    # 构建方程组 A * direction = phase_differences * wavelength / (2*pi)
    A = antenna_positions[1:] - antenna_positions[0]
    b = corrected_phases[1:] * wavelength / (2 * np.pi)
    
    # 求解方向向量（仅x和y分量）
    direction_xy, residuals, rank, s = np.linalg.lstsq(A[:, :2], b, rcond=None)
    
    # 计算z分量（假设方向向量的模为1）
    direction_xy_norm = np.linalg.norm(direction_xy)
    if direction_xy_norm <= 1:
        direction_z = np.sqrt(1 - direction_xy_norm**2)
    else:
        # 如果xy分量的模大于1（由于噪声），则归一化并假设z为0
        direction_xy = direction_xy / direction_xy_norm
        direction_z = 0
    
    # 构建完整的方向向量
    direction_vector = np.append(direction_xy, direction_z)
    
    # 计算方位角和仰角
    estimated_azimuth = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
    estimated_elevation = np.degrees(np.arcsin(direction_vector[2]))
    
    # 确保方位角在[0, 360)范围内
    if estimated_azimuth < 0:
        estimated_azimuth += 360
    
    return estimated_azimuth, estimated_elevation

# 主函数
def main():
    # 生成样本数据
    cir_data, comm_phase, true_aoa = generate_sample_data(num_samples=1000)
    
    # 计算AOA
    estimated_azimuth, estimated_elevation = calculate_aoa(cir_data, comm_phase)
    
    # 打印结果
    print(f"真实方位角: {true_aoa[0]}°, 估计方位角: {estimated_azimuth:.2f}°")
    print(f"真实仰角: {true_aoa[1]}°, 估计仰角: {estimated_elevation:.2f}°")
    
    # 可视化CIR数据
    plt.figure(figsize=(15, 10))
    
    # 绘制三个天线的CIR幅度
    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(np.abs(cir_data[i]), label=f'天线 {i+1}')
    plt.title('三个天线的CIR幅度')
    plt.xlabel('样本索引')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True)
    
    # 绘制三个天线的CIR相位
    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(np.angle(cir_data[i]), label=f'天线 {i+1}')
    plt.title('三个天线的CIR相位')
    plt.xlabel('样本索引')
    plt.ylabel('相位 (弧度)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 如果直接运行此脚本，则执行主函数
if __name__ == "__main__":
    main()