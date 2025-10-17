import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
from scipy.fft import fft, fftfreq

# --------------------------
# 1. 数据加载与参数设置
# --------------------------
def load_cir_data(file_path):
    """加载CIR数据（假设为CSV格式，每行是一个CIR序列）"""
    # 示例：若CIR数据保存在CSV中，每行代表一个时刻的CIR采样
    # 实际使用时根据数据格式修改（如二进制、txt等）
    cir_data = pd.read_csv(file_path, header=None).values
    return cir_data

# 假设参数（需根据实际数据修改）
fs_cir = 50  # CIR数据采样率（Hz），即每组CIR的间隔=1/fs_cir
T = 30  # 分析时长（秒），建议至少10秒以上
heart_rate_band = [0.8, 2.5]  # 心率频率范围（Hz）：0.8~2.5Hz对应48~150次/分钟
resp_rate_band = [0.2, 0.5]   # 呼吸频率范围（Hz）：0.2~0.5Hz对应12~30次/分钟

# 加载数据（替换为你的CIR文件路径）
# cir_data = load_cir_data("uwb_cir_data.csv")
# 若没有实际数据，用模拟数据测试（以下为模拟代码，实际使用时注释掉）
np.random.seed(42)
n_samples = int(fs_cir * T)  # 总采样数
n_bins = 50  # CIR的时域采样点数（距离仓）
t = np.linspace(0, T, n_samples, endpoint=False)
# 模拟胸部距离变化：呼吸（大振幅慢变）+心跳（小振幅快变）+噪声
resp_amp = 0.5  # 呼吸引起的距离变化（单位：CIR bins）
heart_amp = 0.05  # 心跳引起的距离变化（单位：CIR bins）
resp_freq = 0.3  # 呼吸频率（0.3Hz=18次/分钟）
heart_freq = 1.2  # 心跳频率（1.2Hz=72次/分钟）
true_distance_change = resp_amp * np.sin(2*np.pi*resp_freq*t) + heart_amp * np.sin(2*np.pi*heart_freq*t)
# 生成CIR数据：每个时刻的CIR在真实距离附近有一个峰值，叠加噪声
cir_data = np.zeros((n_samples, n_bins))
for i in range(n_samples):
    peak_bin = 20 + true_distance_change[i]  # 峰值位置随距离变化
    bin_idx = np.arange(n_bins)
    # 模拟CIR峰值（高斯形状）
    cir_data[i] = 0.5 * np.exp(-((bin_idx - peak_bin)/2)** 2) + 0.05 * np.random.randn(n_bins)

# --------------------------
# 2. 从CIR中提取距离变化（微动信号）
# --------------------------
def extract_distance_change(cir_data):
    """追踪CIR峰值位置，得到距离随时间的变化"""
    n_samples, n_bins = cir_data.shape
    distance = np.zeros(n_samples)  # 存储每个时刻的峰值位置（距离）
    
    for i in range(n_samples):
        # 找到CIR中的峰值（胸部反射对应的最强信号）
        peaks, _ = find_peaks(cir_data[i], height=0.1)  # 阈值根据实际信号调整
        if len(peaks) > 0:
            # 取最强峰值的位置（若有多个反射峰，需提前确定目标峰）
            peak_amp = cir_data[i, peaks]
            distance[i] = peaks[np.argmax(peak_amp)]
        else:
            # 若未检测到峰值，用前一时刻的值填充（避免断裂）
            distance[i] = distance[i-1] if i > 0 else 0
    
    # 去趋势（消除缓慢漂移，如人体整体移动）
    distance_detrend = distance - np.polyval(np.polyfit(t, distance, 1), t)
    return distance_detrend

# 提取距离变化信号
distance_change = extract_distance_change(cir_data)

# --------------------------
# 3. 滤波分离心跳信号（抑制呼吸和噪声）
# --------------------------
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """设计带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)  # 零相位滤波，避免相位偏移
    return y

# 步骤1：先用带通滤波保留呼吸+心跳信号（去除高频噪声）
resp_heart_signal = butter_bandpass_filter(
    distance_change, 
    lowcut=resp_rate_band[0], 
    highcut=heart_rate_band[1], 
    fs=fs_cir
)

# 步骤2：提取呼吸信号（用于后续抵消）
resp_signal = butter_bandpass_filter(
    resp_heart_signal, 
    lowcut=resp_rate_band[0], 
    highcut=resp_rate_band[1], 
    fs=fs_cir
)

# 步骤3：从呼吸+心跳信号中减去呼吸信号，得到纯净心跳信号
heart_signal = resp_heart_signal - resp_signal

# 步骤4：再次滤波，仅保留心跳频率范围
heart_signal = butter_bandpass_filter(
    heart_signal, 
    lowcut=heart_rate_band[0], 
    highcut=heart_rate_band[1], 
    fs=fs_cir
)

# --------------------------
# 4. 计算心率（频域分析）
# --------------------------
def calculate_heart_rate(signal, fs):
    """通过傅里叶变换计算信号的主频率（心率）"""
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]  # 频率轴（正频率部分）
    yf_abs = 2.0/n * np.abs(yf[:n//2])  # 幅度谱
    
    # 只关注心率频率范围内的峰值
    mask = (xf >= heart_rate_band[0]) & (xf <= heart_rate_band[1])
    xf_heart = xf[mask]
    yf_heart = yf_abs[mask]
    
    if len(yf_heart) == 0:
        return 0.0  # 无有效信号
    # 取最强峰值对应的频率
    heart_freq = xf_heart[np.argmax(yf_heart)]
    return heart_freq * 60  # 转换为次/分钟

# 计算心率
heart_rate = calculate_heart_rate(heart_signal, fs_cir)
print(f"估算心率：{heart_rate:.1f} 次/分钟")

# --------------------------
# 5. 结果可视化
# --------------------------
plt.figure(figsize=(12, 10))

# （1）原始CIR数据示例（第一个和最后一个时刻）
plt.subplot(5, 1, 1)
plt.plot(cir_data[0], label='初始时刻CIR')
plt.plot(cir_data[-1], label='最终时刻CIR')
plt.title('1. CIR数据示例')
plt.xlabel('距离仓（bins）')
plt.ylabel('幅度')
plt.legend()

# （2）提取的距离变化信号
plt.subplot(5, 1, 2)
plt.plot(t, distance_change)
plt.title('2. 胸部距离变化（原始）')
plt.xlabel('时间（s）')
plt.ylabel('距离变化（bins）')

# （3）呼吸+心跳混合信号
plt.subplot(5, 1, 3)
plt.plot(t, resp_heart_signal)
plt.title('3. 呼吸+心跳混合信号（滤波后）')
plt.xlabel('时间（s）')
plt.ylabel('幅度')

# （4）分离出的心跳信号
plt.subplot(5, 1, 4)
plt.plot(t, heart_signal)
plt.title('4. 分离后的心跳信号')
plt.xlabel('时间（s）')
plt.ylabel('幅度')

# （5）心率频谱
plt.subplot(5, 1, 5)
n = len(heart_signal)
yf = fft(heart_signal)
xf = fftfreq(n, 1/fs_cir)[:n//2]
yf_abs = 2.0/n * np.abs(yf[:n//2])
plt.plot(xf, yf_abs)
plt.xlim(heart_rate_band)
plt.axvline(x=heart_rate/60, color='r', linestyle='--', label=f'心率：{heart_rate:.1f}次/分钟')
plt.title('5. 心跳信号频谱')
plt.xlabel('频率（Hz）')
plt.ylabel('幅度')
plt.legend()

plt.tight_layout()
plt.show()