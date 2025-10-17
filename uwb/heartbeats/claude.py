import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

"""
主要功能说明：

CIR数据解析：处理16bit IQ复数数据
生命体征提取：从CIR中提取心跳引起的微小变化
带通滤波：提取心率频段信号（0.8-2.5 Hz，对应48-150 bpm）
FFT频谱分析：找到主频率峰值估计心率
置信度评估：评估检测结果的可靠性
使用建议：

数据收集时间：至少需要20-30秒的数据才能准确估计心率
采样间隔：50ms（20Hz）是较好的选择，太慢会丢失心跳信息
静止状态：被测者应保持相对静止，运动会影响准确性
距离：被测者应在雷达检测范围内（通常0.5-2米）
你可以根据实际的雷达硬件接口修改数据读取部分。需要帮助适配具体的硬件接口吗？
"""
class UWBHeartRateDetector:
    def __init__(self, sampling_interval_ms=50, window_size_seconds=30):
        """
        初始化UWB心率检测器
        
        参数:
        sampling_interval_ms: 采样间隔（毫秒）
        window_size_seconds: 分析窗口大小（秒）
        """
        self.sampling_interval = sampling_interval_ms / 1000  # 转换为秒
        self.fs = 1.0 / self.sampling_interval  # 采样频率
        self.window_size = window_size_seconds
        self.cir_data_buffer = []
        
    def parse_cir_data(self, iq_data):
        """
        解析IQ复数数据
        
        参数:
        iq_data: IQ复数数组，每个元素为16bit的I和Q值
        
        返回:
        complex_data: 复数数组
        """
        if isinstance(iq_data, np.ndarray) and iq_data.dtype == np.complex64:
            return iq_data
        
        # 如果是分离的I和Q值
        if len(iq_data.shape) == 2:
            I = iq_data[:, 0]
            Q = iq_data[:, 1]
            return I + 1j * Q
        
        return iq_data
    
    def extract_vital_sign(self, cir_sequence):
        """
        从CIR序列中提取生命体征信号
        
        参数:
        cir_sequence: CIR数据序列，shape=(时间帧数, IQ点数)
        
        返回:
        vital_signal: 提取的生命体征信号
        """
        # 方法1: 使用幅度变化
        amplitudes = np.abs(cir_sequence)
        
        # 找到主要反射峰（通常对应人体）
        # 计算每帧的平均幅度分布
        avg_amplitude = np.mean(amplitudes, axis=0)
        
        # 找到最强的几个峰值范围（排除近场噪声）
        peak_indices = signal.find_peaks(avg_amplitude, height=np.max(avg_amplitude)*0.3)[0]
        
        if len(peak_indices) == 0:
            # 如果没找到峰值，使用整体信号
            vital_signal = np.sum(amplitudes, axis=1)
        else:
            # 选择最强的峰值区域周围的数据
            main_peak_idx = peak_indices[np.argmax(avg_amplitude[peak_indices])]
            
            # 在主峰周围取一定范围
            range_start = max(0, main_peak_idx - 5)
            range_end = min(amplitudes.shape[1], main_peak_idx + 5)
            
            # 提取该区域的幅度变化
            vital_signal = np.sum(amplitudes[:, range_start:range_end], axis=1)
        
        # 方法2（可选）: 使用相位变化
        # phases = np.angle(cir_sequence)
        # phase_diff = np.diff(np.unwrap(phases, axis=0), axis=0)
        
        return vital_signal
    
    def bandpass_filter(self, data, lowcut=0.8, highcut=2.5):
        """
        带通滤波器，提取心率频段信号
        
        参数:
        data: 输入信号
        lowcut: 低频截止频率（Hz），对应48 bpm
        highcut: 高频截止频率（Hz），对应150 bpm
        
        返回:
        filtered_data: 滤波后的信号
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # 设计带通滤波器
        b, a = signal.butter(4, [low, high], btype='band')
        
        # 应用滤波器
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data
    
    def estimate_heart_rate(self, vital_signal):
        """
        从生命体征信号估计心率
        
        参数:
        vital_signal: 生命体征信号
        
        返回:
        heart_rate: 心率（bpm）
        confidence: 置信度
        """
        # 去除直流分量
        vital_signal = vital_signal - np.mean(vital_signal)
        
        # 带通滤波
        filtered_signal = self.bandpass_filter(vital_signal)
        
        # FFT分析
        N = len(filtered_signal)
        yf = fft(filtered_signal)
        xf = fftfreq(N, self.sampling_interval)
        
        # 只取正频率部分
        positive_freq_idx = xf > 0
        xf_positive = xf[positive_freq_idx]
        yf_positive = np.abs(yf[positive_freq_idx])
        
        # 限制在心率范围内（0.8-2.5 Hz，即48-150 bpm）
        hr_range_idx = (xf_positive >= 0.8) & (xf_positive <= 2.5)
        hr_freqs = xf_positive[hr_range_idx]
        hr_spectrum = yf_positive[hr_range_idx]
        
        if len(hr_spectrum) == 0:
            return 0, 0
        
        # 找到频谱峰值
        peak_idx = np.argmax(hr_spectrum)
        peak_freq = hr_freqs[peak_idx]
        
        # 转换为bpm
        heart_rate = peak_freq * 60
        
        # 计算置信度（基于峰值的显著性）
        mean_power = np.mean(hr_spectrum)
        peak_power = hr_spectrum[peak_idx]
        confidence = min(100, (peak_power / (mean_power + 1e-6)) * 10)
        
        return heart_rate, confidence
    
    def process_cir_sequence(self, cir_sequence, visualize=False):
        """
        处理CIR序列并估计心率
        
        参数:
        cir_sequence: CIR数据序列，可以是:
                     - numpy数组，shape=(时间帧数, IQ点数) 的复数数组
                     - list of numpy arrays
        visualize: 是否可视化结果
        
        返回:
        heart_rate: 估计的心率（bpm）
        confidence: 置信度（0-100）
        """
        # 转换为numpy数组
        if isinstance(cir_sequence, list):
            cir_sequence = np.array(cir_sequence)
        
        # 提取生命体征信号
        vital_signal = self.extract_vital_sign(cir_sequence)
        
        # 估计心率
        heart_rate, confidence = self.estimate_heart_rate(vital_signal)
        
        if visualize:
            self.visualize_results(cir_sequence, vital_signal, heart_rate, confidence)
        
        return heart_rate, confidence
    
    def visualize_results(self, cir_sequence, vital_signal, heart_rate, confidence):
        """
        可视化分析结果
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. CIR热图
        amplitudes = np.abs(cir_sequence)
        im = axes[0].imshow(amplitudes.T, aspect='auto', cmap='hot', 
                           extent=[0, len(amplitudes)*self.sampling_interval, 0, amplitudes.shape[1]])
        axes[0].set_xlabel('时间 (秒)')
        axes[0].set_ylabel('距离单元')
        axes[0].set_title('CIR幅度热图')
        plt.colorbar(im, ax=axes[0])
        
        # 2. 提取的生命体征信号
        time_axis = np.arange(len(vital_signal)) * self.sampling_interval
        axes[1].plot(time_axis, vital_signal)
        axes[1].set_xlabel('时间 (秒)')
        axes[1].set_ylabel('幅度')
        axes[1].set_title('提取的生命体征信号')
        axes[1].grid(True)
        
        # 3. 频谱
        filtered_signal = self.bandpass_filter(vital_signal - np.mean(vital_signal))
        N = len(filtered_signal)
        yf = fft(filtered_signal)
        xf = fftfreq(N, self.sampling_interval)
        
        positive_freq_idx = (xf > 0) & (xf <= 3)
        xf_positive = xf[positive_freq_idx] * 60  # 转换为bpm
        yf_positive = np.abs(yf[positive_freq_idx])
        
        axes[2].plot(xf_positive, yf_positive)
        axes[2].axvline(x=heart_rate, color='r', linestyle='--', 
                       label=f'心率: {heart_rate:.1f} bpm (置信度: {confidence:.1f}%)')
        axes[2].set_xlabel('心率 (bpm)')
        axes[2].set_ylabel('幅度')
        axes[2].set_title('频谱分析')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_xlim([40, 180])
        
        plt.tight_layout()
        plt.show()


# 示例使用
def generate_test_data(duration_seconds=30, sampling_interval_ms=50, num_iq_points=60, 
                       heart_rate_bpm=72, breathing_rate_bpm=15):
    """
    生成测试用的模拟CIR数据
    """
    fs = 1000 / sampling_interval_ms
    num_frames = int(duration_seconds * fs)
    
    # 时间轴
    t = np.arange(num_frames) / fs
    
    # 心率和呼吸率的频率（Hz）
    heart_freq = heart_rate_bpm / 60
    breathing_freq = breathing_rate_bpm / 60
    
    # 生成CIR数据
    cir_data = np.zeros((num_frames, num_iq_points), dtype=np.complex64)
    
    # 模拟人体位置的主峰（假设在第30个距离单元）
    target_range = 30
    
    for i in range(num_frames):
        # 基础CIR（静态反射）
        base_cir = np.random.randn(num_iq_points) + 1j * np.random.randn(num_iq_points)
        base_cir *= 0.1  # 噪声水平
        
        # 添加静态目标
        base_cir[target_range] += 1.0 + 0.1j
        
        # 添加心跳引起的相位调制
        heart_phase = 0.05 * np.sin(2 * np.pi * heart_freq * t[i])
        
        # 添加呼吸引起的相位调制（幅度更大）
        breathing_phase = 0.15 * np.sin(2 * np.pi * breathing_freq * t[i])
        
        # 总的相位调制
        total_phase = heart_phase + breathing_phase
        
        # 应用相位调制到目标反射
        base_cir[target_range] *= np.exp(1j * total_phase)
        
        cir_data[i] = base_cir
    
    return cir_data


# 主程序示例
if __name__ == "__main__":
    print("UWB雷达心率检测示例\n")
    
    # 参数设置
    SAMPLING_INTERVAL_MS = 50  # U秒，这里假设50ms
    DURATION_SECONDS = 30  # 分析30秒的数据
    
    # 创建检测器
    detector = UWBHeartRateDetector(
        sampling_interval_ms=SAMPLING_INTERVAL_MS,
        window_size_seconds=DURATION_SECONDS
    )
    
    print("生成测试数据...")
    # 生成测试数据（模拟真实场景，心率72 bpm）
    test_cir_data = generate_test_data(
        duration_seconds=DURATION_SECONDS,
        sampling_interval_ms=SAMPLING_INTERVAL_MS,
        num_iq_points=60,
        heart_rate_bpm=72,
        breathing_rate_bpm=15
    )
    
    print(f"CIR数据形状: {test_cir_data.shape}")
    print(f"采样频率: {detector.fs:.2f} Hz")
    print(f"总帧数: {len(test_cir_data)}\n")
    
    # 处理数据并估计心率
    print("分析心率...")
    heart_rate, confidence = detector.process_cir_sequence(
        test_cir_data, 
        visualize=True
    )
    
    print(f"\n估计心率: {heart_rate:.1f} bpm")
    print(f"置信度: {confidence:.1f}%")
    
    # 实际使用示例（注释掉的代码）
    """
    # 实际使用时，你的CIR数据可能来自雷达设备
    # 假设你有一个函数从雷达读取数据
    
    cir_buffer = []
    for _ in range(int(30 * detector.fs)):  # 收集30秒数据
        # 从雷达读取一帧CIR数据
        iq_data = read_cir_from_radar()  # 你需要实现这个函数
        cir_buffer.append(iq_data)
    
    # 转换为numpy数组并处理
    cir_sequence = np.array(cir_buffer)
    heart_rate, confidence = detector.process_cir_sequence(cir_sequence)
    print(f"心率: {heart_rate:.1f} bpm, 置信度: {confidence:.1f}%")
    """