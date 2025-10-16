import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq
from uwb_hrp import get_hrp_codes, get_sfd
from uwb_rs import hrpRS

class LRWPANHRPConfig:
    """UWB波形配置类，模拟MATLAB的lrwpanHRPConfig对象"""
    def __init__(self):
        self.Mode = "HPRF"  # 可选: 'HPRF', 'BPRF', '802.15.4a'
        self.DataRateNum = 6810  # 数据速率(bps)
        self.PSDULength = 7  # PSDU长度(字节)
        self.ConstraintLength = 3  # 卷积码约束长度
        self.ConvolutionalCoding = True  # 是否启用卷积编码
        self.PreambleDuration = 64  # 前导码重复次数
        self.PreambleSpreadingFactor = 4  # 前导码扩频因子
        self.CodeIndex = 9  # 扩频码索引
        self.SamplesPerPulse = 4  # 每个脉冲的采样点数
        self.Ranging = False  # 是否启用测距
        self.STSPacketConfiguration = 0  # STS配置(0-3)
        self.NumSTSSegments = 1  # STS段数
        self.STSSegmentLength = 1  # 每段STS长度(512芯片单位)
        self.MeanPRFNum = 124.8  # 平均脉冲重复频率(MHz)
        self.SFDNumber = 0


def lrwpan_hrp_waveform_generator(psdu, cfg):
    """生成IEEE 802.15.4a/z HRP UWB波形"""
    # 1. 输入验证
    validate_config(cfg)
    psdu_len = len(psdu)
    max_len = 8 * (2**12 - 1) if cfg.Mode == "HPRF" else 8 * (2**7 - 1)
    if psdu_len > max_len:
        raise ValueError(f"PSDU长度超出最大值{max_len}字节")
    if psdu_len != 8 * cfg.PSDULength:
        print(f"警告: PSDU长度({psdu_len})与配置({8*cfg.PSDULength})不匹配，将截断")
        psdu_len = min(psdu_len, 8 * cfg.PSDULength)
        psdu = psdu[:psdu_len]

    # 2. Reed-Solomon编码(简化版)
    if cfg.Mode != "HPRF" or cfg.ConstraintLength != 7:
        rs_psdu = rs_encode(psdu)
    else:
        rs_psdu = psdu
    print(f"rs_psdu {len(rs_psdu)} {rs_psdu}")
    # 3. PHR生成与SECDED编码
    phr = create_phr_with_secded(psdu_len, cfg)
    print(f"phr {len(phr)} {phr}")
    # 4. 卷积编码
    convol_cw = convolutional_encode(phr, rs_psdu, cfg)
    print(f"convol_cw {len(convol_cw)} {convol_cw}")
    # 5. 符号映射(调制)
    symbols = symbol_mapper(convol_cw, cfg)
    print(f"symbols {len(symbols)} {symbols}")
    # 6. 前导码插入(SHR)
    shr = create_shr(cfg)
    symbols = np.concatenate([shr, symbols])

    # 7. STS序列插入(可选)
    if cfg.Mode != "802.15.4a" and cfg.STSPacketConfiguration != 0:
        sts = create_sts(cfg)
        if cfg.STSPacketConfiguration == 1:
            symbols = np.concatenate([shr, sts, symbols])
        elif cfg.STSPacketConfiguration == 2:
            symbols = np.concatenate([shr, symbols, sts])
        elif cfg.STSPacketConfiguration == 3:
            symbols = np.concatenate([shr, sts])

    # 8. 脉冲成形滤波
    wave = butterworth_filter(symbols, cfg.SamplesPerPulse)
    return wave, symbols, cfg  # 返回配置用于绘图


# 辅助函数保持不变（省略，与之前相同）
def validate_config(cfg):
    if cfg.SamplesPerPulse == 1:
        raise ValueError("SamplesPerPulse不能为1")
    if cfg.Mode not in ["HPRF", "BPRF", "802.15.4a"]:
        raise ValueError("Mode必须为'HPRF'/'BPRF'/'802.15.4a'")

def rs_encode(data):
    return hrpRS(data, do_encode=True)

def create_phr_with_secded(psdu_len, cfg):
    phr = np.zeros(13, dtype=int)
    if cfg.Mode != "HPRF":
        if cfg.DataRateNum == 110:
            phr[:2] = [0, 0]
        elif cfg.DataRateNum == 850:
            phr[:2] = [0, 1]
        elif cfg.DataRateNum in [6810, 1700]:
            phr[:2] = [1, 0]
        else:
            phr[:2] = [1, 1]
        len_bytes = psdu_len // 8
        phr[2:9] = np.array([(len_bytes >> i) & 1 for i in range(6, -1, -1)])
        phr[9] = 1 if cfg.Ranging else 0
        phr[10] = 0
        preamble_map = {16: [0,0], 64: [0,1], 1024: [1,0], 4096: [1,1]}
        phr[11:13] = preamble_map[cfg.PreambleDuration]
    else:
        len_bytes = psdu_len // 8
        phr[0] = (len_bytes >> 11) & 1
        phr[1] = (len_bytes >> 10) & 1
        phr[2:12] = np.array([(len_bytes >> i) & 1 for i in range(9, -1, -1)])
        phr[12] = 1 if cfg.Ranging else 0
    secded_parity = np.zeros(6, dtype=int)
    phr = np.concatenate([phr, secded_parity])
    phr[18] = np.mod(np.sum(phr[[1, 0, 8, 6, 4, 3, 10, 11]]), 2)
    phr[17] = np.mod(np.sum(phr[[0, 6, 5, 3, 2, 9, 10, 12]]), 2)
    phr[16] = np.mod(np.sum(phr[[1, 8, 7, 3, 2, 9, 10]]), 2)
    phr[15] = np.mod(np.sum(phr[[8, 7, 6, 5, 4, 9, 10]]), 2)
    phr[14] = np.mod(np.sum(phr[[12, 11]]), 2)
    phr[13] = np.mod(np.sum(phr), 2)
    return phr


def convenc3(input_bits):
    # 生成多项式：g0 = [010]_2（对应系数：当前输入×1，延迟1位×0，延迟2位×0）
    #            g1 = [101]_2（对应系数：当前输入×1，延迟1位×0，延迟2位×1）
    g0 = [0, 1, 0]  # 注意：这里调整了系数顺序，[当前输入系数, 延迟1位系数, 延迟2位系数]
    g1 = [1, 0, 1]
    
    state = [0, 0]  # 延迟单元：state[0]是延迟1位，state[1]是延迟2位
    encoded_bits = []

    for bit in input_bits:
        # 系统比特：根据g0计算（当前输入×g0[0] + 延迟1位×g0[1] + 延迟2位×g0[2]）
        systematic_bit = (bit & g0[0]) ^ (state[0] & g0[1]) ^ (state[1] & g0[2])
        # 校验比特：根据g1计算
        parity_bit = (bit & g1[0]) ^ (state[0] & g1[1]) ^ (state[1] & g1[2])
        
        encoded_bits.append(systematic_bit)
        encoded_bits.append(parity_bit)

        # 更新延迟单元（移位：新输入→延迟1位，原延迟1位→延迟2位）
        state[1] = state[0]
        state[0] = bit

    return np.array(encoded_bits)

def convolutional_encode(phr, rs_psdu, cfg):
    tail = np.zeros(cfg.ConstraintLength - 1, dtype=int)
    if cfg.ConvolutionalCoding:
        if cfg.Mode != "HPRF" or cfg.ConstraintLength == 3:
            convol_in = np.concatenate([phr, rs_psdu, tail])
        else:
            convol_in = np.concatenate([phr, tail, rs_psdu, tail])
    else:
        convol_in = np.concatenate([phr, tail])
    print(f"convol_in {len(convol_in)} {convol_in}")

    if not (cfg.Mode == "HPRF" and cfg.ConstraintLength == 7):
        convol_cw = convenc3(convol_in)
    else:
        pass

    if not cfg.ConvolutionalCoding:
        convol_cw = np.concatenate([convol_cw, rs_psdu])

    return convol_cw


def symbol_mapper(convol_cw, cfg):
    cws = np.reshape(convol_cw, (-1, 2))
    print(cws)
    num_sym = cws.shape[0]

    def generate_pn(initial_conditions, length):
        state = initial_conditions.copy()
        pn_seq = []
        for _ in range(length):
            out = state[-1]
            pn_seq.append(out)
            feedback = (state[0] + state[1]) % 2
            state = np.roll(state, 1)
            state[0] = feedback
        return np.array(pn_seq)

    code = get_hrp_codes(cfg.CodeIndex)
    code_hat = code[code != 0][:15]
    code_hat[code_hat == -1] = 0
    initial_conditions = np.flip(code_hat)
    print(f"code_hat {len(code_hat)} {code_hat}")
    print(f"initial_conditions {len(initial_conditions)} {initial_conditions}")
    '''
    s.PeakPRF,                           499.2
    s.BurstsPerSymbol,                   8
    s.NumHopBursts,                      2
    s.ChipsPerBurst,                     8
    s.ChipsPerSymbol,                    64
    s.ConvolutionalCoding,               1
    s.PreambleCodeLength,                127
    s.PreambleSpreadingFactor            4
    '''
    pn_length = num_sym * cfg.SamplesPerPulse
    pn_seq = generate_pn(initial_conditions, pn_length)

    symbols = []
    for i in range(num_sym):
        sys_bit = cws[i, 0]
        parity_bit = cws[i, 1]
        spread_seq = pn_seq[i*16 : (i+1)*16]
        symbol = (1 - 2 * parity_bit) * (1 - 2 * spread_seq)
        symbols.extend(symbol)
    return np.array(symbols)

# checked
def create_shr(cfg):
    code = get_hrp_codes(cfg.CodeIndex)
    L = cfg.PreambleSpreadingFactor
    spread = np.zeros(L * len(code), dtype=int)
    spread[::L] = code
    sync = np.tile(spread, cfg.PreambleDuration)
    sfd_code = get_sfd(cfg.Mode, cfg.SFDNumber, cfg.DataRateNum)
    sfd_expand = np.repeat(sfd_code, np.size(spread))
    spread_expand = np.tile(spread, np.size(sfd_code))
    sfd = sfd_expand * spread_expand
    return np.concatenate([sync, sfd])

def create_sts(cfg):
    gap = np.zeros(512, dtype=int)
    sts = gap.copy()
    if cfg.Mode == "HPRF" and cfg.STSPacketConfiguration == 2:
        sts = np.concatenate([sts, np.zeros(4 * cfg.STSSegmentLength, dtype=int)])
    try:
        drbg_data = loadmat("allDRBG_STS.mat")["allDRBG"]
    except:
        drbg_data = np.random.randint(0, 2, (100, 16))
    for _ in range(cfg.NumSTSSegments):
        drbg_bits = drbg_data[0].flatten()
        drbg_bits = 1 - 2 * drbg_bits
        spread_factor = 4 if cfg.Mode == "HPRF" else 8
        spread_bits = np.repeat(drbg_bits, spread_factor)
        sts = np.concatenate([sts, spread_bits, gap])
    return sts

def butterworth_filter(symbols, samples_per_pulse):
    N = 4
    fc = 500e6
    fs = fc * samples_per_pulse
    nyq = 0.5 * fs
    cutoff = fc / nyq
    b, a = butter(N, cutoff, btype="low")
    impulses = np.zeros(len(symbols) * samples_per_pulse)
    impulses[::samples_per_pulse] = symbols
    wave = lfilter(b, a, impulses)
    return wave.astype(np.complex64)

# 新增：绘制波形函数
def plot_waveform(wave, cfg, num_samples=1000):
    """绘制UWB波形的时域图和频谱图"""
    # 计算采样率
    fs = 500e6 * cfg.SamplesPerPulse  # 采样率 = 截止频率 * 每个脉冲的采样点数
    
    # 截取部分数据（全量数据可能过长）
    wave_truncated = wave[:num_samples]
    t = np.arange(len(wave_truncated)) / fs * 1e9  # 时间轴（纳秒）

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"IEEE 802.15.4a/z UWB波形 (模式: {cfg.Mode})", fontsize=14)

    # 1. 时域图
    ax1.plot(t, np.real(wave_truncated), label='同相分量 (I)')
    ax1.plot(t, np.imag(wave_truncated), label='正交分量 (Q)', alpha=0.7)
    ax1.set_xlabel('时间 (ns)')
    ax1.set_ylabel('幅度')
    ax1.set_title(f'时域波形 (前{num_samples}个采样点)')
    ax1.grid(True)
    ax1.legend()

    # 2. 频谱图（使用FFT）
    n = len(wave_truncated)
    yf = fft(wave_truncated)
    xf = fftfreq(n, 1/fs)
    xf = fftshift(xf) / 1e6  # 频率轴（MHz）
    yf_shifted = fftshift(yf)
    power = 20 * np.log10(np.abs(yf_shifted) + 1e-10)  # 功率（dB）

    ax2.plot(xf, power)
    ax2.set_xlabel('频率 (MHz)')
    ax2.set_ylabel('功率 (dB)')
    ax2.set_title('频谱特性')
    ax2.grid(True)
    ax2.set_xlim(-1000, 1000)  # UWB信号带宽通常在1GHz左右

    plt.tight_layout()
    plt.show()


# 测试代码（包含绘图）
if __name__ == "__main__":
    # 1. 创建配置
    cfg = LRWPANHRPConfig()
    cfg.Mode = "BPRF"
    cfg.PSDULength = 7  # 10字节=80比特
    cfg.SamplesPerPulse = 4  # 每个脉冲4个采样点

    # 2. 生成随机PSDU(80比特)
    psdu = np.array([1,0,1,0,1,0,1,0 ,0,1,0,1,0,1,0,1, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0])

    # 3. 生成波形
    wave, symbols, cfg = lrwpan_hrp_waveform_generator(psdu, cfg)
    print(f"生成波形长度: {len(wave)} 采样点")
    print(f"生成符号序列长度: {len(symbols)} 符号")
    # print(wave[0:100])
    # # 4. 绘制波形（显示前1000个采样点）
    # plot_waveform(wave, cfg, num_samples=19280)