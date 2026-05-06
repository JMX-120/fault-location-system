import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 注意：请确保你的目录下有 vmd_bilstm.py 文件
try:
    from vmd_bilstm import VibrationProcessor
except ImportError:
    st.error("找不到 vmd_bilstm.py 文件，请确保该文件在当前目录下。")

# --- 基础配置 ---
st.set_page_config(page_title="电-振融合故障定位演示系统", layout="wide")

# 设置支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# --- 核心算法类 ---
class HighFidelityFaultGenerator:
    def __init__(self, sample_rate=10000, duration=0.5, wave_speed=5000.0):
        self.fs = sample_rate
        self.duration = duration
        self.t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        self.wave_speed = wave_speed

    def generate_full_data(self, t0, distance):
        """生成模拟故障数据"""
        f0 = 50.0
        w = 2 * np.pi * f0
        # 基础信号
        ua = 10.0 * np.sin(w * self.t)
        ia = 2.0 * np.sin(w * self.t - np.pi / 6)

        # 故障掩码
        step_mask = np.heaviside(self.t - t0, 1)

        # A相电压暂态与电流激增
        ua = ua * (1 - step_mask) + (
                    10.0 * np.exp(-1500 * (self.t - t0)) * np.sin(2 * np.pi * 800 * (self.t - t0)) * step_mask)
        ia = ia + (8.0 * np.exp(-800 * (self.t - t0)) * np.sin(2 * np.pi * 1500 * (self.t - t0)) * step_mask)

        # 振动信号生成
        delay = distance / self.wave_speed
        t_arrive = t0 + delay
        vib_mask = np.heaviside(self.t - t_arrive, 1)
        vib = 2.5 * np.exp(-600 * (self.t - t_arrive)) * np.sin(2 * np.pi * 1200 * (self.t - t_arrive)) * vib_mask
        vib += np.random.normal(0, 0.08, len(self.t))

        return self.t, ua, ia, vib


# --- Streamlit 界面排版 ---

st.title("输电线路短路故障：电-振融合智能定位系统")
st.markdown("---")

# 1. 侧边栏控制参数
with st.sidebar:
    st.header("实验参数设置")
    sim_distance = st.slider("设定故障真实距离 (m)", 10.0, 100.0, 35.0)
    sim_t0 = st.slider("设定故障发生时间 (s)", 0.1, 0.3, 0.15)
    set_wave_speed = st.number_input("波速设置 (m/s)", value=5000)

    st.markdown("---")
    st.subheader("视觉自定义")
    fig_height = st.slider("图表整体高度", 10, 25, 18)
    line_color = st.color_picker("电压曲线颜色", "#FF0000")

    run_btn = st.button("执行融合定位分析", type="primary")

# 2. 主展示区
if run_btn:
    # 步骤 A: 信号生成
    gen = HighFidelityFaultGenerator(wave_speed=set_wave_speed)
    t, ua, ia, vib = gen.generate_full_data(sim_t0, sim_distance)

    # 步骤 B: 算法处理
    # 电突变检测
    t_elec_fault = t[np.argmax(np.abs(np.diff(ia)))]

    # 振动 VMD-BiLSTM 处理 (请确保模型文件 intensive_fault_study.pth 存在)
    try:
        processor = VibrationProcessor(model_path="intensive_fault_study.pth")
        imf2 = processor.apply_vmd(vib)
        prob_curve = processor.predict_probability(imf2)
        t_warning, t_confirm = processor.locate_toa(prob_curve, t)
    except Exception as e:
        st.error(f"算法处理出错: {e}")
        st.stop()

    # 步骤 C: 结果汇总
    if t_confirm:
        calc_dist = abs(t_confirm - t_elec_fault) * set_wave_speed
        error = abs(calc_dist - sim_distance)

        # 结果看板
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("电信号时刻", f"{t_elec_fault:.4f} s")
        col2.metric("振动到达时刻", f"{t_confirm:.4f} s")
        col3.metric("预测距离", f"{calc_dist:.2f} m")
        col4.metric("定位误差", f"{error:.2f} m", delta=f"{error:.2f}", delta_color="inverse")

    # 步骤 D: 综合成果展示图 (5合1排版)
    st.subheader("信号融合全景观测看板")

    fig, axes = plt.subplots(5, 1, figsize=(12, fig_height))
    plt.subplots_adjust(hspace=0.6)

    zoom_win = (t_elec_fault - 0.01, t_elec_fault + 0.04)

    # (1) 电压暂态
    axes[0].plot(t, ua, color=line_color, label='故障相电压')
    axes[0].set_xlim(zoom_win)
    axes[0].set_title("1. 电压暂态突变 (局部放大)", fontweight='bold')
    axes[0].set_ylabel("电压 (V)")
    axes[0].grid(True, alpha=0.3)

    # (2) 电流充放电
    axes[1].plot(t, ia, color='green', label='暂态电流')
    axes[1].axvline(t_elec_fault, color='black', linestyle='--')
    axes[1].set_xlim(zoom_win)
    axes[1].set_title("2. 电流暂态充放电过程 (局部放大)", fontweight='bold')
    axes[1].set_ylabel("电流 (A)")
    axes[1].grid(True, alpha=0.3)

    # (3) 原始振动
    axes[2].plot(t, vib, color='gray', alpha=0.6)
    axes[2].set_title("3. 原始振动信号 (全局视图)", fontweight='bold')
    axes[2].set_ylabel("幅值 (g)")
    axes[2].grid(True, alpha=0.3)

    # (4) VMD 特征
    axes[3].plot(t, imf2, color='royalblue')
    axes[3].set_title("4. VMD-IMF2 高频特征分量", fontweight='bold')
    axes[3].set_ylabel("幅值")
    axes[3].grid(True, alpha=0.3)

    # (5) 决策概率
    axes[4].plot(t, prob_curve, color='purple', linewidth=2, label='识别概率')
    axes[4].axhline(0.8, color='red', linestyle=':', label='阈值 0.8')
    if t_confirm:
        axes[4].axvline(t_confirm, color='green', label=f'确认时刻:{t_confirm:.4f}s')
    axes[4].set_title("5. BiLSTM 识别概率与到达时间决策", fontweight='bold')
    axes[4].set_xlabel("时间 (s)")
    axes[4].set_ylabel("概率")
    axes[4].legend(loc='upper right')
    axes[4].grid(True, alpha=0.3)

    st.pyplot(fig)

else:
    st.info("👈 请在侧边栏设置参数并点击运行按钮。")