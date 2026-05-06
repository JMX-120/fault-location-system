import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import os

# --- 字体兼容性修复 ---
font_path = 'simhei.ttf'
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# --- 导入算法逻辑 ---
try:
    from vmd_bilstm import VibrationProcessor
except ImportError:
    st.error("找不到 vmd_bilstm.py 文件，请确保该文件在当前目录下。")

# --- 页面基础配置 ---
st.set_page_config(page_title="电-振融合故障定位演示系统", layout="wide")

# --- 核心算法类 (补全三相逻辑) ---
class HighFidelityFaultGenerator:
    def __init__(self, sample_rate=10000, duration=0.5, wave_speed=5000.0):
        self.fs = sample_rate
        self.duration = duration
        self.t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        self.wave_speed = wave_speed

    def generate_full_data(self, t0, distance):
        f0 = 50.0
        w = 2 * np.pi * f0
        
        # 1. 生成三相正常电压 (幅值 10V)
        ua = 10.0 * np.sin(w * self.t)
        ub = 10.0 * np.sin(w * self.t - 2*np.pi/3)
        uc = 10.0 * np.sin(w * self.t + 2*np.pi/3)
        
        # 2. 生成三相正常电流 (幅值 2A)
        ia = 2.0 * np.sin(w * self.t - np.pi/6)
        ib = 2.0 * np.sin(w * self.t - 2*np.pi/3 - np.pi/6)
        ic = 2.0 * np.sin(w * self.t + 2*np.pi/3 - np.pi/6)

        # 故障掩码
        mask = np.heaviside(self.t - t0, 1)

        # 3. 模拟中性点不接地系统单相接地 (A相故障)
        # A相电压跌落，B/C相电压升高至1.732倍
        ua_f = ua * (1 - mask) + (5.0 * np.exp(-1200 * (self.t - t0)) * np.sin(2 * np.pi * 800 * (self.t - t0)) * mask)
        ub_f = ub * (1 - mask) + (ub * 1.732 * mask)
        uc_f = uc * (1 - mask) + (uc * 1.732 * mask)
        
        # A相暂态电流激增 (容性暂态)
        ia_f = ia + (8.0 * np.exp(-800 * (self.t - t0)) * np.sin(2 * np.pi * 1500 * (self.t - t0)) * mask)

        # 4. 振动信号 (受 A 相电流影响)
        delay = distance / self.wave_speed
        t_arrive = t0 + delay
        vib_mask = np.heaviside(self.t - t_arrive, 1)
        vib = 2.5 * np.exp(-600 * (self.t - t_arrive)) * np.sin(2 * np.pi * 1200 * (self.t - t_arrive)) * vib_mask
        vib += np.random.normal(0, 0.05, len(self.t))

        return self.t, (ua_f, ub_f, uc_f), (ia_f, ib, ic), vib

# --- Streamlit 界面 ---
st.title("输电线路短路故障：电-振融合智能定位系统")
st.markdown("---")

with st.sidebar:
    st.header("实验参数设置")
    sim_distance = st.slider("设定故障真实距离 (m)", 10.0, 100.0, 35.0)
    sim_t0 = st.slider("设定故障发生时间 (s)", 0.1, 0.3, 0.15)
    set_wave_speed = st.number_input("波速设置 (m/s)", value=5000)
    st.markdown("---")
    st.subheader("视觉自定义")
    fig_height = st.slider("图表整体高度", 10, 30, 22)
    run_btn = st.button("执行融合定位分析", type="primary")

if run_btn:
    gen = HighFidelityFaultGenerator(wave_speed=set_wave_speed)
    t, voltages, currents, vib = gen.generate_full_data(sim_t0, sim_distance)
    ua, ub, uc = voltages
    ia, ib, ic = currents

    try:
        t_elec_fault = t[np.argmax(np.abs(np.diff(ia)))]
        processor = VibrationProcessor(model_path="intensive_fault_study.pth")
        imf2 = processor.apply_vmd(vib)
        prob_curve = processor.predict_probability(imf2)
        _, t_confirm = processor.locate_toa(prob_curve, t)
    except Exception as e:
        st.error(f"算法处理出错: {e}")
        st.stop()

    if t_confirm:
        calc_dist = abs(t_confirm - t_elec_fault) * set_wave_speed
        error = abs(calc_dist - sim_distance)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("电信号时刻", f"{t_elec_fault:.4f} 秒")
        col2.metric("振动到达时刻", f"{t_confirm:.4f} 秒")
        col3.metric("预测距离", f"{calc_dist:.2f} 米")
        col4.metric("定位误差", f"{error:.2f} 米", delta=f"{error:.2f}", delta_color="inverse")

    st.subheader("信号融合全景观测看板")
    fig, axes = plt.subplots(5, 1, figsize=(12, fig_height))
    plt.subplots_adjust(hspace=0.6)
    zoom_win = (t_elec_fault - 0.02, t_elec_fault + 0.06)
    fp = font_prop if os.path.exists(font_path) else None

    # (1) 三相电压
    axes[0].plot(t, ua, color='red', label='电压 A (故障相)')
    axes[0].plot(t, ub, color='green', label='电压 B (非故障相升压)')
    axes[0].plot(t, uc, color='blue', label='电压 C (非故障相升压)')
    axes[0].set_xlim(zoom_win)
    axes[0].set_title("1. 中性点不接地系统：故障瞬间电压暂态", fontproperties=fp, fontweight='bold')
    axes[0].set_ylabel("电压 (V)", fontproperties=fp)
    axes[0].legend(prop=fp, loc='upper right', fontsize='small')
    axes[0].grid(True, alpha=0.3)

    # (2) 三相电流
    axes[1].plot(t, ia, color='red', label='电流 A (容性暂态)')
    axes[1].plot(t, ib, color='green', label='电流 B', alpha=0.6)
    axes[1].plot(t, ic, color='blue', label='电流 C', alpha=0.6)
    axes[1].axvline(t_elec_fault, color='black', linestyle='--', label='电突变起始')
    axes[1].set_xlim(zoom_win)
    axes[1].set_title("2. 三相电流信号：暂态充放电过程", fontproperties=fp, fontweight='bold')
    axes[1].set_ylabel("电流 (A)", fontproperties=fp)
    axes[1].legend(prop=fp, loc='upper right', fontsize='small')
    axes[1].grid(True, alpha=0.3)

    # (3) 原始振动
    axes[2].plot(t, vib, color='gray', alpha=0.6)
    axes[2].set_title("3. 原始振动信号 (全局视图)", fontproperties=fp, fontweight='bold')
    axes[2].set_ylabel("幅值 (g)", fontproperties=fp)
    axes[2].grid(True, alpha=0.3)

    # (4) VMD分量
    axes[3].plot(t, imf2, color='royalblue')
    axes[3].set_title("4. VMD-IMF2 高频特征分量", fontproperties=fp, fontweight='bold')
    axes[3].set_ylabel("幅值", fontproperties=fp)
    axes[3].grid(True, alpha=0.3)

    # (5) 决策概率
    axes[4].plot(t, prob_curve, color='purple', linewidth=2)
    axes[4].set_title("5. BiLSTM 识别概率与到达时间决策", fontproperties=fp, fontweight='bold')
    axes[4].set_xlabel("时间 (s)", fontproperties=fp)
    axes[4].set_ylabel("概率", fontproperties=fp)
    if t_confirm:
        axes[4].axvline(t_confirm, color='green')
    axes[4].grid(True, alpha=0.3)

    st.pyplot(fig)
else:
    st.info("👈 请点击侧边栏按钮开始分析。")
