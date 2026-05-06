import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from vmdpy import VMD
import matplotlib

# 设置简体中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ================= 1. 深度神经网络架构 =================
class VibrationFaultNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(VibrationFaultNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        return self.sigmoid(self.fc(lstm_out))


# ================= 2. 核心处理类 =================
class VibrationProcessor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VibrationFaultNet().to(self.device)
        self.has_model = False
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.has_model = True
            except:
                pass
        self.model.eval()

    def apply_vmd(self, signal):
        K, alpha, tau, tol = 8, 1330, 0, 1e-7
        u, _, _ = VMD(signal, alpha, tau, K, 0, 1, tol)
        return u[1, :]

    def predict_probability(self, imf_signal):
        if not self.has_model:
            print("⚠️ 未检测到训练好的模型，当前展示为 [演示模拟曲线]")
            peak_idx = np.argmax(np.abs(imf_signal))
            prob = np.full_like(imf_signal, 0.5)
            prob[peak_idx: peak_idx + 500] = 0.95
            # 使用较宽的卷积核模拟你图中的概率爬坡过程
            return np.convolve(prob, np.ones(80) / 80, mode='same')

        norm_data = (imf_signal - np.mean(imf_signal)) / (np.std(imf_signal) + 1e-9)
        input_tensor = torch.FloatTensor(norm_data).view(1, -1, 1).to(self.device)
        with torch.no_grad():
            return self.model(input_tensor).cpu().numpy().flatten()

    def locate_toa(self, prob_curve, t):
        """同时提取‘预警时刻’和‘确认融合时刻’"""
        exceed_idx = np.where(prob_curve > 0.8)[0]
        if len(exceed_idx) == 0: return None, None

        # 1. 确认时刻：概率突破 0.8 的瞬间 (用于融合定位)
        trigger_idx = exceed_idx[0]
        t_confirm = t[trigger_idx]

        # 2. 预警时刻：在此前 300 个点内寻找导数最大点 (起跳波头)
        search_start = max(0, trigger_idx - 300)
        window = prob_curve[search_start: trigger_idx]
        if len(window) > 1:
            toa_idx = search_start + np.argmax(np.diff(window))
            t_warning = t[toa_idx]
            return t_warning, t_confirm

        return t_confirm, t_confirm


# ================= 3. 主程序 =================
def run_vmd_bilstm_analysis(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {filepath}")
        return

    t, vib = df['Time'].values, df['Vibration_Signal'].values

    processor = VibrationProcessor(model_path="intensive_fault_study.pth")
    imf2 = processor.apply_vmd(vib)
    prob_curve = processor.predict_probability(imf2)

    # 提取两个时刻
    t_warning, t_confirm = processor.locate_toa(prob_curve, t)

    # --- 绘图增强 ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax1.plot(t, vib, color='gray', alpha=0.5)
    ax1.set_title("原始振动信号", fontsize=12)

    ax2.plot(t, imf2, color='blue')
    ax2.set_title("VMD 特征提取 (IMF2 分量)", fontsize=12)

    # 第三张图：BiLSTM 识别结果双线标注
    ax3.plot(t, prob_curve, color='purple', linewidth=1.5, label='BiLSTM 故障识别概率')
    ax3.axhline(0.8, color='red', linestyle=':', label='置信阈值 (0.8)')

    if t_warning and t_confirm:
        # 【修改点 1】标注预警时刻（橙色，起跳点）
        ax3.axvline(t_warning, color='orange', linestyle='--', linewidth=2, label='预警时刻 (波头起跳)')
        # 为了防止文字重叠，错开 Y 轴高度
        ax3.text(t_warning - 0.03, 0.55, f'预警: {t_warning:.4f}s', color='orange', fontweight='bold', fontsize=11)
        ax3.scatter(t_warning, prob_curve[np.where(t == t_warning)[0][0]], color='orange', s=50, zorder=5)

        # 【修改点 2】标注确认时刻（绿色，用于融合定位）
        ax3.axvline(t_confirm, color='green', linestyle='-', linewidth=2, label='确认时刻 (用于融合定位)')
        ax3.text(t_confirm + 0.005, 0.85, f'确认: {t_confirm:.4f}s', color='green', fontweight='bold', fontsize=11)
        ax3.scatter(t_confirm, 0.8, color='green', s=50, zorder=5)

    ax3.set_title("BiLSTM 时序模式识别与双端点定位", fontsize=12)
    ax3.set_xlabel("时间 (s)", fontsize=10)
    ax3.set_ylabel("识别概率", fontsize=10)
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    if t_warning and t_confirm:
        print("-" * 40)
        print("✅ 时序识别完成！")
        print(f"⚠️ [早期预警] 波头起跳时刻: {t_warning:.4f} s (可能发生故障)")
        print(f"🎯 [确认接收] 融合定位时刻: {t_confirm:.4f} s (算法完全确信)")
        print(f"⏳ 算法判断延迟: {(t_confirm - t_warning) * 1000:.2f} ms")
        print("-" * 40)


if __name__ == "__main__":
    run_vmd_bilstm_analysis("simulated_fault_record_001.csv")