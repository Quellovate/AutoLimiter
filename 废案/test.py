import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.io import wavfile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import os


import platform

system = platform.system()
if system == "Windows":
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
elif system == "Darwin":
    matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "sans-serif"]
else:
    matplotlib.rcParams["font.sans-serif"] = [
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "sans-serif",
    ]

matplotlib.rcParams["axes.unicode_minus"] = False


class AudioGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("正弦波音频生成器")
        self.root.geometry("1000x750")
        self.root.resizable(True, True)

        self.audio_data = None
        self.sample_rate = 44100

        self.setup_ui()

    def setup_ui(self):

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_frame, text="参数控制", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        freq_frame = ttk.Frame(control_frame)
        freq_frame.pack(fill=tk.X, pady=5)

        ttk.Label(freq_frame, text="频率 (Hz):", width=12).pack(side=tk.LEFT)
        self.freq_var = tk.DoubleVar(value=440)
        self.freq_slider = ttk.Scale(
            freq_frame,
            from_=20,
            to=2000,
            variable=self.freq_var,
            orient=tk.HORIZONTAL,
            command=self.on_param_change,
        )
        self.freq_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.freq_label = ttk.Label(freq_frame, text="440.0 Hz", width=12)
        self.freq_label.pack(side=tk.LEFT)

        vol_frame = ttk.Frame(control_frame)
        vol_frame.pack(fill=tk.X, pady=5)

        ttk.Label(vol_frame, text="音量 (dBFS):", width=12).pack(side=tk.LEFT)
        self.volume_var = tk.DoubleVar(value=-6)
        self.volume_slider = ttk.Scale(
            vol_frame,
            from_=-60,
            to=0,
            variable=self.volume_var,
            orient=tk.HORIZONTAL,
            command=self.on_param_change,
        )
        self.volume_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.volume_label = ttk.Label(vol_frame, text="-6.0 dB", width=12)
        self.volume_label.pack(side=tk.LEFT)

        dur_frame = ttk.Frame(control_frame)
        dur_frame.pack(fill=tk.X, pady=5)

        ttk.Label(dur_frame, text="时长 (秒):", width=12).pack(side=tk.LEFT)
        self.duration_var = tk.DoubleVar(value=2)
        self.duration_slider = ttk.Scale(
            dur_frame,
            from_=0.1,
            to=10,
            variable=self.duration_var,
            orient=tk.HORIZONTAL,
            command=self.on_param_change,
        )
        self.duration_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.duration_label = ttk.Label(dur_frame, text="2.0 秒", width=12)
        self.duration_label.pack(side=tk.LEFT)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.generate_btn = ttk.Button(
            button_frame,
            text="🎵 生成音频",
            command=self.generate_audio,
            style="Accent.TButton",
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(
            button_frame, text="💾 保存文件", command=self.save_audio, state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(
            button_frame, text="▶️ 播放", command=self.play_audio, state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="就绪 - 调整参数后点击「生成音频」")
        self.status_label = ttk.Label(
            button_frame, textvariable=self.status_var, foreground="gray"
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

        waveform_container = ttk.LabelFrame(
            main_frame, text="波形显示 (dB)", padding="5"
        )
        waveform_container.pack(fill=tk.BOTH, expand=True)

        waveform_frame = ttk.Frame(waveform_container)
        waveform_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.fig.set_facecolor("#f0f0f0")

        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("完整波形 (dBFS)")
        self.ax1.set_xlabel("时间 (秒)")
        self.ax1.set_ylabel("电平 (dBFS)")
        self.ax1.grid(True, alpha=0.3)

        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("局部放大 (前 10ms)")
        self.ax2.set_xlabel("时间 (ms)")
        self.ax2.set_ylabel("电平 (dBFS)")
        self.ax2.grid(True, alpha=0.3)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=waveform_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        peak_frame = ttk.Frame(waveform_container, width=180)
        peak_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        peak_frame.pack_propagate(False)

        peak_title = ttk.Label(
            peak_frame, text="峰值电平", font=("Microsoft YaHei", 12, "bold")
        )
        peak_title.pack(pady=(20, 10))

        self.peak_value_var = tk.StringVar(value="-- dB")
        self.peak_value_label = tk.Label(
            peak_frame,
            textvariable=self.peak_value_var,
            font=("Consolas", 18, "bold"),
            fg="#00AA00",
            bg="#1a1a1a",
            width=10,
            height=2,
        )
        self.peak_value_label.pack(pady=15, padx=10)

        legend_frame = ttk.LabelFrame(peak_frame, text="参考线图例", padding="10")
        legend_frame.pack(fill=tk.X, padx=5, pady=20)

        legend_0db = ttk.Frame(legend_frame)
        legend_0db.pack(fill=tk.X, pady=3)
        tk.Label(
            legend_0db, text="━━━━", fg="#FF0000", font=("Consolas", 14, "bold")
        ).pack(side=tk.LEFT)
        ttk.Label(legend_0db, text=" 0 dB (削波)").pack(side=tk.LEFT)

        legend_6db = ttk.Frame(legend_frame)
        legend_6db.pack(fill=tk.X, pady=3)
        tk.Label(
            legend_6db, text="━━━━", fg="#0080FF", font=("Consolas", 14, "bold")
        ).pack(side=tk.LEFT)
        ttk.Label(legend_6db, text=" -6 dB").pack(side=tk.LEFT)

        legend_12db = ttk.Frame(legend_frame)
        legend_12db.pack(fill=tk.X, pady=3)
        tk.Label(
            legend_12db, text="━━━━", fg="#4B0082", font=("Consolas", 14, "bold")
        ).pack(side=tk.LEFT)
        ttk.Label(legend_12db, text=" -12 dB").pack(side=tk.LEFT)

        legend_peak = ttk.Frame(legend_frame)
        legend_peak.pack(fill=tk.X, pady=3)
        tk.Label(
            legend_peak, text="━━━━", fg="#006400", font=("Consolas", 14, "bold")
        ).pack(side=tk.LEFT)
        ttk.Label(legend_peak, text=" 当前峰值").pack(side=tk.LEFT)

        info_frame = ttk.LabelFrame(main_frame, text="音频信息", padding="5")
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.info_var = tk.StringVar(value="尚未生成音频")
        ttk.Label(info_frame, textvariable=self.info_var).pack(anchor=tk.W)

    def on_param_change(self, event=None):
        """参数变化时更新标签"""
        self.freq_label.config(text=f"{self.freq_var.get():.1f} Hz")
        self.volume_label.config(text=f"{self.volume_var.get():.1f} dB")
        self.duration_label.config(text=f"{self.duration_var.get():.1f} 秒")

    def db_to_amplitude(self, db):
        """dBFS 转振幅"""
        return 10 ** (db / 20)

    def amplitude_to_db(self, amplitude):
        """振幅转 dBFS，处理零值和负值"""
        min_amplitude = 1e-10
        amplitude_abs = np.abs(amplitude)
        amplitude_safe = np.maximum(amplitude_abs, min_amplitude)
        return 20 * np.log10(amplitude_safe)

    def update_peak_display(self, peak_db):
        """更新峰值显示"""
        self.peak_value_var.set(f"{peak_db:.1f} dB")

        if peak_db >= -3:
            color = "#FF0000"
        elif peak_db >= -6:
            color = "#0080FF"
        elif peak_db >= -12:
            color = "#6600CC"
        else:
            color = "#00AA00"

        self.peak_value_label.config(fg=color)

    def generate_audio(self):
        """生成音频"""
        try:

            frequency = self.freq_var.get()
            volume_db = self.volume_var.get()
            duration = self.duration_var.get()

            amplitude = self.db_to_amplitude(volume_db)

            num_samples = int(self.sample_rate * duration)
            t = np.linspace(0, duration, num_samples, endpoint=False)

            waveform = amplitude * np.sin(2 * np.pi * frequency * t)

            self.audio_data = (waveform * 32767).astype(np.int16)
            self.time_axis = t
            self.waveform_float = waveform

            self.waveform_db = self.amplitude_to_db(waveform)

            peak_db = (
                20 * np.log10(np.max(np.abs(waveform)))
                if np.max(np.abs(waveform)) > 0
                else -np.inf
            )

            self.update_peak_display(peak_db)

            self.update_waveform_display()

            info_text = (
                f"波形类型: 正弦波 | "
                f"频率: {frequency:.1f} Hz | "
                f"时长: {duration:.2f} 秒 | "
                f"采样率: {self.sample_rate} Hz | "
                f"样本数: {num_samples:,} | "
                f"峰值: {peak_db:.1f} dBFS"
            )
            self.info_var.set(info_text)

            self.save_btn.config(state=tk.NORMAL)
            self.play_btn.config(state=tk.NORMAL)

            self.status_var.set("✅ 音频已生成")
            self.status_label.config(foreground="green")

        except Exception as e:
            messagebox.showerror("错误", f"生成音频时出错:\n{str(e)}")
            self.status_var.set("❌ 生成失败")
            self.status_label.config(foreground="red")

    def update_waveform_display(self):
        """更新波形显示 (dB)"""

        self.ax1.clear()
        self.ax2.clear()

        current_volume_db = self.volume_var.get()

        y_min = -80
        y_max = 5

        self.ax1.plot(self.time_axis, self.waveform_db, color="#00FFFF", linewidth=0.5)
        self.ax1.set_title("完整波形 (dBFS)", fontsize=10)
        self.ax1.set_xlabel("时间 (秒)")
        self.ax1.set_ylabel("电平 (dBFS)")
        self.ax1.set_ylim(y_min, y_max)
        self.ax1.set_facecolor("#ffffff")

        self.ax1.axhline(y=0, color="#FF0000", linestyle="-", linewidth=1.0, alpha=0.9)
        self.ax1.axhline(y=-6, color="#0080FF", linestyle="-", linewidth=1.0, alpha=0.9)
        self.ax1.axhline(
            y=-12, color="#4B0082", linestyle="-", linewidth=1.0, alpha=0.9
        )
        self.ax1.axhline(
            y=current_volume_db,
            color="#006400",
            linestyle="-",
            linewidth=1.0,
            alpha=0.9,
        )

        self.ax1.grid(True, alpha=0.2, color="white")
        self.ax1.tick_params(colors="black")

        samples_10ms = int(self.sample_rate * 0.01)
        samples_10ms = min(samples_10ms, len(self.time_axis))

        time_ms = self.time_axis[:samples_10ms] * 1000
        wave_db_segment = self.waveform_db[:samples_10ms]

        self.ax2.plot(time_ms, wave_db_segment, color="#ADFF2F", linewidth=1)
        self.ax2.set_title("局部放大 (前 10ms)", fontsize=10)
        self.ax2.set_xlabel("时间 (ms)")
        self.ax2.set_ylabel("电平 (dBFS)")
        self.ax2.set_ylim(y_min, y_max)
        self.ax2.set_facecolor("#ffffff")

        self.ax2.axhline(y=0, color="#FF0000", linestyle="-", linewidth=1.0, alpha=0.9)
        self.ax2.axhline(y=-6, color="#0080FF", linestyle="-", linewidth=1.0, alpha=0.9)
        self.ax2.axhline(
            y=-12, color="#4B0082", linestyle="-", linewidth=1.0, alpha=0.9
        )
        self.ax2.axhline(
            y=current_volume_db,
            color="#006400",
            linestyle="-",
            linewidth=1.0,
            alpha=0.9,
        )

        self.ax2.grid(True, alpha=0.2, color="white")
        self.ax2.tick_params(colors="black")

        self.fig.tight_layout()
        self.canvas.draw()

    def save_audio(self):
        """保存音频文件"""
        if self.audio_data is None:
            messagebox.showwarning("警告", "请先生成音频")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")],
            initialname="sine_wave.wav",
        )

        if filename:
            try:
                wavfile.write(filename, self.sample_rate, self.audio_data)
                self.status_var.set(f"✅ 已保存: {os.path.basename(filename)}")
                self.status_label.config(foreground="green")
                messagebox.showinfo("成功", f"音频已保存至:\n{filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败:\n{str(e)}")

    def play_audio(self):
        """播放音频"""
        if self.audio_data is None:
            messagebox.showwarning("警告", "请先生成音频")
            return

        try:
            import sounddevice as sd

            sd.play(self.audio_data, self.sample_rate)
            self.status_var.set("🔊 正在播放...")
            self.status_label.config(foreground="blue")
        except ImportError:
            try:
                import tempfile
                import subprocess

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    wavfile.write(temp_path, self.sample_rate, self.audio_data)

                current_system = platform.system()
                if current_system == "Windows":
                    os.startfile(temp_path)
                elif current_system == "Darwin":
                    subprocess.run(["afplay", temp_path])
                else:
                    subprocess.run(["aplay", temp_path])

                self.status_var.set("🔊 使用系统播放器播放")
                self.status_label.config(foreground="blue")
            except Exception as e:
                messagebox.showinfo(
                    "提示",
                    "播放功能需要安装 sounddevice 库:\n"
                    "pip install sounddevice\n\n"
                    "或者请先保存文件后使用其他播放器播放",
                )


def main():
    root = tk.Tk()

    style = ttk.Style()
    style.theme_use("clam")

    app = AudioGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
