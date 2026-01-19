import shutil
import threading
import tkinter as tk
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# --- 依赖库导入与检查 ---
try:
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    # pedalboard 用于加载 VST3 插件和进行音频效果处理
    from pedalboard import Gain, Pedalboard, VST3Plugin
except ImportError as e:
    raise SystemExit(f"缺少必要库: {e}\n请运行: pip install numpy soundfile sounddevice pedalboard")

try:
    from pydub import AudioSegment

    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("警告: 未检测到 pydub 库。MP3 转换功能不可用。\n请运行: pip install pydub")


@dataclass(frozen=True)
class AudioInfo:
    """数据类：存储音频文件的基础元数据"""

    samplerate: int  # 采样率
    channels: int  # 通道数
    frames: int  # 总采样点数

    @property
    def duration(self) -> float:
        """计算音频时长（秒）"""
        return self.frames / self.samplerate if self.samplerate > 0 else 0.0


@dataclass(frozen=True)
class Metrics:
    """数据类：存储详细的声学分析指标"""

    samplerate: int
    channels: int
    duration_s: float
    peak_dbfs: float  # 峰值电平
    rms_dbfs: float  # 平均响度 (RMS)
    crest_db: float  # 动态因子 (Peak - RMS)
    lra_db: float | None  # 响度范围 (Loudness Range)
    clip_sample_percent: float  # 削波样本百分比
    clip_frame_percent: float  # 削波帧百分比


def format_time(seconds: float) -> str:
    """辅助函数：将秒数格式化为 MM:SS 或 HH:MM:SS"""
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _dbfs(x: float, *, floor_db: float = -120.0) -> float:
    """辅助函数：将线性振幅转换为分贝 (dBFS)"""
    x = float(x)
    if x <= 0:
        return floor_db
    return max(20.0 * float(np.log10(x)), floor_db)


def analyze_audio_file(path: str) -> Metrics:
    """
    核心分析逻辑：读取音频文件并计算声学指标。
    使用分块读取 (Streaming) 方式以避免将大文件一次性加载到内存。
    """
    clip_threshold = 0.9999  # 判定削波的阈值
    loudness_gate_db = -70.0  # 计算 LRA 时忽略的静音阈值
    block_seconds = 0.4  # 每次分析的时间块长度

    with sf.SoundFile(path, mode="r") as f:
        samplerate = int(f.samplerate)
        channels = int(f.channels)
        frames_total = int(len(f))
        duration_s = frames_total / samplerate if samplerate > 0 else 0.0
        blocksize = max(int(samplerate * block_seconds), 1024)

        peak = 0.0
        sumsq = 0.0
        total_samples = 0
        clip_samples = 0
        clip_frames = 0
        block_rms_db = []

        # 分块读取音频数据
        for block in f.blocks(blocksize=blocksize, dtype="float32", always_2d=True):
            if block.size == 0:
                continue
            abs_block = np.abs(block)

            # 更新全局峰值
            peak = max(peak, float(abs_block.max()))
            # 累加平方和用于计算全局 RMS
            sumsq += float(np.sum(block * block))
            total_samples += int(block.size)

            # 统计削波情况
            clip_mask = abs_block >= clip_threshold
            clip_samples += int(np.count_nonzero(clip_mask))
            clip_frames += int(np.count_nonzero(np.any(clip_mask, axis=1)))

            # 计算当前块的 RMS，用于后续计算 LRA (响度范围)
            mono = block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
            rms_block = float(np.sqrt(np.mean(mono * mono)))
            rms_block_db = _dbfs(rms_block)

            if rms_block_db > loudness_gate_db:
                block_rms_db.append(rms_block_db)

        # 计算全局 RMS
        rms = float(np.sqrt(sumsq / max(total_samples, 1)))

        # 计算响度范围 (Loudness Range - 简易算法：95%分位 - 10%分位)
        lra_db = None
        if len(block_rms_db) >= 2:
            p10, p95 = np.percentile(np.asarray(block_rms_db, dtype=np.float64), [10, 95])
            lra_db = float(p95 - p10)

        return Metrics(
            samplerate=samplerate,
            channels=channels,
            duration_s=duration_s,
            peak_dbfs=_dbfs(peak),
            rms_dbfs=_dbfs(rms),
            crest_db=_dbfs(peak) - _dbfs(rms),
            lra_db=lra_db,
            clip_sample_percent=(clip_samples / max(total_samples, 1)) * 100.0,
            clip_frame_percent=(clip_frames / max(frames_total, 1)) * 100.0,
        )


class AudioPlayer:
    """
    音频播放器类。
    使用 sounddevice 的 OutputStream 回调机制进行播放，避免阻塞 GUI。
    """

    def __init__(self):
        self._lock = threading.RLock()  # 线程锁，防止 UI 线程和音频回调线程冲突
        self._sf = None  # soundfile 文件句柄
        self._stream = None  # sounddevice 输出流
        self._info = None
        self._paused = True
        self._finished = False
        self._position_frames = 0

    def load(self, path: str) -> AudioInfo:
        """加载音频文件"""
        with self._lock:
            self.close()
            try:
                f = sf.SoundFile(path, mode="r")
            except Exception as e:
                raise RuntimeError(f"无法打开音频: {e}")

            self._sf = f
            self._info = AudioInfo(int(f.samplerate), int(f.channels), int(len(f)))
            self._position_frames = 0
            self._sf.seek(0)
            self._paused = True
            self._finished = False
            return self._info

    def play(self):
        """开始或继续播放"""
        with self._lock:
            if not self._sf:
                return
            if not self._stream:
                self._create_stream()
            if not self._stream.active:
                self._stream.start()
            self._paused = False
            self._finished = False

    def pause(self, close_stream=True):
        """暂停播放"""
        with self._lock:
            self._paused = True
            if close_stream and self._stream:
                if self._stream.active:
                    self._stream.stop()
                self._stream.close()
                self._stream = None

    def seek_seconds(self, seconds: float):
        """跳转进度"""
        with self._lock:
            if not self._sf or not self._info:
                return
            target = int(float(seconds) * self._info.samplerate)
            target = max(0, min(target, self._info.frames))
            self._sf.seek(target)
            self._position_frames = target
            self._finished = False

    def get_state(self):
        """获取当前播放状态（供 UI 刷新进度条使用）"""
        with self._lock:
            pos_s = (self._position_frames / self._info.samplerate) if self._info else 0.0
            return self._info, self._paused, self._finished, pos_s

    def close(self):
        """关闭资源"""
        self.pause(close_stream=True)
        with self._lock:
            if self._sf:
                self._sf.close()
                self._sf = None
            self._info = None

    def _create_stream(self):
        """创建 sounddevice 输出流的回调函数"""

        def callback(outdata, frames, time_info, status):
            with self._lock:
                if not self._sf or self._paused:
                    outdata.fill(0)
                    return
                # 从文件读取数据填充到缓冲区
                data = self._sf.read(frames, dtype="float32", always_2d=True)
                read = len(data)
                if read > 0:
                    outdata[:read, :] = data
                if read < frames:
                    # 如果读到的数据不够（文件结束），补零并停止
                    outdata[read:, :].fill(0)
                    self._position_frames = self._info.frames
                    self._paused = True
                    self._finished = True
                    raise sd.CallbackStop()
                self._position_frames = int(self._sf.tell())

        self._stream = sd.OutputStream(
            samplerate=self._info.samplerate,
            channels=self._info.channels,
            dtype="float32",
            callback=callback,
        )


def set_param_db_linear(plugin: VST3Plugin, name: str, db_value: float) -> None:
    """
    辅助函数：设置 VST 参数。
    将用户直观的 dB 值映射到 VST 插件需要的 0.0-1.0 归一化数值。
    """
    if name not in plugin.parameters:
        return
    p = plugin.parameters[name]
    target = float(np.clip(db_value, p.min_value, p.max_value))
    # 线性映射
    raw = (target - p.min_value) / (p.max_value - p.min_value)
    p.raw_value = float(np.clip(raw, 0.0, 1.0))


def set_bool(plugin: VST3Plugin, name: str, enabled: bool) -> None:
    """辅助函数：设置 VST 的开关型参数"""
    if name in plugin.parameters:
        plugin.parameters[name].raw_value = 1.0 if enabled else 0.0


class TrackPanel(ttk.LabelFrame):
    """
    UI 组件：单轨道控制面板。
    包含文件名显示、播放控制按钮、进度条。
    """

    def __init__(self, master, title, on_play_req, on_load_ok=None):
        super().__init__(master, text=title, padding=5)
        self.player = AudioPlayer()
        self.on_play_req = on_play_req  # 请求播放的回调（用于实现互斥播放）
        self.on_load_ok = on_load_ok  # 文件加载成功后的回调（用于触发分析）
        self.loaded_path = None
        self.dragging = False  # 标记是否正在拖拽进度条

        self._create_ui()

    def _create_ui(self):
        # 路径标签
        self.lbl_path = ttk.Label(self, text="[空]", foreground="gray")
        self.lbl_path.pack(fill="x", pady=(0, 2))

        # 控制区
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=2)

        self.btn_play = ttk.Button(ctrl, text="播放", command=self._toggle_play, state="disabled")
        self.btn_play.pack(side="left")

        self.lbl_time = ttk.Label(ctrl, text="00:00 / 00:00")
        self.lbl_time.pack(side="right", padx=5)

        # 进度条
        self.var_prog = tk.DoubleVar()
        self.scale = ttk.Scale(self, variable=self.var_prog, from_=0, to=1, command=self._on_seek)
        self.scale.pack(fill="x", pady=(2, 0))
        self.scale.bind("<ButtonPress-1>", lambda e: setattr(self, "dragging", True))
        self.scale.bind("<ButtonRelease-1>", self._on_seek_release)

    def load_file(self, path):
        """加载音频文件到播放器"""
        try:
            info = self.player.load(str(path))
            self.loaded_path = str(path)
            self.lbl_path.config(text=f"{Path(path).name}", foreground="black")
            self.scale.config(to=info.duration)
            self.btn_play.config(state="normal")
            self._update_time_label(0, info.duration)
            if self.on_load_ok:
                self.on_load_ok(str(path))
        except Exception as e:
            self.lbl_path.config(text=f"错误: {e}", foreground="red")
            self.btn_play.config(state="disabled")

    def _toggle_play(self):
        """播放/暂停切换"""
        info, paused, _, _ = self.player.get_state()
        if not info:
            return
        if paused:
            # 播放前先通知主界面暂停其他轨道（互斥播放）
            self.on_play_req(self)
            self.player.play()
            self.btn_play.config(text="暂停")
        else:
            self.player.pause()
            self.btn_play.config(text="播放")

    def pause_external(self):
        """供外部调用：强制暂停"""
        self.player.pause()
        self.btn_play.config(text="播放")

    def _on_seek(self, val):
        if self.dragging:
            self.lbl_time.config(text=f"{format_time(float(val))} / ...")

    def _on_seek_release(self, event):
        self.dragging = False
        self.player.seek_seconds(float(self.var_prog.get()))

    def _update_time_label(self, current, total):
        self.lbl_time.config(text=f"{format_time(current)} / {format_time(total)}")

    def ui_tick(self):
        """定时刷新 UI（进度条位置）"""
        info, paused, finished, pos = self.player.get_state()
        if not info:
            return

        if not self.dragging:
            self.var_prog.set(pos)
            self._update_time_label(pos, info.duration)

        if finished:
            self.btn_play.config(text="播放")
            self.player.pause()
        elif not paused:
            self.btn_play.config(text="暂停")
        else:
            self.btn_play.config(text="播放")

    def close(self):
        self.player.close()


class AnalysisPanel(ttk.LabelFrame):
    """
    UI 组件：结果对比表格。
    显示输入与输出音频的指标（峰值、RMS等）及其变化量。
    """

    def __init__(self, master):
        super().__init__(master, text="结果对比", padding=5)
        self.metrics = [
            ("peak_dbfs", "峰值 (dBFS)"),
            ("rms_dbfs", "RMS (dBFS)"),
            ("crest_db", "动态因子 (dB)"),
            ("lra_db", "响度范围 (dB)"),
            ("clip_sample_percent", "削波率 (%)"),
        ]
        self.vars = {}
        self._create_grid()

    def _create_grid(self):
        for i in range(4):
            self.columnconfigure(i, weight=1)

        headers = ["指标", "输入 (In)", "输出 (Out)", "变化 (Δ)"]
        for col, text in enumerate(headers):
            ttk.Label(self, text=text, font=("", 9, "bold")).grid(row=0, column=col, sticky="w", pady=(0, 2))

        # 动态生成表格行
        for i, (key, label) in enumerate(self.metrics, 1):
            ttk.Label(self, text=label).grid(row=i, column=0, sticky="w", pady=1)
            v_a = tk.StringVar(value="-")
            v_b = tk.StringVar(value="-")
            v_d = tk.StringVar(value="-")
            self.vars[key] = (v_a, v_b, v_d)
            ttk.Label(self, textvariable=v_a).grid(row=i, column=1, sticky="w")
            ttk.Label(self, textvariable=v_b).grid(row=i, column=2, sticky="w")
            ttk.Label(self, textvariable=v_d).grid(row=i, column=3, sticky="w")

        self.lbl_summary = ttk.Label(self, text="等待分析...", foreground="gray", wraplength=350)
        self.lbl_summary.grid(row=len(self.metrics) + 1, column=0, columnspan=4, pady=(5, 0), sticky="w")

    def update_data(self, m_a: Metrics, m_b: Metrics, title_suffix: str):
        """更新表格数据"""
        self.config(text=f"结果对比: {title_suffix}")

        def fmt(v):
            return f"{v:.2f}" if v is not None else "-"

        for key, _ in self.metrics:
            val_a = getattr(m_a, key) if m_a else None
            val_b = getattr(m_b, key) if m_b else None

            self.vars[key][0].set(fmt(val_a))
            self.vars[key][1].set(fmt(val_b))

            if val_a is not None and val_b is not None:
                self.vars[key][2].set(f"{val_b - val_a:+.2f}")
            else:
                self.vars[key][2].set("-")

        # 简易结论分析
        if m_b:
            warnings = []
            if m_b.clip_sample_percent > 0:
                warnings.append("发生削波")
            if m_b.peak_dbfs > -0.1:
                warnings.append("峰值过高")

            diff_rms = m_b.rms_dbfs - m_a.rms_dbfs if m_a else 0
            summary = f"结论: RMS变化 {diff_rms:+.1f} dB"
            if warnings:
                summary += "\n警示: " + " ".join(warnings)
                self.lbl_summary.config(text=summary, foreground="red")
            else:
                summary += "\n状态: 信号安全"
                self.lbl_summary.config(text=summary, foreground="green")
        else:
            self.lbl_summary.config(text="等待数据...", foreground="gray")


class MainApp(tk.Tk):
    """主程序窗口类"""

    def __init__(self):
        super().__init__()
        self.title("音频处理器: Limiter & Gain")
        self.geometry("1000x850")

        self.here = Path(__file__).resolve().parent

        # --- 默认路径配置 ---
        self.path_in = tk.StringVar(value=str(self.here / "in.wav"))
        self.path_base = tk.StringVar(value=str(self.here / "base.wav"))
        self.path_out_limiter = tk.StringVar(value=str(self.here / "out_limiter.wav"))
        self.path_out_gain = tk.StringVar(value=str(self.here / "out_gain.wav"))
        # 默认 VST3 插件路径
        self.path_vst = tk.StringVar(value=str(self.here / "WaveShell1-VST3 14.12_x64.vst3"))
        self.path_batch_in = tk.StringVar(value="")

        # 尝试自动查找 import 目录下的文件用于批处理
        try:
            import_files = list((self.here / "import").glob("*.wav"))
            if len(import_files) == 1:
                self.path_batch_in.set(str(import_files[0]))
        except Exception:
            pass

        # --- 处理参数 ---
        # Limiter 参数
        self.var_thresh = tk.DoubleVar(value=-5.0)
        self.var_ceil = tk.DoubleVar(value=-0.3)
        self.var_rel = tk.DoubleVar(value=100.0)
        self.var_auto_rel = tk.BooleanVar(value=True)
        # Gain 参数
        self.var_gain_db = tk.DoubleVar(value=-3.0)

        # 缓存分析结果
        self.metrics_in = None
        self.metrics_limiter = None
        self.metrics_gain = None

        self._init_ui()
        self.after(100, self._ui_tick)  # 启动 UI 刷新循环

        # 自动加载存在的默认文件
        if Path(self.path_in.get()).exists():
            self.track_in.load_file(self.path_in.get())
        if Path(self.path_base.get()).exists():
            self.track_base.load_file(self.path_base.get())

        if not HAS_PYDUB:
            messagebox.showwarning(
                "功能受限",
                "未检测到 pydub 库，MP3 导出功能将不可用。\n请安装: pip install pydub\n并确保已安装 ffmpeg。",
            )

    def _init_ui(self):
        """初始化界面布局"""
        top_container = ttk.Frame(self, padding=5)
        top_container.pack(side="top", fill="x", padx=5, pady=2)

        # 文件选择区域
        top_frame = ttk.LabelFrame(top_container, text="文件设置", padding=5)
        top_frame.pack(side="left", fill="both", expand=True)

        self._build_file_row(top_frame, "输入音频:", self.path_in, False, self._on_input_changed)
        self._build_file_row(top_frame, "基准音频:", self.path_base, False, self._on_base_changed)
        self._build_file_row(top_frame, "VST3插件:", self.path_vst, False, None)
        self._build_file_row(top_frame, "批处理源:", self.path_batch_in, False, None)

        self.btn_batch = ttk.Button(top_container, text="一键批处理", command=self._start_batch_process)
        self.btn_batch.pack(side="right", fill="y", padx=(5, 0))

        # 主网格区域：左侧为轨道，右侧为控制面板
        grid_frame = ttk.Frame(self)
        grid_frame.pack(side="top", fill="both", expand=True, padx=5, pady=2)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

        # 左侧：4个轨道面板
        self.track_base = TrackPanel(grid_frame, "基准音频 (Base/Reference)", self._on_play_req, on_load_ok=None)
        self.track_base.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.track_in = TrackPanel(grid_frame, "输入音频 (Input)", self._on_play_req, self._on_file_in_loaded)
        self.track_in.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        self.track_limiter = TrackPanel(
            grid_frame,
            "Limiter 结果",
            self._on_play_req,
            lambda p: self._on_file_out_loaded(p, "limiter"),
        )
        self.track_limiter.grid(row=2, column=0, sticky="nsew", padx=2, pady=2)

        self.track_gain = TrackPanel(
            grid_frame,
            "Gain 结果",
            self._on_play_req,
            lambda p: self._on_file_out_loaded(p, "gain"),
        )
        self.track_gain.grid(row=3, column=0, sticky="nsew", padx=2, pady=2)

        # 右侧：控制参数与分析表
        right_panel = ttk.Frame(grid_frame)
        right_panel.grid(row=0, column=1, rowspan=4, sticky="nsew")

        self.panel_limiter = self._build_limiter_ui(right_panel)
        self.panel_limiter.pack(fill="x", padx=2, pady=2)

        self.panel_gain = self._build_gain_ui(right_panel)
        self.panel_gain.pack(fill="x", padx=2, pady=2)

        self.analysis_panel = AnalysisPanel(right_panel)
        self.analysis_panel.pack(fill="both", expand=True, padx=2, pady=2)

    def _build_limiter_ui(self, parent):
        """构建 Limiter 模块设置面板"""
        frame = ttk.LabelFrame(parent, text="Limiter 模块 (VST)", padding=5)
        self._build_file_row(frame, "输出路径:", self.path_out_limiter, True, None)
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=2)

        self._build_slider(frame, "阈值 (Threshold)", self.var_thresh, -30, 0, 0.1)
        self._build_slider(frame, "上限 (Ceiling)", self.var_ceil, -30, 0, 0.01)
        self._build_slider(frame, "释放 (Release ms)", self.var_rel, 0.1, 1000, 1.0)

        ttk.Checkbutton(frame, text="自动释放 (Auto Release)", variable=self.var_auto_rel).pack(anchor="w", pady=1)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=2)
        self.btn_run_limiter = ttk.Button(btn_frame, text="执行 Limiter 处理", command=self._start_limiter)
        self.btn_run_limiter.pack(side="left", fill="x", expand=True, padx=(0, 2))

        btn_mp3 = ttk.Button(
            btn_frame,
            text="转存为 MP3",
            command=lambda: self.convert_to_mp3(self.path_out_limiter.get()),
        )
        btn_mp3.pack(side="right", fill="x", padx=(2, 0))

        return frame

    def _build_gain_ui(self, parent):
        """构建 Gain 模块设置面板"""
        frame = ttk.LabelFrame(parent, text="Gain 模块 (Pedalboard)", padding=5)
        self._build_file_row(frame, "输出路径:", self.path_out_gain, True, None)
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=2)

        self._build_slider(frame, "增益 (Gain dB)", self.var_gain_db, -60, 20, 0.1)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=2)
        self.btn_run_gain = ttk.Button(btn_frame, text="执行 Gain 处理", command=self._start_gain)
        self.btn_run_gain.pack(side="left", fill="x", expand=True, padx=(0, 2))

        btn_mp3 = ttk.Button(
            btn_frame,
            text="转存为 MP3",
            command=lambda: self.convert_to_mp3(self.path_out_gain.get()),
        )
        btn_mp3.pack(side="right", fill="x", padx=(2, 0))

        return frame

    def _build_file_row(self, parent, label, var, is_save, callback):
        """构建文件选择行"""
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=1)
        ttk.Label(f, text=label, width=10).pack(side="left")
        ttk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True)

        def browse():
            path = (
                filedialog.asksaveasfilename(filetypes=[("WAV", "*.wav")]) if is_save else filedialog.askopenfilename()
            )
            if path:
                var.set(path)
                if callback:
                    callback()

        ttk.Button(f, text="...", width=4, command=browse).pack(side="right", padx=5)

    def _build_slider(self, parent, label, var, vmin, vmax, res):
        """构建带输入框的滑块控件"""
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        top = ttk.Frame(f)
        top.pack(fill="x")
        ttk.Label(top, text=label).pack(side="left")

        entry_var = tk.StringVar(value=f"{var.get():.2f}")

        # 实现输入框回车确认
        def on_entry_validate(event=None):
            try:
                val = float(entry_var.get())
                val = max(vmin, min(val, vmax))
                var.set(val)
                entry_var.set(f"{val:.2f}")
            except ValueError:
                entry_var.set(f"{var.get():.2f}")
            if event and event.keysym == "Return":
                parent.focus_set()

        entry = ttk.Entry(top, textvariable=entry_var, width=8, justify="right")
        entry.pack(side="right")
        entry.bind("<Return>", on_entry_validate)
        entry.bind("<FocusOut>", on_entry_validate)

        def on_scale_move(v):
            val = float(v)
            entry_var.set(f"{val:.2f}")

        scale = ttk.Scale(f, from_=vmin, to=vmax, variable=var, command=on_scale_move)
        scale.pack(fill="x")

    def convert_to_mp3(self, wav_path_str):
        """交互式 MP3 转换（弹出保存对话框）"""
        if not HAS_PYDUB:
            messagebox.showerror("错误", "请先安装 pydub 和 ffmpeg")
            return
        if not wav_path_str or not Path(wav_path_str).exists():
            messagebox.showwarning("提示", "源 WAV 文件不存在")
            return

        p_wav = Path(wav_path_str)
        initial_name = p_wav.stem + ".mp3"
        out_path = filedialog.asksaveasfilename(
            title="保存为 MP3",
            defaultextension=".mp3",
            filetypes=[("MP3 Audio", "*.mp3")],
            initialfile=initial_name,
            initialdir=p_wav.parent,
        )
        if not out_path:
            return

        def _do_convert():
            try:
                self._silent_mp3_convert(p_wav, Path(out_path))
                messagebox.showinfo("成功", f"MP3 已保存至:\n{out_path}")
            except Exception as e:
                messagebox.showerror("转换失败", str(e))

        threading.Thread(target=_do_convert, daemon=True).start()

    def _silent_mp3_convert(self, src_wav: Path, dest_mp3: Path):
        """静默 MP3 转换后端逻辑"""
        if not HAS_PYDUB:
            raise RuntimeError("Missing pydub")
        audio = AudioSegment.from_wav(str(src_wav))
        audio.export(str(dest_mp3), format="mp3", bitrate="320k")
        self.log(f"Generated MP3: {dest_mp3.name}")

    def _start_batch_process(self):
        """
        开始一键批处理。
        根据输入源生成：中音量(原件)、大音量(Limiter处理)、小音量(Gain处理) 三个版本。
        """
        if not HAS_PYDUB:
            if not messagebox.askyesno("警告", "未检测到pydub库，无法生成MP3文件。\n是否仅生成WAV文件继续？"):
                return

        manual_path_str = self.path_batch_in.get().strip()
        src_wav = None

        # 确定源文件逻辑
        if manual_path_str and Path(manual_path_str).exists():
            src_wav = Path(manual_path_str)
        else:
            import_dir = self.here / "import"
            if not import_dir.exists():
                messagebox.showerror("错误", f"既未指定文件，也找不到 import 文件夹:\n{import_dir}")
                return

            wavs = list(import_dir.glob("*.wav"))
            if len(wavs) != 1:
                messagebox.showerror(
                    "错误",
                    f"未指定源文件，且 import 文件夹中不是唯一的1个wav文件。\n当前 import 中找到: {len(wavs)} 个",
                )
                return
            src_wav = wavs[0]
            self.path_batch_in.set(str(src_wav))

        export_dir = self.here / "export"
        vst_path = Path(self.path_vst.get())
        if not vst_path.exists():
            messagebox.showerror("错误", f"VST3 插件路径无效:\n{vst_path}")
            return

        # 预加载插件（在主线程进行，避免某些VST的兼容性问题）
        loaded_plugin = None
        try:
            info = sf.info(str(src_wav))
            # 自动判断单声道/立体声并加载对应的插件名称（针对 Waves L1）
            plugin_name = "L1 limiter Stereo" if info.channels == 2 else "L1 limiter Mono"

            self.log(f"正在主线程加载 VST 插件: {plugin_name}...")
            loaded_plugin = VST3Plugin(str(vst_path), plugin_name=plugin_name)
        except Exception as e:
            messagebox.showerror("插件加载失败", f"无法加载 VST3 插件:\n{e}")
            self.log(traceback.format_exc())
            return

        self.btn_batch.config(state="disabled")

        # 在后台线程运行批处理逻辑
        threading.Thread(target=self._batch_worker, args=(src_wav, export_dir, loaded_plugin), daemon=True).start()

    def _batch_worker(self, src_wav, export_dir, plugin):
        """批处理工作线程"""
        try:
            self.log(f"开始批处理: {src_wav.name}")
            export_dir.mkdir(exist_ok=True)
            stem = src_wav.stem

            name_loud = f"{stem}（音量大）"
            name_quiet = f"{stem}（音量小）"
            name_mid = f"{stem}（音量中）"

            # 1. 处理中音量 (直接复制原件)
            self.log("处理: 音量中 (Original)...")
            p_mid_wav = export_dir / f"{name_mid}.wav"
            p_mid_mp3 = export_dir / f"{name_mid}.mp3"

            shutil.copy(src_wav, p_mid_wav)
            if HAS_PYDUB:
                self._silent_mp3_convert(p_mid_wav, p_mid_mp3)

            # 读取音频数据准备进行 DSP 处理
            info = sf.info(str(src_wav))
            audio, sr = sf.read(str(src_wav), dtype="float32", always_2d=True)

            # 2. 处理大音量 (应用 Limiter)
            self.log("处理: 音量大 (Limiter)...")
            p_loud_wav = export_dir / f"{name_loud}.wav"
            p_loud_mp3 = export_dir / f"{name_loud}.mp3"

            # 硬编码 Limiter 参数
            set_bool(plugin, "bypass", False)
            set_bool(plugin, "auto_release", True)
            set_param_db_linear(plugin, "threshold", -4.0)
            set_param_db_linear(plugin, "ceiling", -0.3)

            # 设置 Release 参数 (需要映射)
            p_rel = plugin.parameters["release"]
            target_rel = float(np.clip(100.0, p_rel.min_value, p_rel.max_value))
            raw_rel = (target_rel - p_rel.min_value) / (p_rel.max_value - p_rel.min_value)
            p_rel.raw_value = float(np.clip(raw_rel, 0, 1))

            board_loud = Pedalboard([plugin])
            processed_loud = board_loud(audio, sr)
            sf.write(str(p_loud_wav), processed_loud, sr, subtype=info.subtype)

            if HAS_PYDUB:
                self._silent_mp3_convert(p_loud_wav, p_loud_mp3)

            # 3. 处理小音量 (应用 -3dB Gain)
            self.log("处理: 音量小 (Gain)...")
            p_quiet_wav = export_dir / f"{name_quiet}.wav"
            p_quiet_mp3 = export_dir / f"{name_quiet}.mp3"

            board_quiet = Pedalboard([Gain(gain_db=-3.0)])
            processed_quiet = board_quiet(audio, sr)
            sf.write(str(p_quiet_wav), processed_quiet, sr, subtype=info.subtype)

            if HAS_PYDUB:
                self._silent_mp3_convert(p_quiet_wav, p_quiet_mp3)

            self.log("批处理完成!")
            messagebox.showinfo("完成", f"处理完成！\n文件已保存至:\n{export_dir}")

        except Exception as e:
            self.log(f"批处理错误: {e}")
            self.log(traceback.format_exc())
            messagebox.showerror("批处理失败", str(e))

        finally:
            self.after(0, lambda: self.btn_batch.config(state="normal"))

    def log(self, msg):
        print(f"[System] {msg}")

    def _on_input_changed(self):
        """输入文件改变时重新加载"""
        if Path(self.path_in.get()).exists():
            self.track_in.load_file(self.path_in.get())

    def _on_base_changed(self):
        """基准文件改变时重新加载"""
        if Path(self.path_base.get()).exists():
            self.track_base.load_file(self.path_base.get())

    def _on_play_req(self, requester):
        """互斥播放逻辑：当一个轨道请求播放时，暂停其他所有轨道"""
        tracks = [self.track_base, self.track_in, self.track_limiter, self.track_gain]
        for t in tracks:
            if t != requester:
                t.pause_external()

    def _ui_tick(self):
        """全局 UI 刷新循环"""
        self.track_in.ui_tick()
        self.track_limiter.ui_tick()
        self.track_gain.ui_tick()
        self.track_base.ui_tick()
        self.after(100, self._ui_tick)

    def _on_file_in_loaded(self, path):
        """文件加载后的回调：触发后台分析"""
        threading.Thread(target=self._analyze_bg, args=(path, "in"), daemon=True).start()

    def _on_file_out_loaded(self, path, source_type):
        """处理后文件加载的回调：触发后台分析"""
        threading.Thread(target=self._analyze_bg, args=(path, source_type), daemon=True).start()

    def _analyze_bg(self, path, target_type):
        """后台分析线程"""
        try:
            m = analyze_audio_file(path)
            if target_type == "in":
                self.metrics_in = m
            elif target_type == "limiter":
                self.metrics_limiter = m
                self.after(
                    0,
                    lambda: self.analysis_panel.update_data(self.metrics_in, self.metrics_limiter, "Input vs Limiter"),
                )
            elif target_type == "gain":
                self.metrics_gain = m
                self.after(
                    0,
                    lambda: self.analysis_panel.update_data(self.metrics_in, self.metrics_gain, "Input vs Gain"),
                )
        except Exception as e:
            self.log(f"分析失败: {e}")

    def _process_generic(self, plugins_list, p_in, p_out, finish_callback):
        """通用的音频处理线程逻辑（读取 -> Pedalboard处理 -> 写入 -> 回调）"""
        try:
            info = sf.info(str(p_in))
            self.log("读取音频...")
            audio, sr = sf.read(str(p_in), dtype="float32", always_2d=True)

            self.log("应用插件处理...")
            board = Pedalboard(plugins_list)
            processed = board(audio, sr)

            self.log(f"写入文件: {p_out.name}...")
            sf.write(str(p_out), processed, sr, subtype=info.subtype)
            self.log("处理完成")

            self.after(0, lambda: finish_callback(str(p_out)))

        except Exception as e:
            self.log(f"处理失败: {e}")
            self.log(traceback.format_exc())
            messagebox.showerror("处理错误", str(e))
        finally:
            self.after(0, self._reset_buttons)

    def _reset_buttons(self):
        self.btn_run_limiter.config(state="normal")
        self.btn_run_gain.config(state="normal")

    def _start_limiter(self):
        """Limiter 按钮点击事件处理"""
        p_in = Path(self.path_in.get())
        p_out = Path(self.path_out_limiter.get())
        p_vst = Path(self.path_vst.get())

        if not p_in.exists() or not p_vst.exists():
            messagebox.showerror("错误", "输入文件或VST路径不存在")
            return

        self.btn_run_limiter.config(state="disabled")

        try:
            info = sf.info(str(p_in))
            name = "L1 limiter Stereo" if info.channels == 2 else "L1 limiter Mono"

            plugin = VST3Plugin(str(p_vst), plugin_name=name)

            # 获取 UI 参数
            thresh = self.var_thresh.get()
            ceil = self.var_ceil.get()
            rel = self.var_rel.get()
            auto = self.var_auto_rel.get()

            # 设置插件参数
            set_bool(plugin, "bypass", False)
            set_bool(plugin, "auto_release", auto)
            set_param_db_linear(plugin, "threshold", thresh)
            set_param_db_linear(plugin, "ceiling", ceil)
            if not auto:
                p_rel = plugin.parameters["release"]
                target = float(np.clip(rel, p_rel.min_value, p_rel.max_value))
                raw = (target - p_rel.min_value) / (p_rel.max_value - p_rel.min_value)
                p_rel.raw_value = float(np.clip(raw, 0, 1))

            threading.Thread(
                target=self._process_generic,
                args=([plugin], p_in, p_out, self.track_limiter.load_file),
                daemon=True,
            ).start()

        except Exception as e:
            messagebox.showerror("插件加载错误", str(e))
            self.btn_run_limiter.config(state="normal")

    def _start_gain(self):
        """Gain 按钮点击事件处理"""
        p_in = Path(self.path_in.get())
        p_out = Path(self.path_out_gain.get())

        if not p_in.exists():
            messagebox.showerror("错误", "输入文件不存在")
            return

        self.btn_run_gain.config(state="disabled")

        try:
            db_val = self.var_gain_db.get()
            plugin = Gain(gain_db=db_val)

            threading.Thread(
                target=self._process_generic,
                args=([plugin], p_in, p_out, self.track_gain.load_file),
                daemon=True,
            ).start()

        except Exception as e:
            messagebox.showerror("Gain 错误", str(e))
            self.btn_run_gain.config(state="normal")

    def destroy(self):
        """关闭窗口时清理资源"""
        self.track_in.close()
        self.track_limiter.close()
        self.track_gain.close()
        self.track_base.close()
        super().destroy()


if __name__ == "__main__":
    # 尝试设置 Windows 高 DPI 感知，防止界面模糊
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = MainApp()
    app.mainloop()
