import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import traceback
import dataclasses
from dataclasses import dataclass

try:
    import numpy as np
    import soundfile as sf
    import sounddevice as sd
    from pedalboard import Pedalboard, VST3Plugin
except ImportError as e:
    raise SystemExit(
        f"缺少必要库: {e}\n请运行: pip install numpy soundfile sounddevice pedalboard")


@dataclass(frozen=True)
class AudioInfo:
    """存储音频文件的元数据"""
    samplerate: int       
    channels: int         
    frames: int             

    @property
    def duration(self) -> float:
        """计算音频时长（秒）"""
        return self.frames / self.samplerate if self.samplerate > 0 else 0.0


@dataclass(frozen=True)
class Metrics:
    """存储音频分析的声学指标"""
    samplerate: int
    channels: int
    duration_s: float
    peak_dbfs: float                     
    rms_dbfs: float                      
    crest_db: float                                   
    lra_db: float | None                             
    clip_sample_percent: float          
    clip_frame_percent: float          


def format_time(seconds: float) -> str:
    """将秒数格式化为 MM:SS 或 HH:MM:SS"""
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _dbfs(x: float, *, floor_db: float = -120.0) -> float:
    """将线性振幅转换为 dBFS"""
    x = float(x)
    if x <= 0:
        return floor_db
    return max(20.0 * float(np.log10(x)), floor_db)


def analyze_audio_file(path: str) -> Metrics:
    """
    分析函数：读取音频并计算指标。
    采用分块读取方式。
    """
    clip_threshold = 0.9999           
    loudness_gate_db = -70.0                
    block_seconds = 0.4                       

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

                  
        for block in f.blocks(blocksize=blocksize, dtype="float32", always_2d=True):
            if block.size == 0:
                continue
            abs_block = np.abs(block)
            
                    
            peak = max(peak, float(abs_block.max()))
                             
            sumsq += float(np.sum(block * block))
            total_samples += int(block.size)

                    
            clip_mask = abs_block >= clip_threshold
            clip_samples += int(np.count_nonzero(clip_mask))
            clip_frames += int(np.count_nonzero(np.any(clip_mask, axis=1)))

                                   
            mono = block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
            rms_block = float(np.sqrt(np.mean(mono * mono)))
            rms_block_db = _dbfs(rms_block)
            
                                
            if rms_block_db > loudness_gate_db:
                block_rms_db.append(rms_block_db)

                
        rms = float(np.sqrt(sumsq / max(total_samples, 1)))
        
                                          
        lra_db = None
        if len(block_rms_db) >= 2:
            p10, p95 = np.percentile(np.asarray(
                block_rms_db, dtype=np.float64), [10, 95])
            lra_db = float(p95 - p10)

        return Metrics(
            samplerate=samplerate, channels=channels, duration_s=duration_s,
            peak_dbfs=_dbfs(peak), rms_dbfs=_dbfs(rms), crest_db=_dbfs(peak) - _dbfs(rms),
            lra_db=lra_db,
            clip_sample_percent=(clip_samples / max(total_samples, 1)) * 100.0,
            clip_frame_percent=(clip_frames / max(frames_total, 1)) * 100.0,
        )


class AudioPlayer:
    """
    音频播放器类。
    """
    def __init__(self):
        self._lock = threading.RLock()                    
        self._sf = None                     
        self._stream = None                  
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
            self._info = AudioInfo(
                int(f.samplerate), int(f.channels), int(len(f)))
            self._position_frames = 0
            self._sf.seek(0)
            self._paused = True
            self._finished = False
            return self._info

    def play(self):
        """播放"""
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
        """暂停"""
        with self._lock:
            self._paused = True
            if close_stream and self._stream:
                if self._stream.active:
                    self._stream.stop()
                self._stream.close()
                self._stream = None

    def seek_seconds(self, seconds: float):
        """跳转到指定时间"""
        with self._lock:
            if not self._sf or not self._info:
                return
            target = int(float(seconds) * self._info.samplerate)
            target = max(0, min(target, self._info.frames))
            self._sf.seek(target)
            self._position_frames = target
            self._finished = False

    def get_state(self):
        """获取播放状态"""
        with self._lock:
            pos_s = (self._position_frames /
                     self._info.samplerate) if self._info else 0.0
            return self._info, self._paused, self._finished, pos_s

    def close(self):
        """关闭"""
        self.pause(close_stream=True)
        with self._lock:
            if self._sf:
                self._sf.close()
                self._sf = None
            self._info = None

    def _create_stream(self):
        """创建音频输出流"""
        def callback(outdata, frames, time_info, status):
            with self._lock:
                if not self._sf or self._paused:
                    outdata.fill(0)
                    return
                          
                data = self._sf.read(frames, dtype="float32", always_2d=True)
                read = len(data)
                if read > 0:
                    outdata[:read, :] = data
                                      
                if read < frames:
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
            callback=callback
        )


def set_param_db_linear(plugin: VST3Plugin, name: str, db_value: float) -> None:
    """
    VST 辅助函数：将 dB 值设置给参数。
    """
    if name not in plugin.parameters:
        return
    p = plugin.parameters[name]
                     
    target = float(np.clip(db_value, p.min_value, p.max_value))
                
    raw = (target - p.min_value) / (p.max_value - p.min_value)
    p.raw_value = float(np.clip(raw, 0.0, 1.0))


def set_bool(plugin: VST3Plugin, name: str, enabled: bool) -> None:
    """VST 辅助函数：设置布尔参数"""
    if name in plugin.parameters:
        plugin.parameters[name].raw_value = 1.0 if enabled else 0.0


class TrackPanel(ttk.LabelFrame):
    """轨道控制面板"""

    def __init__(self, master, title, on_play_req, on_load_ok):
        super().__init__(master, text=title, padding=10)
        self.player = AudioPlayer()
        self.on_play_req = on_play_req                 
        self.on_load_ok = on_load_ok           
        self.loaded_path = None
        self.dragging = False                         

        self._create_ui()

    def _create_ui(self):
                                 
        self.lbl_path = ttk.Label(
            self, text="未选择文件", foreground="gray")
        self.lbl_path.pack(fill="x", pady=(0, 5))

        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=5)

        self.btn_play = ttk.Button(
            ctrl, text="播放", command=self._toggle_play, state="disabled")
        self.btn_play.pack(side="left")

        self.lbl_time = ttk.Label(ctrl, text="00:00 / 00:00")
        self.lbl_time.pack(side="right", padx=5)

        self.var_prog = tk.DoubleVar()
        self.scale = ttk.Scale(self, variable=self.var_prog,
                               from_=0, to=1, command=self._on_seek)
        self.scale.pack(fill="x", pady=(5, 0))
                      
        self.scale.bind("<ButtonPress-1>",
                        lambda e: setattr(self, 'dragging', True))
        self.scale.bind("<ButtonRelease-1>", self._on_seek_release)

    def load_file(self, path):
        """加载文件"""
        try:
            info = self.player.load(str(path))
            self.loaded_path = str(path)
            self.lbl_path.config(
                text=f"{Path(path).name}", foreground="black")
            self.scale.config(to=info.duration)
            self.btn_play.config(state="normal")
            self._update_time_label(0, info.duration)
            self.on_load_ok(str(path))
        except Exception as e:
            self.lbl_path.config(text=f"错误: {e}", foreground="red")
            self.btn_play.config(state="disabled")

    def _toggle_play(self):
        """切换播放状态"""
        info, paused, _, _ = self.player.get_state()
        if not info:
            return
        if paused:
                             
            self.on_play_req(self)
            self.player.play()
            self.btn_play.config(text="暂停")
        else:
            self.player.pause()
            self.btn_play.config(text="播放")

    def pause_external(self):
        """外部暂停"""
        self.player.pause()
        self.btn_play.config(text="播放")

    def _on_seek(self, val):
        if self.dragging:
            self.lbl_time.config(text=f"{format_time(float(val))} / ...")

    def _on_seek_release(self, event):
        self.dragging = False
        self.player.seek_seconds(float(self.var_prog.get()))

    def _update_time_label(self, current, total):
        self.lbl_time.config(
            text=f"{format_time(current)} / {format_time(total)}")

    def ui_tick(self):
        """UI 更新"""
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
    """结果分析模块"""

    def __init__(self, master):
        super().__init__(master, text="结果分析", padding=10)
        self.metrics = [
            ("peak_dbfs", "峰值 (dBFS)"),
            ("rms_dbfs", "RMS (dBFS)"),
            ("crest_db", "动态因子 (dB)"),
            ("lra_db", "响度范围 (dB)"),
            ("clip_sample_percent", "削波率 (%)")
        ]
        self.vars = {}
        self._create_grid()

    def _create_grid(self):
                              
        for i in range(4):
            self.columnconfigure(i, weight=1)

        headers = ["指标", "输入 (In)", "输出 (Out)", "变化 (Δ)"]
        for col, text in enumerate(headers):
            ttk.Label(self, text=text, font=("", 9, "bold")).grid(
                row=0, column=col, sticky="w", pady=(0, 5))

        for i, (key, label) in enumerate(self.metrics, 1):
            ttk.Label(self, text=label).grid(
                row=i, column=0, sticky="w", pady=2)
                                 
            v_a = tk.StringVar(value="-")
            v_b = tk.StringVar(value="-")
            v_d = tk.StringVar(value="-")
            self.vars[key] = (v_a, v_b, v_d)
            ttk.Label(self, textvariable=v_a).grid(row=i, column=1, sticky="w")
            ttk.Label(self, textvariable=v_b).grid(row=i, column=2, sticky="w")
            ttk.Label(self, textvariable=v_d).grid(row=i, column=3, sticky="w")

        self.lbl_summary = ttk.Label(
            self, text="无数据", foreground="gray", wraplength=350)
        self.lbl_summary.grid(row=len(self.metrics)+1,
                              column=0, columnspan=4, pady=(10, 0), sticky="w")

    def update_data(self, m_a: Metrics, m_b: Metrics):
        """更新数据"""
        def fmt(v): return f"{v:.2f}" if v is not None else "-"

        for key, _ in self.metrics:
            val_a = getattr(m_a, key) if m_a else None
            val_b = getattr(m_b, key) if m_b else None

            self.vars[key][0].set(fmt(val_a))
            self.vars[key][1].set(fmt(val_b))

            if val_a is not None and val_b is not None:
                self.vars[key][2].set(f"{val_b - val_a:+.2f}")
            else:
                self.vars[key][2].set("-")

                    
        if m_b:
            warnings = []
            if m_b.clip_sample_percent > 0:
                warnings.append("发生削波")
            if m_b.peak_dbfs > -0.1:
                warnings.append("峰值过高")

            diff_rms = m_b.rms_dbfs - m_a.rms_dbfs if m_a else 0

            summary = f"结论: RMS提升 {diff_rms:+.1f} dB"
            if warnings:
                summary += "\n注意: " + " ".join(warnings)
                self.lbl_summary.config(text=summary, foreground="red")
            else:
                summary += "\n状态: 信号正常"
                self.lbl_summary.config(text=summary, foreground="green")
        else:
            self.lbl_summary.config(text="无数据", foreground="gray")


class MainApp(tk.Tk):
    """主程序"""
    def __init__(self):
        super().__init__()
        self.title("L1 Limiter 处理工具")
        self.geometry("900x750")

        self.here = Path(__file__).resolve().parent
                           
        self.path_in = tk.StringVar(value=str(self.here / "in.wav"))
        self.path_out = tk.StringVar(value=str(self.here / "out.wav"))
        self.path_vst = tk.StringVar(
            value=str(self.here / "WaveShell1-VST3 14.12_x64.vst3"))

                      
        self.var_thresh = tk.DoubleVar(value=-6.0)
        self.var_ceil = tk.DoubleVar(value=-0.3)
        self.var_rel = tk.DoubleVar(value=100.0)
        self.var_auto_rel = tk.BooleanVar(value=True)

        self.metrics_a = None
        self.metrics_b = None

        self._init_ui()
        self.after(100, self._ui_tick)             

                       
        if Path(self.path_in.get()).exists():
            self.track_a.load_file(self.path_in.get())

    def _init_ui(self):
                           
        top_frame = ttk.LabelFrame(self, text="文件设置", padding=10)
        top_frame.pack(side="top", fill="x", padx=10, pady=5)

        self._build_file_row(top_frame, "输入音频:", self.path_in,
                             False, self._on_input_changed)
        self._build_file_row(top_frame, "VST3插件:", self.path_vst, False, None)
        self._build_file_row(top_frame, "输出路径:", self.path_out, True, None)

        grid_frame = ttk.Frame(self)
        grid_frame.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)
        grid_frame.rowconfigure(0, weight=1)
        grid_frame.rowconfigure(1, weight=1)

        self.track_a = TrackPanel(
            grid_frame, "输入音频 (Original)", self._on_play_req, self._on_file_a_loaded)
        self.track_a.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.track_b = TrackPanel(
            grid_frame, "输出结果 (Processed)", self._on_play_req, self._on_file_b_loaded)
        self.track_b.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.params_panel = self._build_params_panel(grid_frame)
        self.params_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.analysis_panel = AnalysisPanel(grid_frame)
        self.analysis_panel.grid(
            row=1, column=1, sticky="nsew", padx=5, pady=5)

    def _build_params_panel(self, parent):
                            
        frame = ttk.LabelFrame(parent, text="参数调节", padding=10)

        self._build_slider(frame, "阈值 (Threshold)",
                           self.var_thresh, -30, 0, 0.1)
        self._build_slider(frame, "上限 (Ceiling)", self.var_ceil, -30, 0, 0.01)
        self._build_slider(frame, "释放时间 (Release ms)",
                           self.var_rel, 0.1, 1000, 1.0)

        ttk.Checkbutton(frame, text="自动释放 (Auto Release)",
                        variable=self.var_auto_rel).pack(anchor="w", pady=2)

        self.btn_process = ttk.Button(
            frame, text="开始处理 (Run Limiter)", command=self._start_processing)
        self.btn_process.pack(fill="x", pady=5)

        return frame

    def _build_file_row(self, parent, label, var, is_save, callback):
                             
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=10).pack(side="left")
        ttk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True)

        def browse():
            if is_save:
                path = filedialog.asksaveasfilename(
                    filetypes=[("WAV", "*.wav")])
            else:
                path = filedialog.askopenfilename()
            if path:
                var.set(path)
                if callback:
                    callback()
        ttk.Button(f, text="浏览...", width=6, command=browse).pack(
            side="right", padx=5)

    def _build_slider(self, parent, label, var, vmin, vmax, res):
                          
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=1)
        top = ttk.Frame(f)
        top.pack(fill="x")
        ttk.Label(top, text=label).pack(side="left")
        lbl_val = ttk.Label(top, text="0.0")
        lbl_val.pack(side="right")
        scale = ttk.Scale(f, from_=vmin, to=vmax,
                          variable=var, orient="horizontal")
        scale.pack(fill="x")
        def update(*args): lbl_val.config(text=f"{var.get():.2f}")
        var.trace_add("write", update)
        update()

    def log(self, msg):
        print(f"[System] {msg}")

    def _on_input_changed(self):
        path = self.path_in.get()
        if Path(path).exists():
            self.track_a.load_file(path)

    def _on_play_req(self, requester):
        """互斥播放"""
        if requester == self.track_a:
            self.track_b.pause_external()
        else:
            self.track_a.pause_external()

    def _ui_tick(self):
        """主界面刷新循环"""
        self.track_a.ui_tick()
        self.track_b.ui_tick()
        self.after(100, self._ui_tick)

    def _on_file_a_loaded(self, path):
                      
        threading.Thread(target=self._analyze_bg, args=(
            path, True), daemon=True).start()

    def _on_file_b_loaded(self, path):
                      
        threading.Thread(target=self._analyze_bg, args=(
            path, False), daemon=True).start()

    def _analyze_bg(self, path, is_a):
        """后台分析线程"""
        try:
            m = analyze_audio_file(path)
            if is_a:
                self.metrics_a = m
            else:
                self.metrics_b = m
                            
            self.after(0, lambda: self.analysis_panel.update_data(
                self.metrics_a, self.metrics_b))
        except Exception as e:
            self.log(f"分析失败: {e}")

    def _start_processing(self):
        """开始处理"""
        p_in = Path(self.path_in.get())
        p_out = Path(self.path_out.get())
        p_vst = Path(self.path_vst.get())

        if not p_in.exists() or not p_vst.exists():
            messagebox.showerror("错误", "输入文件或VST路径不存在")
            return

        self.btn_process.config(state="disabled")

        try:
                                
            info = sf.info(str(p_in))
            if info.channels == 1:
                name = "L1 limiter Mono"
            elif info.channels == 2:
                name = "L1 limiter Stereo"
            else:
                raise ValueError(f"不支持的通道数: {info.channels}")

            self.log(f"加载插件: {name}...")
                                                 
            plugin = VST3Plugin(str(p_vst), plugin_name=name)

            t = threading.Thread(target=self._process_thread,
                                 args=(plugin, p_in, p_out, info))
            t.daemon = True
            t.start()

        except Exception as e:
            self.log(f"启动错误: {e}")
            self.log(traceback.format_exc())
            self.btn_process.config(state="normal")

    def _process_thread(self, plugin, p_in, p_out, info):
        """后台处理线程"""
        try:
            thresh = self.var_thresh.get()
            ceil = self.var_ceil.get()
            rel = self.var_rel.get()
            auto = self.var_auto_rel.get()

            self.log("读取音频...")
            audio, sr = sf.read(str(p_in), dtype="float32", always_2d=True)

            self.log("应用参数...")
            set_bool(plugin, "bypass", False)
            set_bool(plugin, "auto_release", auto)
                               
            set_param_db_linear(plugin, "threshold", thresh)
            set_param_db_linear(plugin, "ceiling", ceil)

                                  
            if not auto:
                p_rel = plugin.parameters["release"]
                target = float(np.clip(rel, p_rel.min_value, p_rel.max_value))
                raw = (target - p_rel.min_value) /\
                    (p_rel.max_value - p_rel.min_value)
                p_rel.raw_value = float(np.clip(raw, 0, 1))

            self.log(f"正在渲染 (T={thresh}, C={ceil})...")
                                  
            board = Pedalboard([plugin])
            processed = board(audio, sr)

            self.log("写入文件...")
            sf.write(str(p_out), processed, sr, subtype=info.subtype)
            self.log("处理完成")

                       
            self.after(0, lambda: self._on_process_complete(str(p_out)))

        except Exception as e:
            self.log(f"处理失败: {e}")
        finally:
            self.after(0, lambda: self.btn_process.config(state="normal"))

    def _on_process_complete(self, out_path):
        self.track_b.load_file(out_path)

    def destroy(self):
        """清理资源"""
        self.track_a.close()
        self.track_b.close()
        super().destroy()


if __name__ == "__main__":
                                    
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = MainApp()
    app.mainloop()
