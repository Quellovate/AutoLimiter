from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as np
except Exception as e:
    raise SystemExit("Missing dependency: numpy. Install with:\n" "  pip install numpy\n\n" f"Import error: {e}")

try:
    import sounddevice as sd
    import soundfile as sf
except Exception as e:
    raise SystemExit(
        "Missing dependencies. Install with:\n" "  pip install sounddevice soundfile\n\n" f"Import error: {e}"
    )


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass(frozen=True)
class AudioInfo:
    samplerate: int
    channels: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.samplerate if self.samplerate > 0 else 0.0


@dataclass(frozen=True)
class Metrics:
    samplerate: int
    channels: int
    duration_s: float
    peak_dbfs: float
    rms_dbfs: float
    crest_db: float
    lra_db: float | None
    clip_sample_percent: float
    clip_frame_percent: float


def _dbfs(x: float, *, floor_db: float = -120.0) -> float:
    x = float(x)
    if x <= 0:
        return floor_db
    return max(20.0 * float(np.log10(x)), floor_db)


def analyze_audio_file(path: str) -> Metrics:
    """
    计算音频指标：
    - 峰值 (dBFS)
    - RMS (dBFS)
    - 峰值因子 (dB)
    - 响度范围 (LRA)
    - 削波率
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
        block_rms_db: list[float] = []

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
        peak_dbfs = _dbfs(peak)
        rms_dbfs = _dbfs(rms)
        crest_db = peak_dbfs - rms_dbfs

        clip_sample_percent = (clip_samples / max(total_samples, 1)) * 100.0
        clip_frame_percent = (clip_frames / max(frames_total, 1)) * 100.0

        if len(block_rms_db) >= 2:
            p10, p95 = np.percentile(np.asarray(block_rms_db, dtype=np.float64), [10, 95])
            lra_db = float(p95 - p10)
        else:
            lra_db = None

        return Metrics(
            samplerate=samplerate,
            channels=channels,
            duration_s=duration_s,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            crest_db=crest_db,
            lra_db=lra_db,
            clip_sample_percent=float(clip_sample_percent),
            clip_frame_percent=float(clip_frame_percent),
        )


class AudioPlayer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sf: sf.SoundFile | None = None
        self._stream: sd.OutputStream | None = None

        self._info: AudioInfo | None = None
        self._paused = True
        self._finished = False
        self._position_frames = 0

    def load(self, path: str) -> AudioInfo:
        with self._lock:
            self.close()
            try:
                f = sf.SoundFile(path, mode="r")
            except Exception as e:
                raise RuntimeError(f"Cannot open audio file: {e}") from e

            if f.channels < 1:
                f.close()
                raise RuntimeError("Invalid channel count")

            self._sf = f
            self._info = AudioInfo(
                samplerate=int(f.samplerate),
                channels=int(f.channels),
                frames=int(len(f)),
            )
            self._position_frames = 0
            self._sf.seek(0)
            self._paused = True
            self._finished = False
            return self._info

    def _ensure_stream(self) -> None:
        assert self._info is not None
        if self._stream is not None:
            return

        def callback(outdata, frames, time_info, status):
            with self._lock:
                if self._sf is None or self._info is None or self._paused:
                    outdata.fill(0)
                    return

                data = self._sf.read(frames, dtype="float32", always_2d=True)
                read_frames = int(data.shape[0])

                if read_frames > 0:
                    outdata[:read_frames, :] = data

                if read_frames < frames:
                    outdata[read_frames:, :].fill(0)
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

    def play(self) -> None:
        with self._lock:
            if self._sf is None or self._info is None:
                return
            self._ensure_stream()
            assert self._stream is not None
            if not self._stream.active:
                self._stream.start()
            self._paused = False
            self._finished = False

    def pause(self, *, close_stream: bool = True) -> None:
        with self._lock:
            self._paused = True
            if close_stream:
                self._close_stream_only()

    def seek_seconds(self, seconds: float) -> None:
        with self._lock:
            if self._sf is None or self._info is None:
                return
            target = int(float(seconds) * self._info.samplerate)
            target = max(0, min(target, self._info.frames))
            self._sf.seek(target)
            self._position_frames = target
            self._finished = False

    def get_state(self) -> tuple[AudioInfo | None, bool, bool, float]:
        with self._lock:
            info = self._info
            paused = self._paused
            finished = self._finished
            pos_s = (self._position_frames / info.samplerate) if info and info.samplerate else 0.0
            return info, paused, finished, pos_s

    def _close_stream_only(self) -> None:
        if self._stream is None:
            return
        try:
            if self._stream.active:
                self._stream.stop()
        finally:
            self._stream.close()
            self._stream = None

    def close(self) -> None:
        with self._lock:
            self._paused = True
            self._finished = False
            self._position_frames = 0
            self._close_stream_only()
            if self._sf is not None:
                self._sf.close()
                self._sf = None
            self._info = None


class TrackPanel(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        title: str,
        default_path: Path,
        on_request_play: callable,
        on_file_loaded: callable,
    ) -> None:
        super().__init__(master)
        self.player = AudioPlayer()
        self._on_request_play = on_request_play
        self._on_file_loaded = on_file_loaded

        self._dragging = False
        self._loaded_path: str | None = None

        self._title = title
        self._path_var = tk.StringVar(value=str(default_path))
        self._time_var = tk.StringVar(value="00:00 / 00:00")
        self._status_var = tk.StringVar(value="Idle")
        self._progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()

        if default_path.exists():
            self.load_file(str(default_path), show_error=False)

    def _build_ui(self) -> None:
        ttk.Label(self, text=self._title, font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 4))

        row1 = ttk.Frame(self)
        row1.pack(fill="x")
        ttk.Label(row1, text="File:").pack(side="left")
        ttk.Entry(row1, textvariable=self._path_var).pack(side="left", fill="x", expand=True, padx=(8, 8))
        ttk.Button(row1, text="Browse...", command=self._browse).pack(side="left")
        ttk.Button(row1, text="Load", command=self._load_from_entry).pack(side="left", padx=(8, 0))

        row2 = ttk.Frame(self)
        row2.pack(fill="x", pady=(8, 0))
        self._play_btn = ttk.Button(row2, text="Play", command=self._on_play_pause)
        self._play_btn.pack(side="left")
        ttk.Label(row2, textvariable=self._time_var).pack(side="left", padx=(12, 0))

        row3 = ttk.Frame(self)
        row3.pack(fill="x", pady=(6, 0))
        self._scale = ttk.Scale(
            row3,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self._progress_var,
            command=self._on_slider_move,
        )
        self._scale.pack(side="left", fill="x", expand=True)
        self._scale.bind("<ButtonPress-1>", self._on_slider_press, add=True)
        self._scale.bind("<ButtonRelease-1>", self._on_slider_release, add=True)

        row4 = ttk.Frame(self)
        row4.pack(fill="x", pady=(6, 0))
        ttk.Label(row4, textvariable=self._status_var).pack(side="left")

    def _browse(self) -> None:
        initial = Path(self._path_var.get()).expanduser()
        initialdir = initial.parent if initial.parent.exists() else Path.cwd()
        path = filedialog.askopenfilename(
            title=f"Select audio file ({self._title})",
            initialdir=str(initialdir),
            filetypes=[
                ("Audio files (soundfile)", "*.wav *.flac *.ogg *.aiff *.aif *.aifc"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._path_var.set(path)
            self.load_file(path, show_error=True)

    def _load_from_entry(self) -> None:
        path = self._path_var.get().strip()
        if not path:
            return
        self.load_file(path, show_error=True)

    def load_file(self, path: str, *, show_error: bool) -> None:
        try:
            info = self.player.load(path)
        except Exception as e:
            self._loaded_path = None
            self._play_btn.configure(text="Play")
            self._status_var.set("Load failed")
            if show_error:
                messagebox.showerror("Load error", f"{self._title}: {e}")
            return

        self._loaded_path = path
        self._status_var.set(
            f"Loaded: {Path(path).name} | {info.samplerate} Hz | {info.channels} ch | {format_time(info.duration)}"
        )
        self._progress_var.set(0.0)
        self._scale.configure(to=max(info.duration, 0.001))
        self._time_var.set(f"00:00 / {format_time(info.duration)}")
        self._play_btn.configure(text="Play")
        self._on_file_loaded()

    def _on_play_pause(self) -> None:
        if self._loaded_path is None:
            self._load_from_entry()
            if self._loaded_path is None:
                return

        info, paused, finished, pos = self.player.get_state()
        if info is None:
            return

        if paused:
            self._on_request_play(self)
            self.player.play()
            self._play_btn.configure(text="Pause")
        else:
            self.player.pause(close_stream=True)
            self._play_btn.configure(text="Play")

    def _on_slider_press(self, _event) -> None:
        self._dragging = True

    def _on_slider_release(self, _event) -> None:
        self._dragging = False
        self.player.seek_seconds(float(self._progress_var.get()))

    def _on_slider_move(self, _value: str) -> None:
        info, _, _, _ = self.player.get_state()
        if info and self._dragging:
            cur = float(self._progress_var.get())
            self._time_var.set(f"{format_time(cur)} / {format_time(info.duration)}")

    def ui_tick(self) -> None:
        info, paused, finished, pos = self.player.get_state()
        if info is None:
            return

        if not self._dragging:
            self._progress_var.set(pos)
            self._time_var.set(f"{format_time(pos)} / {format_time(info.duration)}")

        if finished:
            self.player.pause(close_stream=True)
            self._play_btn.configure(text="Play")
            self._status_var.set(f"Finished: {Path(self._loaded_path).name}" if self._loaded_path else "Finished")
        else:
            self._play_btn.configure(text=("Play" if paused else "Pause"))

    def pause_external(self) -> None:
        info, paused, _, _ = self.player.get_state()
        if info is None:
            return
        if not paused:
            self.player.pause(close_stream=True)
        self._play_btn.configure(text="Play")

    def get_loaded_path(self) -> str | None:
        return self._loaded_path

    def close(self) -> None:
        self.player.close()


class AnalysisPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, *, on_refresh: callable) -> None:
        super().__init__(master)
        self._on_refresh = on_refresh

        self._status_var = tk.StringVar(value="等待加载")
        self._path_a_var = tk.StringVar(value="-")
        self._path_b_var = tk.StringVar(value="-")
        self._summary_var = tk.StringVar(value="-")

        self._vars: dict[str, tuple[tk.StringVar, tk.StringVar, tk.StringVar]] = {}

        self._build_ui()
        self.clear_metrics(keep_paths=False)

    def _build_ui(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x")
        ttk.Label(header, text="数据分析", font=("TkDefaultFont", 11, "bold")).pack(side="left")
        ttk.Button(header, text="刷新", command=self._on_refresh).pack(side="right")

        ttk.Label(self, textvariable=self._status_var, wraplength=380).pack(fill="x", pady=(6, 8))

        paths = ttk.LabelFrame(self, text="文件")
        paths.pack(fill="x")
        paths.columnconfigure(1, weight=1)

        ttk.Label(paths, text="A：").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(paths, textvariable=self._path_a_var, wraplength=360).grid(
            row=0, column=1, sticky="w", padx=8, pady=4
        )
        ttk.Label(paths, text="B：").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(paths, textvariable=self._path_b_var, wraplength=360).grid(
            row=1, column=1, sticky="w", padx=8, pady=4
        )

        table = ttk.LabelFrame(self, text="指标")
        table.pack(fill="both", expand=True, pady=(10, 0))
        for c in range(4):
            table.columnconfigure(c, weight=1)

        ttk.Label(table, text="指标").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(table, text="A").grid(row=0, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(table, text="B").grid(row=0, column=2, sticky="w", padx=8, pady=4)
        ttk.Label(table, text="Δ(B-A)").grid(row=0, column=3, sticky="w", padx=8, pady=4)

        metrics = [
            ("peak_dbfs", "峰值 (dBFS)"),
            ("rms_dbfs", "RMS (dBFS)"),
            ("crest_db", "峰值因子 (dB)"),
            ("lra_db", "响度范围 (dB)"),
            ("clip_sample_percent", "样本削波率 (%)"),
            ("clip_frame_percent", "帧削波率 (%)"),
        ]

        for i, (key, label) in enumerate(metrics, start=1):
            a = tk.StringVar(value="-")
            b = tk.StringVar(value="-")
            d = tk.StringVar(value="-")
            self._vars[key] = (a, b, d)
            ttk.Label(table, text=label, wraplength=170).grid(row=i, column=0, sticky="w", padx=8, pady=4)
            ttk.Label(table, textvariable=a).grid(row=i, column=1, sticky="w", padx=8, pady=4)
            ttk.Label(table, textvariable=b).grid(row=i, column=2, sticky="w", padx=8, pady=4)
            ttk.Label(table, textvariable=d).grid(row=i, column=3, sticky="w", padx=8, pady=4)

        summary = ttk.LabelFrame(self, text="分析结论")
        summary.pack(fill="x", pady=(10, 0))
        ttk.Label(summary, textvariable=self._summary_var, wraplength=430, justify="left").pack(
            fill="x", padx=8, pady=6
        )

    def set_status(self, text: str) -> None:
        self._status_var.set(text)

    def clear_metrics(self, *, keep_paths: bool) -> None:
        if not keep_paths:
            self._path_a_var.set("-")
            self._path_b_var.set("-")
        for a, b, d in self._vars.values():
            a.set("-")
            b.set("-")
            d.set("-")
        self._summary_var.set("-")

    def set_results(
        self,
        *,
        path_a: str | None,
        path_b: str | None,
        metrics_a: Metrics | None,
        metrics_b: Metrics | None,
        error_a: str | None,
        error_b: str | None,
    ) -> None:
        self._path_a_var.set(Path(path_a).name if path_a else "-")
        self._path_b_var.set(Path(path_b).name if path_b else "-")

        if error_a and metrics_a is None:
            self.set_status(f"A 分析失败：{error_a}")
        elif error_b and metrics_b is None:
            self.set_status(f"B 分析失败：{error_b}")
        else:
            self.set_status("分析完成")

        def fmt_db(x: float | None) -> str:
            if x is None:
                return "-"
            return f"{x:.2f}"

        def fmt_pct(x: float | None) -> str:
            if x is None:
                return "-"
            return f"{x:.3f}%"

        if metrics_a is not None:
            self._vars["peak_dbfs"][0].set(f"{fmt_db(metrics_a.peak_dbfs)}")
            self._vars["rms_dbfs"][0].set(f"{fmt_db(metrics_a.rms_dbfs)}")
            self._vars["crest_db"][0].set(f"{fmt_db(metrics_a.crest_db)}")
            self._vars["lra_db"][0].set(fmt_db(metrics_a.lra_db))
            self._vars["clip_sample_percent"][0].set(fmt_pct(metrics_a.clip_sample_percent))
            self._vars["clip_frame_percent"][0].set(fmt_pct(metrics_a.clip_frame_percent))

        if metrics_b is not None:
            self._vars["peak_dbfs"][1].set(f"{fmt_db(metrics_b.peak_dbfs)}")
            self._vars["rms_dbfs"][1].set(f"{fmt_db(metrics_b.rms_dbfs)}")
            self._vars["crest_db"][1].set(f"{fmt_db(metrics_b.crest_db)}")
            self._vars["lra_db"][1].set(fmt_db(metrics_b.lra_db))
            self._vars["clip_sample_percent"][1].set(fmt_pct(metrics_b.clip_sample_percent))
            self._vars["clip_frame_percent"][1].set(fmt_pct(metrics_b.clip_frame_percent))

        if metrics_a is not None and metrics_b is not None:
            self._vars["peak_dbfs"][2].set(fmt_db(metrics_b.peak_dbfs - metrics_a.peak_dbfs))
            self._vars["rms_dbfs"][2].set(fmt_db(metrics_b.rms_dbfs - metrics_a.rms_dbfs))
            self._vars["crest_db"][2].set(fmt_db(metrics_b.crest_db - metrics_a.crest_db))
            if metrics_a.lra_db is not None and metrics_b.lra_db is not None:
                self._vars["lra_db"][2].set(fmt_db(metrics_b.lra_db - metrics_a.lra_db))
            self._vars["clip_sample_percent"][2].set(
                fmt_pct(metrics_b.clip_sample_percent - metrics_a.clip_sample_percent)
            )
            self._vars["clip_frame_percent"][2].set(
                fmt_pct(metrics_b.clip_frame_percent - metrics_a.clip_frame_percent)
            )

            self._summary_var.set(self._build_summary(metrics_a, metrics_b))
        else:
            self._summary_var.set("需加载两份文件")

    def _build_summary(self, a: Metrics, b: Metrics) -> str:
        rms_delta = b.rms_dbfs - a.rms_dbfs
        crest_delta = b.crest_db - a.crest_db
        peak_delta = b.peak_dbfs - a.peak_dbfs

        lines = [
            f"RMS 变化：{rms_delta:+.2f} dB",
            f"峰值因子变化：{crest_delta:+.2f} dB",
            f"峰值变化：{peak_delta:+.2f} dB",
        ]

        if a.lra_db is not None and b.lra_db is not None:
            lines.append(f"响度范围变化：{(b.lra_db - a.lra_db):+.2f} dB")

        if b.clip_sample_percent > 0.01 or b.clip_frame_percent > 0.01:
            lines.append("B 存在削波")
        elif b.peak_dbfs > -0.2:
            lines.append("B 接近 0dB")
        else:
            lines.append("无削波")

        return "\n".join(lines)


class DualPlayerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Dual Audio Player")
        self.geometry("1060x420")
        self.minsize(980, 380)

        base = Path(__file__).resolve().parent
        default1 = base / "in.wav"
        default2 = base / "out.wav"

        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(0, weight=1)

        left = ttk.Frame(container)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.panel1 = TrackPanel(
            left,
            title="音频 A",
            default_path=default1,
            on_request_play=self._on_request_play,
            on_file_loaded=self._request_analysis,
        )
        self.panel1.pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=10)

        self.panel2 = TrackPanel(
            left,
            title="音频 B",
            default_path=default2,
            on_request_play=self._on_request_play,
            on_file_loaded=self._request_analysis,
        )
        self.panel2.pack(fill="x")

        right = ttk.Frame(container)
        right.grid(row=0, column=1, sticky="nsew")
        self.analysis_panel = AnalysisPanel(right, on_refresh=self._request_analysis)
        self.analysis_panel.pack(fill="both", expand=True)

        self._analysis_token = 0
        self._analysis_lock = threading.Lock()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._tick)
        self._request_analysis()

    def _on_request_play(self, who: TrackPanel) -> None:
        other = self.panel2 if who is self.panel1 else self.panel1
        other.pause_external()

    def _tick(self) -> None:
        self.panel1.ui_tick()
        self.panel2.ui_tick()
        self.after(100, self._tick)

    def _on_close(self) -> None:
        try:
            self.panel1.close()
            self.panel2.close()
        finally:
            self.destroy()

    def _request_analysis(self) -> None:
        if not hasattr(self, "panel1") or not hasattr(self, "panel2") or not hasattr(self, "analysis_panel"):
            return

        with self._analysis_lock:
            self._analysis_token += 1
            token = self._analysis_token

        path_a = self.panel1.get_loaded_path()
        path_b = self.panel2.get_loaded_path()

        self.analysis_panel.set_status("分析中")
        self.analysis_panel.clear_metrics(keep_paths=True)

        def worker() -> None:
            result_a: Metrics | None = None
            result_b: Metrics | None = None
            err_a: str | None = None
            err_b: str | None = None

            if path_a:
                try:
                    result_a = analyze_audio_file(path_a)
                except Exception as e:
                    err_a = str(e)
            else:
                err_a = "未加载"

            if path_b:
                try:
                    result_b = analyze_audio_file(path_b)
                except Exception as e:
                    err_b = str(e)
            else:
                err_b = "未加载"

            def apply() -> None:
                with self._analysis_lock:
                    if token != self._analysis_token:
                        return
                self.analysis_panel.set_results(
                    path_a=path_a,
                    path_b=path_b,
                    metrics_a=result_a,
                    metrics_b=result_b,
                    error_a=err_a,
                    error_b=err_b,
                )

            self.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = DualPlayerApp()
    app.mainloop()
