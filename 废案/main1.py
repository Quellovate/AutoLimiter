import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pygame
import subprocess
import struct
import threading
import traceback
import time
import shutil


try:
    from pedalboard import Pedalboard, Limiter, Gain
    from pedalboard.io import AudioFile

    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    print("警告: pedalboard 库未安装，将使用 ffmpeg 流式解码。")


try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("警告: pydub 库未安装。")


class MixerManager:
    """管理 pygame.mixer 的初始化，支持动态切换采样率"""

    _current_frequency = None
    _current_channels = None
    _initialized = False

    @classmethod
    def init(cls, frequency=44100, channels=2):
        """初始化或重新初始化 mixer"""
        if cls._initialized:
            if (
                cls._current_frequency == frequency
                and cls._current_channels == channels
            ):
                return
            pygame.mixer.quit()

        pygame.mixer.init(frequency=frequency, channels=channels, size=-16)
        cls._current_frequency = frequency
        cls._current_channels = channels
        cls._initialized = True
        print(f"[Mixer] 初始化: {frequency}Hz, {channels}ch")

    @classmethod
    def get_current_config(cls):
        return cls._current_frequency, cls._current_channels

    @classmethod
    def quit(cls):
        if cls._initialized:
            pygame.mixer.quit()
            cls._initialized = False


def get_audio_duration_ffprobe(filepath):
    """使用 ffprobe 快速获取音频时长（秒），不解码整个文件"""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filepath,
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, creationflags=creationflags
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"ffprobe 获取时长失败: {e}")
        return None


def get_audio_metadata_ffprobe(filepath):
    """使用 ffprobe 获取音频元数据"""
    metadata = {
        "sample_rate": None,
        "channels": None,
        "bit_depth": None,
        "codec": None,
        "duration": None,
        "bitrate": None,
        "file_size": None,
    }

    try:
        if os.path.exists(filepath):
            metadata["file_size"] = os.path.getsize(filepath)

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,bits_per_sample,codec_name,bit_rate:format=duration,bit_rate",
            "-of",
            "json",
            filepath,
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, creationflags=creationflags
        )

        if result.returncode == 0:
            import json

            data = json.loads(result.stdout)

            if "streams" in data and len(data["streams"]) > 0:
                stream = data["streams"][0]
                metadata["sample_rate"] = (
                    int(stream.get("sample_rate", 0))
                    if stream.get("sample_rate")
                    else None
                )
                metadata["channels"] = (
                    int(stream.get("channels", 0)) if stream.get("channels") else None
                )
                metadata["bit_depth"] = (
                    int(stream.get("bits_per_sample", 0))
                    if stream.get("bits_per_sample")
                    else None
                )
                metadata["codec"] = stream.get("codec_name", "Unknown")
                if stream.get("bit_rate"):
                    metadata["bitrate"] = int(stream.get("bit_rate", 0))

            if "format" in data:
                fmt = data["format"]
                if fmt.get("duration"):
                    metadata["duration"] = float(fmt.get("duration"))
                if fmt.get("bit_rate") and not metadata["bitrate"]:
                    metadata["bitrate"] = int(fmt.get("bit_rate", 0))

    except Exception as e:
        print(f"ffprobe 获取元数据失败: {e}")

    return metadata


def is_pygame_compatible_format(filepath, metadata):
    """检查文件是否可以直接被 pygame 播放"""
    ext = os.path.splitext(filepath)[1].lower()
    codec = metadata.get("codec", "").lower()

    if ext in [".wav", ".ogg"]:
        return True
    if ext == ".mp3":

        return True

    return False


def convert_to_wav_preserve_quality(input_path, output_wav_path, metadata=None):
    """
    转换音频到 WAV 格式，保留原始采样率和通道数
    """
    cmd = ["ffmpeg", "-y", "-i", input_path]

    if metadata:
        if metadata.get("sample_rate"):
            cmd.extend(["-ar", str(metadata["sample_rate"])])
        if metadata.get("channels"):
            cmd.extend(["-ac", str(metadata["channels"])])

    cmd.extend(["-c:a", "pcm_s16le", "-f", "wav", output_wav_path])

    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.run(
            cmd, capture_output=True, text=True, creationflags=creationflags
        )
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg 错误: {process.stderr}")
        return True
    except FileNotFoundError:
        raise RuntimeError("未找到 ffmpeg，请确保已安装并添加到 PATH。")


def copy_file_as_temp(src_path, dst_path):
    """直接复制文件作为临时文件"""
    shutil.copy2(src_path, dst_path)
    return True


def format_file_size(size_bytes):
    """格式化文件大小"""
    if size_bytes is None:
        return "N/A"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def format_duration(seconds):
    """格式化时长"""
    if seconds is None or seconds <= 0:
        return "N/A"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def format_bitrate(bitrate):
    """格式化比特率"""
    if bitrate is None or bitrate <= 0:
        return "N/A"
    return f"{bitrate // 1000} kbps"


class PlayerPanel(tk.LabelFrame):
    """
    通用播放器面板，支持保留原始音频属性
    """

    def __init__(
        self, parent, title, panel_id, stop_other_callback, show_load_btn=True
    ):
        super().__init__(parent, text=title, padx=10, pady=10)
        self.panel_id = panel_id
        self.stop_other_callback = stop_other_callback

        self.is_playing = False
        self.has_started = False
        self.mixer_owned = False
        self.duration_sec = 0
        self.current_file_path = None
        self.playback_file_path = None
        self.temp_wav = f"temp_playback_{self.panel_id}.wav"
        self.is_dragging = False
        self.is_loading = False
        self.metadata = {}
        self.playback_metadata = {}
        self.is_converted = False

        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, pady=5)

        if show_load_btn:
            self.btn_select = tk.Button(
                top_frame, text="选择文件", command=self.select_file
            )
            self.btn_select.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_file = tk.Label(top_frame, text="[空]", anchor="w", fg="gray")
        self.lbl_file.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.metadata_frame = tk.LabelFrame(self, text="音频信息", padx=5, pady=5)
        self.metadata_frame.pack(fill=tk.X, pady=(5, 5))

        self.metadata_labels = {}
        metadata_items = [
            ("sample_rate", "采样率:"),
            ("channels", "通道:"),
            ("bit_depth", "位深:"),
            ("codec", "编码:"),
            ("duration_fmt", "时长:"),
            ("bitrate", "比特率:"),
            ("file_size", "文件大小:"),
        ]

        for i, (key, label_text) in enumerate(metadata_items):
            row = i // 2
            col = (i % 2) * 2

            lbl_name = tk.Label(self.metadata_frame, text=label_text, fg="gray")
            lbl_name.grid(row=row, column=col, sticky="w", padx=(5, 2))

            lbl_value = tk.Label(self.metadata_frame, text="--", fg="gray")
            lbl_value.grid(row=row, column=col + 1, sticky="w", padx=(0, 15))

            self.metadata_labels[key] = lbl_value

        self.lbl_convert_status = tk.Label(
            self.metadata_frame, text="", fg="gray", font=("Arial", 7)
        )
        self.lbl_convert_status.grid(
            row=4, column=0, columnspan=4, sticky="w", padx=5, pady=(5, 0)
        )

        self.metadata_frame.columnconfigure(1, weight=1)
        self.metadata_frame.columnconfigure(3, weight=1)

        self.var_progress = tk.DoubleVar()
        self.scale_progress = tk.Scale(
            self,
            variable=self.var_progress,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            showvalue=0,
        )
        self.scale_progress.pack(fill=tk.X, pady=5)
        self.scale_progress.bind("<ButtonPress-1>", self.on_slider_click)
        self.scale_progress.bind("<ButtonRelease-1>", self.on_slider_release)
        self.scale_progress.bind("<B1-Motion>", self.on_slider_drag)

        ctrl_frame = tk.Frame(self)
        ctrl_frame.pack(fill=tk.X, pady=5)

        self.lbl_time = tk.Label(ctrl_frame, text="00:00 / 00:00")
        self.lbl_time.pack(side=tk.LEFT)

        self.btn_play = tk.Button(
            ctrl_frame,
            text="播放",
            command=self.toggle_play,
            width=8,
            state=tk.DISABLED,
        )
        self.btn_play.pack(side=tk.RIGHT)

        self.update_progress_loop()

    def update_metadata_display(self, metadata=None):
        """更新元数据显示"""
        if metadata is None:
            metadata = self.metadata

        if metadata.get("sample_rate"):
            self.metadata_labels["sample_rate"].config(
                text=f"{metadata['sample_rate']} Hz"
            )
        else:
            self.metadata_labels["sample_rate"].config(text="--")

        channels = metadata.get("channels")
        if channels:
            ch_text = (
                "单声道"
                if channels == 1
                else f"{channels} 声道" if channels > 2 else "立体声"
            )
            self.metadata_labels["channels"].config(text=ch_text)
        else:
            self.metadata_labels["channels"].config(text="--")

        if metadata.get("bit_depth"):
            self.metadata_labels["bit_depth"].config(
                text=f"{metadata['bit_depth']}-bit"
            )
        else:
            self.metadata_labels["bit_depth"].config(text="--")

        if metadata.get("codec"):
            self.metadata_labels["codec"].config(text=metadata["codec"].upper())
        else:
            self.metadata_labels["codec"].config(text="--")

        self.metadata_labels["duration_fmt"].config(
            text=format_duration(metadata.get("duration"))
        )
        self.metadata_labels["bitrate"].config(
            text=format_bitrate(metadata.get("bitrate"))
        )
        self.metadata_labels["file_size"].config(
            text=format_file_size(metadata.get("file_size"))
        )

        if self.is_converted:
            self.lbl_convert_status.config(
                text=f"⚠ 已转换为 WAV 播放 (原格式: {metadata.get('codec', 'unknown').upper()})",
                fg="orange",
            )
        else:
            self.lbl_convert_status.config(text="✓ 原始格式直接播放", fg="green")

    def clear_metadata_display(self):
        """清空元数据显示"""
        for key in self.metadata_labels:
            self.metadata_labels[key].config(text="--")
        self.metadata = {}
        self.playback_metadata = {}
        self.lbl_convert_status.config(text="")

    def _calculate_slider_value_from_event(self, event):
        slider_length = self.scale_progress.winfo_width()
        slider_max = float(self.scale_progress.cget("to"))
        padding = 8
        effective_length = slider_length - 2 * padding
        if effective_length <= 0:
            return 0
        click_x = event.x - padding
        click_x = max(0, min(click_x, effective_length))
        ratio = click_x / effective_length
        target_value = ratio * slider_max
        return target_value

    def on_slider_click(self, event):
        if not self.current_file_path or self.is_loading:
            return
        self.is_dragging = True
        target_value = self._calculate_slider_value_from_event(event)
        self.var_progress.set(target_value)
        self.update_time_label(target_value)

    def on_slider_drag(self, event):
        if not self.is_dragging:
            return
        target_value = self._calculate_slider_value_from_event(event)
        self.var_progress.set(target_value)
        self.update_time_label(target_value)

    def on_slider_release(self, event):
        if not self.current_file_path or self.is_loading:
            self.is_dragging = False
            return
        target = self.var_progress.get()
        self.is_dragging = False
        self.stop_other_callback(self.panel_id)
        self.load_to_mixer_and_play(start_pos=target)

    def select_file(self):
        if self.is_loading:
            return
        path = filedialog.askopenfilename(
            filetypes=[
                ("Audio", "*.mp3 *.wav *.mp4 *.m4a *.flac *.ogg *.aac"),
                ("All", "*.*"),
            ]
        )
        if path:
            self.load_audio_file_async(path)

    def load_audio_file_async(self, path):
        self.is_loading = True
        self.stop_playback(reset_ui=True)
        self.clear_metadata_display()
        self.lbl_file.config(text="分析中...", fg="blue")
        self.btn_play.config(state=tk.DISABLED)
        self.update()
        thread = threading.Thread(target=self._load_worker, args=(path,), daemon=True)
        thread.start()

    def _load_worker(self, path):
        try:

            metadata = get_audio_metadata_ffprobe(path)
            duration = metadata.get("duration") or get_audio_duration_ffprobe(path)
            metadata["duration"] = duration

            need_convert = not is_pygame_compatible_format(path, metadata)

            if need_convert:

                self.after(0, lambda: self.lbl_file.config(text="转换中...", fg="blue"))
                convert_to_wav_preserve_quality(path, self.temp_wav, metadata)
                playback_path = self.temp_wav
                is_converted = True

                playback_metadata = get_audio_metadata_ffprobe(self.temp_wav)
            else:

                ext = os.path.splitext(path)[1].lower()
                if ext == ".wav":

                    copy_file_as_temp(path, self.temp_wav)
                    playback_path = self.temp_wav
                else:

                    playback_path = path

                is_converted = False
                playback_metadata = metadata.copy()

            self.after(
                0,
                lambda: self._on_load_complete(
                    path,
                    duration,
                    metadata,
                    playback_path,
                    playback_metadata,
                    is_converted,
                ),
            )

        except Exception as e:
            self.after(0, lambda: self._on_load_error(str(e)))

    def _on_load_complete(
        self, path, duration, metadata, playback_path, playback_metadata, is_converted
    ):
        self.is_loading = False
        self.current_file_path = path
        self.playback_file_path = playback_path
        self.duration_sec = duration
        self.metadata = metadata
        self.playback_metadata = playback_metadata
        self.is_converted = is_converted

        self._setup_player_ui(duration, os.path.basename(path))
        self.update_metadata_display(self.metadata)

    def _on_load_error(self, error_msg):
        self.is_loading = False
        messagebox.showerror("加载失败", error_msg)
        self.lbl_file.config(text="加载失败", fg="red")
        self.clear_metadata_display()

    def release_file_handle(self):
        """释放文件句柄"""
        if self.mixer_owned:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except:
                pass
        self.is_playing = False
        self.mixer_owned = False
        self.has_started = False
        self.btn_play.config(text="播放")

    def load_from_temp_file(self, temp_path, label_text, duration_sec, metadata=None):
        """直接加载一个已处理好的临时文件 (用于预览面板)"""
        try:
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"临时文件不存在: {temp_path}")

            file_size = os.path.getsize(temp_path)
            if file_size < 100:
                raise ValueError(f"文件大小异常 ({file_size} bytes)")

            with open(temp_path, "rb") as f:
                riff = f.read(4)
                if riff != b"RIFF":
                    raise ValueError(f"不是有效的 WAV 文件")
                f.read(4)
                wave = f.read(4)
                if wave != b"WAVE":
                    raise ValueError(f"不是有效的 WAV 文件")

            if metadata is None:
                metadata = get_audio_metadata_ffprobe(temp_path)

            if duration_sec <= 0:
                duration_sec = (
                    metadata.get("duration")
                    or get_audio_duration_ffprobe(temp_path)
                    or 0
                )

            metadata["duration"] = duration_sec
            metadata["file_size"] = file_size

            self.stop_playback(reset_ui=True)
            self.playback_file_path = temp_path
            self.current_file_path = temp_path
            self.metadata = metadata
            self.playback_metadata = metadata
            self.is_converted = True

            self._setup_player_ui(duration_sec, label_text)
            self.update_metadata_display(metadata)

        except Exception as e:
            print(f"[DEBUG load_from_temp_file] 加载失败: {e}")
            traceback.print_exc()
            messagebox.showerror("预览加载失败", f"无法加载处理后的文件:\n{e}")
            self.lbl_file.config(text="加载失败", fg="red")
            self.clear_metadata_display()

    def _setup_player_ui(self, duration, name_text):
        self.duration_sec = duration
        self.lbl_file.config(text=name_text, fg="black")
        self.scale_progress.config(to=self.duration_sec)
        self.var_progress.set(0)
        self.update_time_label(0)
        self.btn_play.config(state=tk.NORMAL, text="播放")
        self.has_started = False
        self.mixer_owned = False

    def toggle_play(self):
        if not self.current_file_path or self.is_loading:
            return
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
            self.btn_play.config(text="播放")
        else:
            self.stop_other_callback(self.panel_id)
            if not self.mixer_owned:
                start_pos = self.var_progress.get() if self.has_started else 0
                self.load_to_mixer_and_play(start_pos)
            else:
                try:
                    pygame.mixer.music.unpause()
                    self.is_playing = True
                    self.btn_play.config(text="暂停")
                except:
                    self.load_to_mixer_and_play(self.var_progress.get())

    def load_to_mixer_and_play(self, start_pos=0):
        """加载到 Pygame Mixer 并播放，根据文件元数据初始化 mixer"""
        try:
            if not os.path.exists(self.playback_file_path):
                raise FileNotFoundError(f"播放文件不存在: {self.playback_file_path}")

            sample_rate = self.playback_metadata.get("sample_rate") or 44100
            channels = self.playback_metadata.get("channels") or 2

            MixerManager.init(frequency=sample_rate, channels=channels)

            pygame.mixer.music.load(self.playback_file_path)
            pygame.mixer.music.play(start=start_pos)

            self.is_playing = True
            self.has_started = True
            self.mixer_owned = True
            self.btn_play.config(text="暂停")

        except Exception as e:
            print(f"[DEBUG load_to_mixer_and_play] 播放错误: {e}")
            traceback.print_exc()
            messagebox.showerror(
                "播放错误", f"无法播放音频:\n{e}\n\n文件: {self.playback_file_path}"
            )

    def stop_playback(self, reset_ui=False):
        self.is_playing = False
        self.mixer_owned = False
        self.btn_play.config(text="播放")
        if reset_ui:
            self.has_started = False
            self.var_progress.set(0)
            self.update_time_label(0)

    def force_stop_logic(self):
        self.is_playing = False
        self.mixer_owned = False
        self.btn_play.config(text="播放")

    def update_progress_loop(self):
        if self.is_playing and not self.is_dragging:
            cur = self.var_progress.get() + 0.1
            if cur <= self.duration_sec:
                self.var_progress.set(cur)
                self.update_time_label(cur)
            else:
                self.stop_playback(reset_ui=True)
                pygame.mixer.music.stop()
        self.after(100, self.update_progress_loop)

    def update_time_label(self, sec):
        m, s = divmod(int(sec), 60)
        dm, ds = divmod(int(self.duration_sec), 60)
        self.lbl_time.config(text=f"{m:02}:{s:02} / {dm:02}:{ds:02}")


class LimiterControlPanel(tk.LabelFrame):
    def __init__(self, parent, apply_callback):
        super().__init__(parent, text="Limiter 参数调节", padx=10, pady=10)
        self.apply_callback = apply_callback

        engine_text = (
            "引擎: Pedalboard (专业)"
            if PEDALBOARD_AVAILABLE
            else "引擎: FFmpeg + 简易算法"
        )
        engine_color = "green" if PEDALBOARD_AVAILABLE else "orange"
        lbl_engine = tk.Label(
            self, text=engine_text, fg=engine_color, font=("Arial", 9, "italic")
        )
        lbl_engine.pack(anchor="w", pady=(0, 10))

        frame1 = tk.Frame(self)
        frame1.pack(fill=tk.X)
        tk.Label(frame1, text="输入增益 (Drive):").pack(side=tk.LEFT)
        self.lbl_gain_val = tk.Label(frame1, text="3.0 dB", width=8, anchor="e")
        self.lbl_gain_val.pack(side=tk.RIGHT)

        self.var_gain = tk.DoubleVar(value=3.0)
        self.scale_gain = tk.Scale(
            self,
            variable=self.var_gain,
            from_=0,
            to=18,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            showvalue=0,
            command=lambda v: self.lbl_gain_val.config(text=f"{float(v):.1f} dB"),
        )
        self.scale_gain.pack(fill=tk.X, pady=(0, 10))

        frame2 = tk.Frame(self)
        frame2.pack(fill=tk.X)
        tk.Label(frame2, text="输出天花板 (Ceiling):").pack(side=tk.LEFT)
        self.lbl_ceil_val = tk.Label(frame2, text="-0.3 dB", width=8, anchor="e")
        self.lbl_ceil_val.pack(side=tk.RIGHT)

        self.var_ceiling = tk.DoubleVar(value=-0.3)
        self.scale_ceiling = tk.Scale(
            self,
            variable=self.var_ceiling,
            from_=-10.0,
            to=0.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            showvalue=0,
            command=lambda v: self.lbl_ceil_val.config(text=f"{float(v):.1f} dB"),
        )
        self.scale_ceiling.pack(fill=tk.X, pady=(0, 10))

        frame3 = tk.Frame(self)
        frame3.pack(fill=tk.X)
        tk.Label(frame3, text="释放时间 (Release):").pack(side=tk.LEFT)
        self.lbl_release_val = tk.Label(frame3, text="100 ms", width=8, anchor="e")
        self.lbl_release_val.pack(side=tk.RIGHT)

        self.var_release = tk.DoubleVar(value=100.0)
        self.scale_release = tk.Scale(
            self,
            variable=self.var_release,
            from_=10,
            to=500,
            resolution=10,
            orient=tk.HORIZONTAL,
            showvalue=0,
            command=lambda v: self.lbl_release_val.config(text=f"{int(float(v))} ms"),
        )
        self.scale_release.pack(fill=tk.X, pady=(0, 15))

        if not PEDALBOARD_AVAILABLE:
            self.scale_release.config(state=tk.DISABLED)
            self.lbl_release_val.config(text="N/A", fg="gray")

        self.btn_apply = tk.Button(
            self,
            text="▶ 生成预览 (Apply to Preview)",
            command=self.on_apply,
            bg="gray",
            font=("Arial", 10, "bold"),
        )
        self.btn_apply.pack(fill=tk.X, pady=(10, 0))

    def on_apply(self):
        gain = self.var_gain.get()
        ceiling = self.var_ceiling.get()
        release = self.var_release.get()
        self.apply_callback(gain, ceiling, release)


class FourQuadrantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("音频 Limiter 模拟与对比工作台 (保留原始属性版)")
        self.root.geometry("900x800")
        self.root.minsize(800, 700)

        MixerManager.init()

        self.processed_temp_file = "temp_processed_preview.wav"

        self.paned_window = tk.PanedWindow(
            root, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.RAISED
        )
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.frame_left = tk.Frame(self.paned_window)
        self.paned_window.add(self.frame_left, width=430)

        self.panel_ref = PlayerPanel(
            self.frame_left, "1. 基准音频 (Reference)", "ref", self.on_any_play
        )
        self.panel_ref.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.panel_source = PlayerPanel(
            self.frame_left,
            "2. 原始音频 (Source for Limiter)",
            "source",
            self.on_any_play,
        )
        self.panel_source.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.frame_right = tk.Frame(self.paned_window)
        self.paned_window.add(self.frame_right, width=450)

        self.panel_controls = LimiterControlPanel(
            self.frame_right, apply_callback=self.process_audio
        )
        self.panel_controls.pack(fill=tk.X, pady=(0, 5))

        self.panel_preview = PlayerPanel(
            self.frame_right,
            "3. 效果预览 (Processed Preview)",
            "preview",
            self.on_any_play,
            show_load_btn=False,
        )
        self.panel_preview.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.init_defaults()

    def init_defaults(self):
        default_path = os.path.join(os.getcwd(), "music.wav")
        if os.path.exists(default_path):
            self.panel_ref.load_audio_file_async(default_path)
        else:
            self.panel_ref.lbl_file.config(text="[请加载 music.wav]")

        self.panel_source.lbl_file.config(text="[请选择待处理文件]", fg="gray")

    def on_any_play(self, active_id):
        panels = [self.panel_ref, self.panel_source, self.panel_preview]
        for p in panels:
            if p.panel_id != active_id:
                p.force_stop_logic()

    def process_audio(self, gain_db, ceiling_db, release_ms):
        """核心处理函数 - 保留原始采样率和通道数"""
        source_path = self.panel_source.current_file_path
        source_metadata = self.panel_source.metadata

        if not source_path:
            messagebox.showwarning("提示", "请先在左下角 (Source) 加载一个音频文件。")
            return

        if not os.path.exists(source_path):
            messagebox.showwarning("提示", f"源文件不存在: {source_path}")
            return

        self.panel_controls.btn_apply.config(state=tk.DISABLED, text="处理中...")
        self.root.config(cursor="watch")
        self.root.update()

        try:
            self.panel_preview.release_file_handle()
            self.panel_preview.clear_metadata_display()

            if os.path.exists(self.processed_temp_file):
                try:
                    os.remove(self.processed_temp_file)
                except PermissionError:
                    time.sleep(0.2)
                    os.remove(self.processed_temp_file)

            if PEDALBOARD_AVAILABLE:
                self._process_with_pedalboard(
                    source_path, gain_db, ceiling_db, release_ms, source_metadata
                )
            elif PYDUB_AVAILABLE:
                self._process_with_pydub(
                    source_path, gain_db, ceiling_db, source_metadata
                )
            else:
                raise RuntimeError("需要安装 pedalboard 或 pydub 来处理音频。")

            if not os.path.exists(self.processed_temp_file):
                raise FileNotFoundError(f"处理后的文件未生成")

            metadata = get_audio_metadata_ffprobe(self.processed_temp_file)
            duration = metadata.get("duration", 0)

            if duration <= 0:
                duration = get_audio_duration_ffprobe(self.processed_temp_file) or 0
                metadata["duration"] = duration

            if duration <= 0:
                raise ValueError("无法获取有效的音频时长")

            info_text = f"Limiter (G:+{gain_db:.1f}, C:{ceiling_db:.1f}dB)"
            self.panel_preview.load_from_temp_file(
                self.processed_temp_file, info_text, duration, metadata
            )

        except Exception as e:
            print(f"处理失败: {e}")
            traceback.print_exc()
            messagebox.showerror("处理失败", str(e))
        finally:
            self.panel_controls.btn_apply.config(
                state=tk.NORMAL, text="▶ 生成预览 (Apply to Preview)"
            )
            self.root.config(cursor="")

    def _process_with_pedalboard(
        self, source_path, gain_db, ceiling_db, release_ms, source_metadata=None
    ):
        """使用 Pedalboard 处理，保留原始采样率"""
        with AudioFile(source_path) as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate
            num_channels = audio.shape[0] if len(audio.shape) > 1 else 1

        board = Pedalboard(
            [
                Gain(gain_db=gain_db),
                Limiter(threshold_db=ceiling_db, release_ms=release_ms),
            ]
        )
        processed = board(audio, samplerate)

        with AudioFile(self.processed_temp_file, "w", samplerate, num_channels) as f:
            f.write(processed)

    def _process_with_pydub(
        self, source_path, gain_db, ceiling_db, source_metadata=None
    ):
        """使用 Pydub 处理，保留原始采样率"""
        audio = AudioSegment.from_file(source_path)
        processed = audio + gain_db

        if processed.max_dBFS > ceiling_db:
            reduction = ceiling_db - processed.max_dBFS
            processed = processed + reduction

        original_sample_rate = (
            source_metadata.get("sample_rate", 44100) if source_metadata else 44100
        )
        processed.export(
            self.processed_temp_file,
            format="wav",
            parameters=["-ar", str(original_sample_rate)],
        )

    def cleanup(self):
        MixerManager.quit()
        files_to_clean = [
            "temp_playback_ref.wav",
            "temp_playback_source.wav",
            self.processed_temp_file,
        ]
        for f in files_to_clean:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass


if __name__ == "__main__":
    root = tk.Tk()
    app = FourQuadrantApp(root)

    def on_closing():
        app.panel_ref.stop_playback()
        app.panel_source.stop_playback()
        app.panel_preview.stop_playback()
        try:
            pygame.mixer.music.stop()
        except:
            pass
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
