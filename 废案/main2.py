import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pygame
import subprocess
import threading
import traceback
import time
import atexit

from pedalboard import Pedalboard, Limiter, Gain
from pedalboard.io import AudioFile


class TempFileTracker:
    """追踪所有临时文件，确保程序退出时清理"""

    _files = set()

    @classmethod
    def register(cls, filepath):
        cls._files.add(filepath)

    @classmethod
    def unregister(cls, filepath):
        cls._files.discard(filepath)

    @classmethod
    def cleanup_all(cls):
        for f in list(cls._files):
            safe_remove(f)
        cls._files.clear()


atexit.register(TempFileTracker.cleanup_all)


class MixerManager:
    """管理 pygame.mixer 的初始化，支持动态切换采样率"""

    _current_frequency = None
    _current_channels = None
    _initialized = False

    @classmethod
    def init(cls, frequency=44100, channels=2):
        channels = min(channels, 2)

        if cls._initialized:
            if (
                cls._current_frequency == frequency
                and cls._current_channels == channels
            ):
                return
            try:
                pygame.mixer.quit()
            except:
                pass

        try:
            pygame.mixer.init(frequency=frequency, channels=channels, size=-16)
            cls._current_frequency = frequency
            cls._current_channels = channels
            cls._initialized = True
            print(f"[Mixer] 初始化: {frequency}Hz, {channels}ch, 16-bit")
        except Exception as e:
            print(f"[Mixer] 初始化失败: {e}, 尝试默认配置")
            pygame.mixer.init()
            cls._current_frequency = 44100
            cls._current_channels = 2
            cls._initialized = True

    @classmethod
    def quit(cls):
        if cls._initialized:
            try:
                pygame.mixer.quit()
            except:
                pass
            cls._initialized = False


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

            if "streams" in data and data["streams"]:
                stream = data["streams"][0]
                metadata["sample_rate"] = (
                    int(stream["sample_rate"]) if stream.get("sample_rate") else None
                )
                metadata["channels"] = (
                    int(stream["channels"]) if stream.get("channels") else None
                )
                metadata["bit_depth"] = (
                    int(stream["bits_per_sample"])
                    if stream.get("bits_per_sample")
                    else None
                )
                metadata["codec"] = stream.get("codec_name", "Unknown")
                if stream.get("bit_rate"):
                    metadata["bitrate"] = int(stream["bit_rate"])

            if "format" in data:
                fmt = data["format"]
                if fmt.get("duration"):
                    metadata["duration"] = float(fmt["duration"])
                if fmt.get("bit_rate") and not metadata["bitrate"]:
                    metadata["bitrate"] = int(fmt["bit_rate"])

    except Exception as e:
        print(f"ffprobe 获取元数据失败: {e}")

    return metadata


def get_audio_duration_ffprobe(filepath):
    """使用 ffprobe 快速获取音频时长"""
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


def is_pygame_compatible(filepath, metadata):
    """检查文件是否可以直接被 pygame 播放"""
    ext = os.path.splitext(filepath)[1].lower()
    bit_depth = metadata.get("bit_depth", 16)

    if bit_depth and bit_depth > 16:
        return False
    return ext in [".wav", ".ogg", ".mp3"]


def convert_for_playback(input_path, output_path, metadata=None):
    """仅为播放目的转换文件到 pygame 兼容格式"""
    sample_rate = metadata.get("sample_rate", 44100) if metadata else 44100
    channels = min(metadata.get("channels", 2) if metadata else 2, 2)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        output_path,
    ]

    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.run(
            cmd, capture_output=True, text=True, creationflags=creationflags
        )
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg 错误: {process.stderr}")
        TempFileTracker.register(output_path)
        return True
    except FileNotFoundError:
        raise RuntimeError("未找到 ffmpeg，请确保已安装并添加到 PATH。")


def format_file_size(size_bytes):
    if size_bytes is None:
        return "N/A"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def format_duration(seconds):
    if seconds is None or seconds <= 0:
        return "N/A"
    minutes, secs = divmod(int(seconds), 60)
    ms = int((seconds % 1) * 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def format_bitrate(bitrate):
    if bitrate is None or bitrate <= 0:
        return "N/A"
    return f"{bitrate // 1000} kbps"


def safe_remove(filepath):
    """安全删除文件"""
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
            TempFileTracker.unregister(filepath)
            print(f"[Cleanup] 已删除: {filepath}")
        except Exception as e:
            print(f"[Cleanup] 删除失败 {filepath}: {e}")


def ensure_export_dir():
    """确保导出目录存在"""
    export_dir = os.path.join(os.getcwd(), "export")
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print(f"[Export] 创建导出目录: {export_dir}")
    return export_dir


class PlayerPanel(tk.LabelFrame):
    """通用播放器面板 - 紧凑布局"""

    def __init__(
        self, parent, title, panel_id, stop_other_callback, show_load_btn=True
    ):
        super().__init__(parent, text=title, padx=5, pady=5)
        self.panel_id = panel_id
        self.stop_other_callback = stop_other_callback

        self.is_playing = False
        self.has_started = False
        self.mixer_owned = False
        self.duration_sec = 0
        self.current_file_path = None
        self.playback_file_path = None
        self.temp_playback_file = f"temp_playback_{self.panel_id}_16bit.wav"
        self.is_dragging = False
        self.is_loading = False
        self.metadata = {}
        self.is_converted_for_playback = False
        self.temp_file_ready = False

        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, pady=1)

        if show_load_btn:
            self.btn_select = tk.Button(
                top_frame, text="选择文件", command=self.select_file, font=("Arial", 10)
            )
            self.btn_select.pack(side=tk.LEFT, padx=(0, 5))

        self.lbl_file = tk.Label(
            top_frame,
            text="[空]",
            anchor="w",
            fg="#666",
            wraplength=180,
            font=("Arial", 10),
        )
        self.lbl_file.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.metadata_frame = tk.LabelFrame(self, text="音频信息", padx=3, pady=2)
        self.metadata_frame.pack(fill=tk.X, pady=1)

        self.metadata_labels = {}
        metadata_items = [
            ("sample_rate", "采样率:"),
            ("channels", "通道:"),
            ("bit_depth", "位深:"),
            ("codec", "编码:"),
            ("duration_fmt", "时长:"),
            ("bitrate", "比特率:"),
            ("file_size", "大小:"),
        ]

        for i, (key, label_text) in enumerate(metadata_items):
            row, col = i // 2, (i % 2) * 2
            tk.Label(
                self.metadata_frame, text=label_text, fg="#555", font=("Arial", 8)
            ).grid(row=row, column=col, sticky="w", padx=(2, 1))
            lbl_value = tk.Label(
                self.metadata_frame,
                text="--",
                fg="#000",
                font=("Arial", 7, "bold"),
                anchor="w",
            )
            lbl_value.grid(row=row, column=col + 1, sticky="w", padx=(0, 8))
            self.metadata_labels[key] = lbl_value

        self.lbl_playback_status = tk.Label(
            self.metadata_frame, text="", fg="gray", font=("Arial", 7)
        )
        self.lbl_playback_status.grid(
            row=4, column=0, columnspan=4, sticky="w", padx=2, pady=(2, 0)
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
            length=200,
        )
        self.scale_progress.pack(fill=tk.X, pady=1)
        self.scale_progress.bind("<ButtonPress-1>", self.on_slider_click)
        self.scale_progress.bind("<ButtonRelease-1>", self.on_slider_release)
        self.scale_progress.bind("<B1-Motion>", self.on_slider_drag)

        ctrl_frame = tk.Frame(self)
        ctrl_frame.pack(fill=tk.X, pady=1)

        self.lbl_time = tk.Label(ctrl_frame, text="00:00 / 00:00", font=("Arial", 10))
        self.lbl_time.pack(side=tk.LEFT)

        self.btn_play = tk.Button(
            ctrl_frame,
            text="播放",
            command=self.toggle_play,
            width=6,
            state=tk.DISABLED,
            font=("Arial", 10),
        )
        self.btn_play.pack(side=tk.RIGHT)

        self.update_progress_loop()

    def update_metadata_display(self, metadata=None):
        if metadata is None:
            metadata = self.metadata

        self.metadata_labels["sample_rate"].config(
            text=(
                f"{metadata['sample_rate']} Hz" if metadata.get("sample_rate") else "--"
            )
        )

        channels = metadata.get("channels")
        if channels:
            ch_text = (
                "单声道"
                if channels == 1
                else "立体声" if channels == 2 else f"{channels}ch"
            )
            self.metadata_labels["channels"].config(text=ch_text)
        else:
            self.metadata_labels["channels"].config(text="--")

        self.metadata_labels["bit_depth"].config(
            text=f"{metadata['bit_depth']}bit" if metadata.get("bit_depth") else "--"
        )
        self.metadata_labels["codec"].config(
            text=metadata["codec"].upper() if metadata.get("codec") else "--"
        )
        self.metadata_labels["duration_fmt"].config(
            text=format_duration(metadata.get("duration"))
        )
        self.metadata_labels["bitrate"].config(
            text=format_bitrate(metadata.get("bitrate"))
        )
        self.metadata_labels["file_size"].config(
            text=format_file_size(metadata.get("file_size"))
        )

    def update_playback_status(self):
        if self.is_converted_for_playback:
            bit_depth = self.metadata.get("bit_depth", 16)
            self.lbl_playback_status.config(
                text=f"⚠ 播放临时转16bit (原{bit_depth}bit保留)", fg="orange"
            )
        else:
            self.lbl_playback_status.config(text="✓ 原始格式播放", fg="green")

    def clear_metadata_display(self):
        for lbl in self.metadata_labels.values():
            lbl.config(text="--")
        self.metadata = {}
        self.lbl_playback_status.config(text="")

    def _calculate_slider_value_from_event(self, event):
        slider_length = self.scale_progress.winfo_width()
        padding = 8
        effective_length = slider_length - 2 * padding
        if effective_length <= 0:
            return 0
        click_x = max(0, min(event.x - padding, effective_length))
        return (click_x / effective_length) * float(self.scale_progress.cget("to"))

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

        self._cleanup_old_temp_file()

        self.clear_metadata_display()
        self.lbl_file.config(text="分析中...", fg="blue")
        self.btn_play.config(state=tk.DISABLED)
        self.update()
        threading.Thread(target=self._load_worker, args=(path,), daemon=True).start()

    def _cleanup_old_temp_file(self):
        try:
            pygame.mixer.music.unload()
        except:
            pass
        time.sleep(0.05)

        if self.temp_file_ready:
            safe_remove(self.temp_playback_file)
            self.temp_file_ready = False
            self.is_converted_for_playback = False

    def _load_worker(self, path):
        try:
            metadata = get_audio_metadata_ffprobe(path)
            metadata["duration"] = metadata.get(
                "duration"
            ) or get_audio_duration_ffprobe(path)
            self.after(
                0, lambda: self._on_load_complete(path, metadata["duration"], metadata)
            )
        except Exception as e:
            self.after(0, lambda: self._on_load_error(str(e)))

    def _on_load_complete(self, path, duration, metadata):
        self.is_loading = False
        self.current_file_path = path
        self.duration_sec = duration
        self.metadata = metadata
        self.playback_file_path = None
        self.is_converted_for_playback = False
        self.temp_file_ready = False

        self._setup_player_ui(duration, os.path.basename(path))
        self.update_metadata_display(self.metadata)
        self.lbl_playback_status.config(text="")

    def _on_load_error(self, error_msg):
        self.is_loading = False
        messagebox.showerror("加载失败", error_msg)
        self.lbl_file.config(text="加载失败", fg="red")
        self.clear_metadata_display()

    def release_file_handle(self):
        if self.mixer_owned:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except:
                pass
        self.is_playing = False
        self.mixer_owned = False
        self.btn_play.config(text="播放")

    def load_from_processed_file(
        self, file_path, label_text, duration_sec, metadata=None
    ):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            file_size = os.path.getsize(file_path)
            if file_size < 100:
                raise ValueError(f"文件大小异常 ({file_size} bytes)")

            if metadata is None:
                metadata = get_audio_metadata_ffprobe(file_path)

            if duration_sec <= 0:
                duration_sec = (
                    metadata.get("duration")
                    or get_audio_duration_ffprobe(file_path)
                    or 0
                )

            metadata["duration"] = duration_sec
            metadata["file_size"] = file_size

            self.stop_playback(reset_ui=True)
            self._cleanup_old_temp_file()

            self.current_file_path = file_path
            self.metadata = metadata
            self.playback_file_path = None
            self.is_converted_for_playback = False
            self.temp_file_ready = False

            self._setup_player_ui(duration_sec, label_text)
            self.update_metadata_display(metadata)
            self.lbl_playback_status.config(text="")

        except Exception as e:
            print(f"[DEBUG load_from_processed_file] 加载失败: {e}")
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

    def _prepare_for_playback(self):
        if not self.current_file_path or not os.path.exists(self.current_file_path):
            raise FileNotFoundError("源文件不存在")

        if is_pygame_compatible(self.current_file_path, self.metadata):
            self.is_converted_for_playback = False
            return self.current_file_path

        if self.temp_file_ready and os.path.exists(self.temp_playback_file):
            self.is_converted_for_playback = True
            return self.temp_playback_file

        try:
            pygame.mixer.music.unload()
        except:
            pass

        time.sleep(0.05)

        print(f"[Playback] 文件需要转换为 16-bit: {self.current_file_path}")
        convert_for_playback(
            self.current_file_path, self.temp_playback_file, self.metadata
        )

        self.is_converted_for_playback = True
        self.temp_file_ready = True
        return self.temp_playback_file

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
                self.load_to_mixer_and_play(
                    self.var_progress.get() if self.has_started else 0
                )
            else:
                try:
                    pygame.mixer.music.unpause()
                    self.is_playing = True
                    self.btn_play.config(text="暂停")
                except:
                    self.load_to_mixer_and_play(self.var_progress.get())

    def load_to_mixer_and_play(self, start_pos=0):
        try:
            playback_path = self._prepare_for_playback()
            self.playback_file_path = playback_path
            self.update_playback_status()

            MixerManager.init(
                frequency=self.metadata.get("sample_rate") or 44100,
                channels=min(self.metadata.get("channels") or 2, 2),
            )

            pygame.mixer.music.load(playback_path)
            pygame.mixer.music.play(start=start_pos)

            self.is_playing = True
            self.has_started = True
            self.mixer_owned = True
            self.btn_play.config(text="暂停")

        except Exception as e:
            print(f"[DEBUG load_to_mixer_and_play] 播放错误: {e}")
            traceback.print_exc()
            messagebox.showerror("播放错误", f"无法播放音频:\n{e}")

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

    def cleanup_temp_files(self):
        try:
            pygame.mixer.music.unload()
        except:
            pass
        safe_remove(self.temp_playback_file)
        self.temp_file_ready = False


class LimiterControlPanel(tk.LabelFrame):
    def __init__(self, parent, apply_callback):
        super().__init__(parent, text="Limiter 参数", padx=5, pady=5)
        self.apply_callback = apply_callback

        tk.Label(
            self,
            text="引擎: Pedalboard",
            fg="green",
            font=("Arial", 8, "italic"),
        ).pack(anchor="w", pady=(0, 3))

        frame1 = tk.Frame(self)
        frame1.pack(fill=tk.X)
        tk.Label(frame1, text="增益 (Gain):", font=("Arial", 10)).pack(side=tk.LEFT)
        self.lbl_gain_val = tk.Label(
            frame1, text="0.0 dB", width=8, anchor="e", font=("Arial", 10)
        )
        self.lbl_gain_val.pack(side=tk.RIGHT)

        self.var_gain = tk.DoubleVar(value=0.0)
        self.scale_gain = tk.Scale(
            self,
            variable=self.var_gain,
            from_=-18,
            to=18,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            showvalue=0,
            command=self._update_gain_label,
            length=200,
        )
        self.scale_gain.pack(fill=tk.X, pady=(0, 3))

        frame2 = tk.Frame(self)
        frame2.pack(fill=tk.X)
        tk.Label(frame2, text="天花板 (Ceiling):", font=("Arial", 10)).pack(
            side=tk.LEFT
        )
        self.lbl_ceil_val = tk.Label(
            frame2, text="-0.3 dB", width=8, anchor="e", font=("Arial", 10)
        )
        self.lbl_ceil_val.pack(side=tk.RIGHT)

        self.var_ceiling = tk.DoubleVar(value=-0.3)
        tk.Scale(
            self,
            variable=self.var_ceiling,
            from_=-10.0,
            to=0.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            showvalue=0,
            length=200,
            command=lambda v: self.lbl_ceil_val.config(text=f"{float(v):.1f} dB"),
        ).pack(fill=tk.X, pady=(0, 3))

        frame3 = tk.Frame(self)
        frame3.pack(fill=tk.X)
        tk.Label(frame3, text="释放 (Release):", font=("Arial", 10)).pack(side=tk.LEFT)
        self.lbl_release_val = tk.Label(
            frame3,
            text="100 ms",
            width=8,
            anchor="e",
            font=("Arial", 10),
        )
        self.lbl_release_val.pack(side=tk.RIGHT)

        self.var_release = tk.DoubleVar(value=100.0)
        tk.Scale(
            self,
            variable=self.var_release,
            from_=10,
            to=500,
            resolution=10,
            orient=tk.HORIZONTAL,
            showvalue=0,
            length=200,
            command=lambda v: self.lbl_release_val.config(text=f"{int(float(v))} ms"),
        ).pack(fill=tk.X, pady=(0, 5))

        self.btn_apply = tk.Button(
            self,
            text="▶ 生成预览",
            command=self.on_apply,
            bg="#c0e0c0",
            font=("Arial", 9, "bold"),
        )
        self.btn_apply.pack(fill=tk.X)

    def _update_gain_label(self, v):
        val = float(v)
        if val > 0:
            text = f"+{val:.1f} dB"
            color = "#006400"
        elif val < 0:
            text = f"{val:.1f} dB"
            color = "#8B0000"
        else:
            text = "0.0 dB"
            color = "black"
        self.lbl_gain_val.config(text=text, fg=color)

    def on_apply(self):
        self.apply_callback(
            self.var_gain.get(), self.var_ceiling.get(), self.var_release.get()
        )


class ExportPanel(tk.LabelFrame):
    """导出/保存模块 - 紧凑布局"""

    def __init__(self, parent, get_source_callback):
        super().__init__(parent, text="导出", padx=5, pady=5)
        self.get_source_callback = get_source_callback

        self.default_export_dir = ensure_export_dir()
        self.current_export_dir = self.default_export_dir

        format_frame = tk.Frame(self)
        format_frame.pack(fill=tk.X, pady=(0, 3))

        tk.Label(format_frame, text="格式:", font=("Arial", 10)).pack(side=tk.LEFT)

        self.var_format = tk.StringVar(value="wav")
        tk.Radiobutton(
            format_frame,
            text="WAV",
            variable=self.var_format,
            value="wav",
            command=self._on_format_change,
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=(3, 0))
        tk.Radiobutton(
            format_frame,
            text="MP3",
            variable=self.var_format,
            value="mp3",
            command=self._on_format_change,
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=(3, 0))

        tk.Label(format_frame, text="|", fg="gray").pack(side=tk.LEFT, padx=5)

        self.wav_widgets_frame = tk.Frame(format_frame)
        self.wav_widgets_frame.pack(side=tk.LEFT)
        tk.Label(self.wav_widgets_frame, text="位深:", font=("Arial", 10)).pack(
            side=tk.LEFT
        )
        self.var_wav_bit_depth = tk.StringVar(value="原始")
        self.combo_bit_depth = ttk.Combobox(
            self.wav_widgets_frame,
            textvariable=self.var_wav_bit_depth,
            values=["原始", "16-bit", "24-bit", "32-bit"],
            state="readonly",
            width=6,
            font=("Arial", 10),
        )
        self.combo_bit_depth.pack(side=tk.LEFT, padx=(2, 0))

        self.mp3_widgets_frame = tk.Frame(format_frame)
        tk.Label(self.mp3_widgets_frame, text="比特率:", font=("Arial", 10)).pack(
            side=tk.LEFT
        )
        self.var_mp3_bitrate = tk.StringVar(value="320")
        self.combo_mp3_bitrate = ttk.Combobox(
            self.mp3_widgets_frame,
            textvariable=self.var_mp3_bitrate,
            values=["128", "192", "256", "320"],
            state="readonly",
            width=5,
            font=("Arial", 10),
        )
        self.combo_mp3_bitrate.pack(side=tk.LEFT, padx=(2, 0))
        tk.Label(self.mp3_widgets_frame, text="k", font=("Arial", 10)).pack(
            side=tk.LEFT
        )

        path_frame = tk.Frame(self)
        path_frame.pack(fill=tk.X, pady=(0, 3))

        tk.Label(path_frame, text="位置:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.lbl_export_path = tk.Label(
            path_frame,
            text=self._shorten_path(self.current_export_dir),
            anchor="w",
            fg="#333",
            font=("Arial", 10),
        )
        self.lbl_export_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        tk.Button(
            path_frame,
            text="📁",
            command=self._browse_export_dir,
            font=("Arial", 10),
            width=2,
        ).pack(side=tk.LEFT, padx=(3, 0))
        tk.Button(
            path_frame,
            text="↺",
            command=self._reset_export_dir,
            font=("Arial", 10),
            width=2,
        ).pack(side=tk.LEFT, padx=(2, 0))

        name_frame = tk.Frame(self)
        name_frame.pack(fill=tk.X, pady=(0, 3))

        tk.Label(name_frame, text="文件名:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.var_filename = tk.StringVar(value="processed_audio")
        self.entry_filename = tk.Entry(
            name_frame, textvariable=self.var_filename, font=("Arial", 10), width=22
        )
        self.entry_filename.pack(side=tk.LEFT, padx=(2, 0), fill=tk.X, expand=True)
        self.lbl_extension = tk.Label(
            name_frame, text=".wav", font=("Arial", 8, "bold"), fg="#006400"
        )
        self.lbl_extension.pack(side=tk.LEFT)

        bottom_frame = tk.Frame(self)
        bottom_frame.pack(fill=tk.X)

        self.btn_export = tk.Button(
            bottom_frame,
            text="💾 导出文件",
            command=self._do_export,
            bg="#4a90d9",
            fg="white",
            font=("Arial", 9, "bold"),
            state=tk.DISABLED,
        )
        self.btn_export.pack(side=tk.LEFT)

        self.lbl_status = tk.Label(
            bottom_frame, text="请先生成预览", fg="gray", font=("Arial", 10)
        )
        self.lbl_status.pack(side=tk.LEFT, padx=(8, 0))

    def _shorten_path(self, path, max_len=28):
        if len(path) <= max_len:
            return path
        return "..." + path[-(max_len - 3) :]

    def _on_format_change(self):
        fmt = self.var_format.get()
        self.lbl_extension.config(text=f".{fmt}")

        if fmt == "wav":
            self.mp3_widgets_frame.pack_forget()
            self.wav_widgets_frame.pack(side=tk.LEFT)
            self.lbl_extension.config(fg="#006400")
        else:
            self.wav_widgets_frame.pack_forget()
            self.mp3_widgets_frame.pack(side=tk.LEFT)
            self.lbl_extension.config(fg="#8B0000")

    def _browse_export_dir(self):
        dir_path = filedialog.askdirectory(
            initialdir=self.current_export_dir, title="选择导出目录"
        )
        if dir_path:
            self.current_export_dir = dir_path
            self.lbl_export_path.config(text=self._shorten_path(dir_path))

    def _reset_export_dir(self):
        self.current_export_dir = self.default_export_dir
        self.lbl_export_path.config(text=self._shorten_path(self.default_export_dir))

    def enable_export(self, source_name=""):
        self.btn_export.config(state=tk.NORMAL)
        if source_name:
            base_name = os.path.splitext(source_name)[0]
            self.var_filename.set(f"{base_name}_processed")
        self.lbl_status.config(text="✓ 可导出", fg="green")

    def disable_export(self):
        self.btn_export.config(state=tk.DISABLED)
        self.lbl_status.config(text="请先生成预览", fg="gray")

    def _do_export(self):
        source_file, metadata = self.get_source_callback()

        if not source_file or not os.path.exists(source_file):
            messagebox.showwarning("导出失败", "没有可导出的处理结果，请先生成预览。")
            return

        fmt = self.var_format.get()
        filename = self.var_filename.get().strip()

        if not filename:
            messagebox.showwarning("文件名错误", "请输入有效的文件名。")
            return

        illegal_chars = '<>:"/\\|?*'
        for char in illegal_chars:
            filename = filename.replace(char, "_")

        output_path = os.path.join(self.current_export_dir, f"{filename}.{fmt}")

        if os.path.exists(output_path):
            if not messagebox.askyesno(
                "文件已存在", f"文件 '{filename}.{fmt}' 已存在，是否覆盖？"
            ):
                return

        self.btn_export.config(state=tk.DISABLED, text="导出中...")
        self.lbl_status.config(text="正在导出...", fg="blue")
        self.update()

        threading.Thread(
            target=self._export_worker,
            args=(source_file, output_path, fmt, metadata),
            daemon=True,
        ).start()

    def _export_worker(self, source_file, output_path, fmt, metadata):
        try:
            if fmt == "wav":
                self._export_wav(source_file, output_path, metadata)
            else:
                self._export_mp3(source_file, output_path, metadata)

            self.after(0, lambda: self._on_export_success(output_path))
        except Exception as e:
            self.after(0, lambda: self._on_export_error(str(e)))

    def _export_wav(self, source_file, output_path, metadata):
        bit_depth_choice = self.var_wav_bit_depth.get()

        if bit_depth_choice == "原始":
            original_bit_depth = metadata.get("bit_depth", 16) if metadata else 16
            bit_depth = original_bit_depth
        else:
            bit_depth = int(bit_depth_choice.replace("-bit", ""))

        if bit_depth == 32:
            acodec = "pcm_s32le"
        elif bit_depth == 24:
            acodec = "pcm_s24le"
        else:
            acodec = "pcm_s16le"

        sample_rate = metadata.get("sample_rate", 44100) if metadata else 44100
        channels = metadata.get("channels", 2) if metadata else 2

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source_file,
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-acodec",
            acodec,
            output_path,
        ]

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(
            cmd, capture_output=True, text=True, creationflags=creationflags
        )

        if result.returncode != 0:
            raise RuntimeError(f"WAV 导出失败: {result.stderr}")

    def _export_mp3(self, source_file, output_path, metadata):
        bitrate = self.var_mp3_bitrate.get()
        sample_rate = metadata.get("sample_rate", 44100) if metadata else 44100
        channels = min(metadata.get("channels", 2) if metadata else 2, 2)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source_file,
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-acodec",
            "libmp3lame",
            "-b:a",
            f"{bitrate}k",
            output_path,
        ]

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(
            cmd, capture_output=True, text=True, creationflags=creationflags
        )

        if result.returncode != 0:
            raise RuntimeError(f"MP3 导出失败: {result.stderr}")

    def _on_export_success(self, output_path):
        self.btn_export.config(state=tk.NORMAL, text="💾 导出文件")
        self.lbl_status.config(text="✓ 导出成功!", fg="green")

        if messagebox.askyesno(
            "导出成功", f"文件已保存至:\n{output_path}\n\n是否打开所在文件夹？"
        ):
            self._open_folder(os.path.dirname(output_path))

    def _on_export_error(self, error_msg):
        self.btn_export.config(state=tk.NORMAL, text="💾 导出文件")
        self.lbl_status.config(text="✗ 导出失败", fg="red")
        messagebox.showerror("导出失败", error_msg)

    def _open_folder(self, folder_path):
        try:
            if os.name == "nt":
                os.startfile(folder_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", folder_path])
            else:
                subprocess.run(["xdg-open", folder_path])
        except Exception as e:
            print(f"打开文件夹失败: {e}")


class FourQuadrantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("音频 Limiter 模拟与对比工作台")
        self.root.geometry("950x750")
        self.root.minsize(850, 700)

        MixerManager.init()

        self.processed_hq_file = "temp_processed_hq.wav"
        self.temp_float_file = "temp_processed_float.wav"

        TempFileTracker.register(self.processed_hq_file)
        TempFileTracker.register(self.temp_float_file)

        self.paned_window = tk.PanedWindow(
            root, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.RAISED
        )
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.frame_left = tk.Frame(self.paned_window)
        self.paned_window.add(self.frame_left, width=420)

        self.panel_ref = PlayerPanel(
            self.frame_left, "1. 基准音频 (Reference)", "ref", self.on_any_play
        )
        self.panel_ref.pack(fill=tk.BOTH, expand=True, pady=(0, 3))

        self.panel_source = PlayerPanel(
            self.frame_left,
            "2. 原始音频 (Source for Limiter)",
            "source",
            self.on_any_play,
        )
        self.panel_source.pack(fill=tk.BOTH, expand=True, pady=(3, 0))

        self.frame_right = tk.Frame(self.paned_window)
        self.paned_window.add(self.frame_right, width=480)

        self.panel_controls = LimiterControlPanel(
            self.frame_right, apply_callback=self.process_audio
        )
        self.panel_controls.pack(fill=tk.X, pady=(0, 3))

        self.panel_preview = PlayerPanel(
            self.frame_right,
            "3. 效果预览 (Processed Preview)",
            "preview",
            self.on_any_play,
            show_load_btn=False,
        )
        self.panel_preview.pack(fill=tk.X, pady=(3, 3))

        self.panel_export = ExportPanel(
            self.frame_right, get_source_callback=self.get_processed_file_info
        )
        self.panel_export.pack(fill=tk.X, pady=(3, 0))

        self.init_defaults()

    def get_processed_file_info(self):
        if os.path.exists(self.processed_hq_file):
            return self.processed_hq_file, self.panel_preview.metadata
        return None, None

    def init_defaults(self):
        default_path = os.path.join(os.getcwd(), "music.wav")
        if os.path.exists(default_path):
            self.panel_ref.load_audio_file_async(default_path)
        else:
            self.panel_ref.lbl_file.config(text="[请加载 music.wav]")
        self.panel_source.lbl_file.config(text="[请选择待处理文件]", fg="#888")

    def on_any_play(self, active_id):
        for p in [self.panel_ref, self.panel_source, self.panel_preview]:
            if p.panel_id != active_id:
                p.force_stop_logic()

    def process_audio(self, gain_db, ceiling_db, release_ms):
        source_path = self.panel_source.current_file_path
        source_metadata = self.panel_source.metadata

        if not source_path or not os.path.exists(source_path):
            messagebox.showwarning(
                "提示", "请先在左下角 (Source) 加载一个有效的音频文件。"
            )
            return

        self.panel_controls.btn_apply.config(state=tk.DISABLED, text="处理中...")
        self.panel_export.disable_export()
        self.root.config(cursor="watch")
        self.root.update()

        try:
            self.panel_preview.release_file_handle()
            self.panel_preview.clear_metadata_display()

            self._cleanup_processing_files()

            output_metadata = self._process_with_pedalboard(
                source_path, gain_db, ceiling_db, release_ms, source_metadata
            )

            if not os.path.exists(self.processed_hq_file):
                raise FileNotFoundError("处理后的文件未生成")

            actual_metadata = get_audio_metadata_ffprobe(self.processed_hq_file)
            duration = (
                actual_metadata.get("duration")
                or output_metadata.get("duration")
                or get_audio_duration_ffprobe(self.processed_hq_file)
                or 0
            )

            if duration <= 0:
                raise ValueError("无法获取有效的音频时长")

            actual_metadata["duration"] = duration

            gain_str = f"+{gain_db:.1f}" if gain_db >= 0 else f"{gain_db:.1f}"
            info_text = (
                f"Limiter (G:{gain_str}, C:{ceiling_db:.1f}dB) "
                f"[{actual_metadata.get('sample_rate', 44100)}Hz/{actual_metadata.get('bit_depth', 16)}bit]"
            )

            self.panel_preview.load_from_processed_file(
                self.processed_hq_file, info_text, duration, actual_metadata
            )

            source_name = os.path.basename(source_path)
            self.panel_export.enable_export(source_name)

            print(f"[处理完成] 输出: {self.processed_hq_file}")

        except Exception as e:
            print(f"处理失败: {e}")
            traceback.print_exc()
            messagebox.showerror("处理失败", str(e))
            self.panel_export.disable_export()
        finally:
            self.panel_controls.btn_apply.config(state=tk.NORMAL, text="▶ 生成预览")
            self.root.config(cursor="")

    def _cleanup_processing_files(self):
        for f in [self.processed_hq_file, self.temp_float_file]:
            safe_remove(f)

    def _process_with_pedalboard(
        self, source_path, gain_db, ceiling_db, release_ms, source_metadata=None
    ):
        """使用 Pedalboard 处理音频"""
        print(f"[Pedalboard] 开始处理: {source_path}")

        with AudioFile(source_path) as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate
            num_channels = f.num_channels

        original_bit_depth = (source_metadata or {}).get("bit_depth", 16)

        board = Pedalboard(
            [
                Gain(gain_db=gain_db),
                Limiter(threshold_db=ceiling_db, release_ms=release_ms),
            ]
        )
        processed = board(audio, samplerate)

        with AudioFile(
            self.temp_float_file, "w", samplerate=samplerate, num_channels=num_channels
        ) as f:
            f.write(processed)

        TempFileTracker.register(self.temp_float_file)

        acodec = "pcm_s24le" if original_bit_depth >= 24 else "pcm_s16le"
        output_bit_depth = 24 if original_bit_depth >= 24 else 16

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            self.temp_float_file,
            "-ar",
            str(samplerate),
            "-ac",
            str(num_channels),
            "-acodec",
            acodec,
            self.processed_hq_file,
        ]

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(
            cmd, capture_output=True, text=True, creationflags=creationflags
        )

        safe_remove(self.temp_float_file)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 转换失败: {result.stderr}")

        TempFileTracker.register(self.processed_hq_file)

        return {
            "sample_rate": samplerate,
            "channels": num_channels,
            "bit_depth": output_bit_depth,
            "codec": acodec,
            "duration": (source_metadata or {}).get("duration"),
        }

    def cleanup(self):
        MixerManager.quit()
        for p in [self.panel_ref, self.panel_source, self.panel_preview]:
            p.cleanup_temp_files()
        self._cleanup_processing_files()


if __name__ == "__main__":
    root = tk.Tk()
    app = FourQuadrantApp(root)

    def on_closing():
        for p in [app.panel_ref, app.panel_source, app.panel_preview]:
            p.stop_playback()
        try:
            pygame.mixer.music.stop()
        except:
            pass
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
