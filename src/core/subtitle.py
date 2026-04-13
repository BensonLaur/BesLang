#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字幕识别模块

功能：
- 从视频中提取音频
- 使用Whisper识别语音生成字幕
- 保存为SRT格式字幕
- 支持进度回调
- GPU/CPU自动切换

作者: Benson Laur
日期: 2025-11-30
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import whisper
    import torch
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请先安装依赖: pip install openai-whisper torch")
    sys.exit(1)

from src.utils.logger import get_logger


class SubtitleGenerator:
    """
    字幕生成器

    负责从视频中提取音频、识别语音、生成字幕
    """

    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None
    ):
        """
        初始化字幕生成器

        Args:
            model_name: Whisper模型名称 (tiny/base/small/medium/large)
            device: 设备 (cuda/cpu)，None则自动选择
            language: 语言代码 (zh/en/ja等)，None则自动检测
        """
        self.logger = get_logger()
        self.model_name = model_name
        self.language = language

        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # GPU可用性检查
        if self.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("GPU不可用，自动切换到CPU模式")
            self.device = "cpu"

        self.logger.info(f"字幕生成器初始化: 模型={model_name}, 设备={self.device}, 语言={language or '自动检测'}")

        # 模型实例（延迟加载）
        self._model = None

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self.logger.info(f"正在加载 Whisper-{self.model_name} 模型...")
            try:
                self._model = whisper.load_model(self.model_name, device=self.device)
                self.logger.info(f"模型加载成功: {self.model_name} on {self.device}")
            except Exception as e:
                self.logger.error(f"模型加载失败: {e}")
                raise
        return self._model

    def extract_audio(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> str:
        """
        从视频中提取音频

        Args:
            video_path: 视频文件路径
            output_path: 输出音频路径，None则自动生成临时文件
            progress_callback: 进度回调函数 callback(stage: str, progress: float)

        Returns:
            str: 音频文件路径

        Raises:
            FileNotFoundError: 视频文件不存在
            RuntimeError: FFmpeg处理失败
        """
        self.logger.info(f"开始提取音频: {video_path}")

        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            error_msg = f"视频文件不存在: {video_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 进度回调
        if progress_callback:
            progress_callback("extract_audio", 0.0)

        # 生成输出路径
        if output_path is None:
            # 创建临时音频文件
            temp_dir = Path(tempfile.gettempdir()) / "beslang"
            temp_dir.mkdir(exist_ok=True)

            video_name = Path(video_path).stem
            output_path = str(temp_dir / f"{video_name}.wav")

        # 构建FFmpeg命令
        # -i: 输入文件
        # -vn: 不处理视频
        # -acodec pcm_s16le: 使用PCM 16位编码
        # -ar 16000: 采样率16kHz（Whisper推荐）
        # -ac 1: 单声道
        # -y: 覆盖输出文件
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            output_path
        ]

        try:
            self.logger.debug(f"执行FFmpeg命令: {' '.join(ffmpeg_cmd)}")

            # 执行FFmpeg命令
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                encoding='utf-8',
                errors='ignore'
            )

            # 检查输出文件是否存在
            if not os.path.exists(output_path):
                raise RuntimeError("音频文件未生成")

            # 进度回调
            if progress_callback:
                progress_callback("extract_audio", 1.0)

            self.logger.info(f"音频提取成功: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg处理失败: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"音频提取失败: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def recognize(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[Dict]:
        """
        识别音频生成字幕

        Args:
            audio_path: 音频文件路径
            progress_callback: 进度回调函数 callback(stage: str, progress: float)

        Returns:
            List[Dict]: 字幕片段列表，每个片段包含:
                - start: 开始时间（秒）
                - end: 结束时间（秒）
                - text: 文本内容

        Raises:
            FileNotFoundError: 音频文件不存在
            RuntimeError: 识别失败
        """
        self.logger.info(f"开始识别音频: {audio_path}")

        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            error_msg = f"音频文件不存在: {audio_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 进度回调
        if progress_callback:
            progress_callback("recognize", 0.0)

        try:
            # 加载模型（延迟加载）
            model = self.model

            # 进度回调
            if progress_callback:
                progress_callback("recognize", 0.2)

            # 识别音频
            self.logger.debug(f"开始Whisper识别: 语言={self.language or '自动检测'}")

            result = model.transcribe(
                audio_path,
                language=self.language,
                verbose=False  # 不输出详细信息
            )

            # 进度回调
            if progress_callback:
                progress_callback("recognize", 0.9)

            # 提取字幕片段
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                })

            # 进度回调
            if progress_callback:
                progress_callback("recognize", 1.0)

            self.logger.info(f"识别完成: {len(segments)} 个片段, "
                           f"文本长度={len(result['text'])} 字符")

            return segments

        except Exception as e:
            error_msg = f"音频识别失败: {e}"
            self.logger.error(error_msg)
            self.logger.exception(error_msg)  # 记录详细堆栈
            raise RuntimeError(error_msg)

    def save_srt(
        self,
        segments: List[Dict],
        output_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> bool:
        """
        保存字幕为SRT格式

        Args:
            segments: 字幕片段列表
            output_path: 输出SRT文件路径
            progress_callback: 进度回调函数 callback(stage: str, progress: float)

        Returns:
            bool: 保存是否成功

        Raises:
            RuntimeError: 保存失败
        """
        self.logger.info(f"保存字幕到: {output_path}")

        # 进度回调
        if progress_callback:
            progress_callback("save_srt", 0.0)

        try:
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # 生成SRT内容
            srt_content = self._generate_srt_content(segments)

            # 进度回调
            if progress_callback:
                progress_callback("save_srt", 0.5)

            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)

            # 进度回调
            if progress_callback:
                progress_callback("save_srt", 1.0)

            self.logger.info(f"字幕保存成功: {output_path}")
            return True

        except Exception as e:
            error_msg = f"字幕保存失败: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _generate_srt_content(self, segments: List[Dict]) -> str:
        """
        生成SRT格式内容

        Args:
            segments: 字幕片段列表

        Returns:
            str: SRT格式字符串
        """
        srt_lines = []

        for i, segment in enumerate(segments, start=1):
            # 序号
            srt_lines.append(str(i))

            # 时间轴
            start_time = self._format_timestamp(segment['start'])
            end_time = self._format_timestamp(segment['end'])
            srt_lines.append(f"{start_time} --> {end_time}")

            # 文本内容
            srt_lines.append(segment['text'])

            # 空行（片段间分隔）
            srt_lines.append('')

        return '\n'.join(srt_lines)

    def _format_timestamp(self, seconds: float) -> str:
        """
        格式化时间戳为SRT格式

        Args:
            seconds: 秒数

        Returns:
            str: SRT时间格式 (HH:MM:SS,mmm)

        Example:
            >>> _format_timestamp(65.5)
            '00:01:05,500'
        """
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        millis = td.microseconds // 1000

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def generate_subtitle(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        keep_audio: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> str:
        """
        一键生成字幕（完整流程）

        Args:
            video_path: 视频文件路径
            output_path: 输出SRT文件路径，None则自动生成
            keep_audio: 是否保留提取的音频文件
            progress_callback: 进度回调函数 callback(stage: str, progress: float)

        Returns:
            str: 字幕文件路径

        Raises:
            FileNotFoundError: 视频文件不存在
            RuntimeError: 处理失败
        """
        self.logger.info(f"开始生成字幕: {video_path}")

        # 生成输出路径
        if output_path is None:
            output_path = str(Path(video_path).with_suffix('.srt'))

        audio_path = None

        try:
            # 步骤1: 提取音频
            self.logger.info("步骤 1/3: 提取音频")
            audio_path = self.extract_audio(video_path, progress_callback=progress_callback)

            # 步骤2: 识别语音
            self.logger.info("步骤 2/3: 识别语音")
            segments = self.recognize(audio_path, progress_callback=progress_callback)

            # 步骤3: 保存字幕
            self.logger.info("步骤 3/3: 保存字幕")
            self.save_srt(segments, output_path, progress_callback=progress_callback)

            self.logger.info(f"字幕生成完成: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"字幕生成失败: {e}")
            raise

        finally:
            # 清理临时音频文件
            if audio_path and not keep_audio:
                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                        self.logger.debug(f"临时音频文件已删除: {audio_path}")
                except Exception as e:
                    self.logger.warning(f"删除临时音频文件失败: {e}")

    def __del__(self):
        """释放资源"""
        if self._model is not None:
            # 释放GPU显存
            del self._model
            if self.device == "cuda":
                torch.cuda.empty_cache()
                self.logger.debug("GPU显存已释放")


# 测试代码
if __name__ == "__main__":
    """测试字幕生成功能"""
    import sys

    # 设置输出编码为UTF-8
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # 测试文件路径
    test_video = project_root / "tests" / "fixtures" / "砥柱_01_26s.mp4"

    if not test_video.exists():
        print(f"❌ 测试视频不存在: {test_video}")
        print("请准备测试视频后再运行")
        sys.exit(1)

    # 进度回调函数
    def progress_callback(stage: str, progress: float):
        print(f"  {stage}: {progress*100:.1f}%")

    print("\n" + "="*60)
    print("🎯 字幕生成测试")
    print("="*60 + "\n")

    # 创建字幕生成器
    generator = SubtitleGenerator(
        model_name="base",
        language="zh"  # 中文
    )

    # 生成字幕
    try:
        output_srt = generator.generate_subtitle(
            str(test_video),
            progress_callback=progress_callback
        )

        print(f"\n✅ 字幕生成成功!")
        print(f"   输出文件: {output_srt}")

        # 读取并显示字幕内容
        with open(output_srt, 'r', encoding='utf-8') as f:
            content = f.read()
            print("\n字幕内容预览:")
            print("-" * 60)
            print(content[:500])
            if len(content) > 500:
                print("...")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
