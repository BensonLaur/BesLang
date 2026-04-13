#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
翻译模块

⚠️ 重要提示 (2025-12-15)：
    此模块使用 MarianMT 进行翻译，但因翻译质量差（无上下文理解能力）
    已决定重构为使用 **Qwen2.5 本地LLM**。

    当前状态：Week 4 优先验证 Qwen2.5 翻译方案

    详见：
    - docs/Bes语坊_技术方案决策记录_v1.0.md 第4章
    - docs/Bes语坊_Windows开发计划_v5.0.md (v5.1更新)

    新方案分级配置：
    - 低配(2GB+): Qwen2.5-0.5B (Q4) ~300MB
    - 标准(4GB): Qwen2.5-3B (Q4) ~2GB
    - 高配(6GB+): Qwen2.5-7B (Q4) ~4.5GB

功能（当前版本 - MarianMT，待重构）：
- 使用MarianMT进行中英互译
- 支持批量翻译优化
- 支持双语字幕生成
- GPU/CPU自动切换
- 语言自动检测

作者: Benson Laur
日期: 2025-12-11
更新: 2025-12-15 (标记为待重构)
"""

import os
import sys
import gc
from pathlib import Path
from typing import List, Dict, Optional, Callable, Literal, Union
from datetime import timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请先安装依赖: pip install transformers sentencepiece")
    sys.exit(1)

from src.utils.logger import get_logger


# 支持的翻译方向和对应的模型
TRANSLATION_MODELS = {
    'en-zh': 'Helsinki-NLP/opus-mt-en-zh',  # 英文→中文
    'zh-en': 'Helsinki-NLP/opus-mt-zh-en',  # 中文→英文
    'en-ja': 'Helsinki-NLP/opus-mt-en-jap', # 英文→日文
    'ja-en': 'Helsinki-NLP/opus-mt-jap-en', # 日文→英文
}

# 语言代码映射
LANGUAGE_NAMES = {
    'en': '英文',
    'zh': '中文',
    'ja': '日文',
    'auto': '自动检测'
}


class SubtitleTranslator:
    """
    字幕翻译器

    负责翻译字幕文本，支持批量处理和双语字幕生成
    """

    def __init__(
        self,
        src_lang: str = 'en',
        tgt_lang: str = 'zh',
        device: Optional[str] = None,
        batch_size: int = 8
    ):
        """
        初始化字幕翻译器

        Args:
            src_lang: 源语言代码 (en/zh/ja)
            tgt_lang: 目标语言代码 (en/zh/ja)
            device: 设备 (cuda/cpu)，None则自动选择
            batch_size: 批量翻译大小
        """
        self.logger = get_logger()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size

        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # GPU可用性检查
        if self.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("GPU不可用，自动切换到CPU模式")
            self.device = "cpu"

        # 确定翻译方向
        self.direction = f"{src_lang}-{tgt_lang}"
        if self.direction not in TRANSLATION_MODELS:
            raise ValueError(f"不支持的翻译方向: {self.direction}")

        self.model_name = TRANSLATION_MODELS[self.direction]

        self.logger.info(
            f"翻译器初始化: {LANGUAGE_NAMES.get(src_lang, src_lang)}→"
            f"{LANGUAGE_NAMES.get(tgt_lang, tgt_lang)}, "
            f"设备={self.device}, 批量大小={batch_size}"
        )

        # 模型和分词器（延迟加载）
        self._model = None
        self._tokenizer = None

    @property
    def model(self) -> MarianMTModel:
        """延迟加载模型"""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self) -> MarianTokenizer:
        """延迟加载分词器"""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """加载模型和分词器"""
        self.logger.info(f"正在加载翻译模型: {self.model_name}")

        try:
            import sys
            sys.stdout.flush()
            sys.stderr.flush()

            self.logger.info("步骤1: 加载tokenizer...")
            self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.logger.info("步骤1完成: tokenizer加载成功")

            self.logger.info("步骤2: 加载模型...")
            self._model = MarianMTModel.from_pretrained(self.model_name)
            self.logger.info("步骤2完成: 模型加载成功")

            # 尝试加载到GPU，失败则回退到CPU
            if self.device == "cuda":
                try:
                    self.logger.info("步骤3: 移动模型到GPU...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    self._model = self._model.to(self.device)
                    self.logger.info(f"步骤3完成: 模型已移动到 {self.device}")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "CUDA" in str(e):
                        self.logger.warning(f"GPU显存不足，回退到CPU模式: {e}")
                        self.device = "cpu"
                        gc.collect()
                        torch.cuda.empty_cache()
                        self.logger.info(f"模型加载成功: {self.model_name} on {self.device} (回退)")
                    else:
                        raise
            else:
                self.logger.info(f"模型将在 {self.device} 上运行")

            # 设置为评估模式
            self._model.eval()
            self.logger.info(f"翻译模型准备就绪: {self.model_name} on {self.device}")

        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def translate_text(
        self,
        text: str,
        max_length: int = 512
    ) -> str:
        """
        翻译单条文本

        Args:
            text: 待翻译文本
            max_length: 最大输出长度

        Returns:
            str: 翻译结果
        """
        if not text or not text.strip():
            return ""

        try:
            # 编码
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 翻译
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )

            # 解码
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return translated.strip()

        except Exception as e:
            self.logger.error(f"翻译失败: {e}")
            return text  # 返回原文

    def translate_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[str]:
        """
        批量翻译文本

        Args:
            texts: 待翻译文本列表
            max_length: 最大输出长度
            progress_callback: 进度回调函数 callback(progress: float)

        Returns:
            List[str]: 翻译结果列表
        """
        if not texts:
            return []

        self.logger.info(f"开始批量翻译: {len(texts)} 条文本")

        results = []
        total = len(texts)

        # 分批处理
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]

            # 过滤空文本
            non_empty_indices = [j for j, t in enumerate(batch) if t and t.strip()]
            non_empty_texts = [batch[j] for j in non_empty_indices]

            if non_empty_texts:
                try:
                    # 编码
                    inputs = self.tokenizer(
                        non_empty_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # 翻译
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=max_length,
                            num_beams=4,
                            early_stopping=True
                        )

                    # 解码
                    translations = [
                        self.tokenizer.decode(output, skip_special_tokens=True).strip()
                        for output in outputs
                    ]

                    # 重建结果（包含空文本）
                    batch_results = [""] * len(batch)
                    for j, idx in enumerate(non_empty_indices):
                        batch_results[idx] = translations[j]

                    results.extend(batch_results)

                except Exception as e:
                    self.logger.error(f"批量翻译失败: {e}")
                    # 回退到单条翻译
                    for text in batch:
                        results.append(self.translate_text(text, max_length))
            else:
                results.extend([""] * len(batch))

            # 进度回调
            if progress_callback:
                progress = min(i + self.batch_size, total) / total
                progress_callback(progress)

        self.logger.info(f"批量翻译完成: {len(results)} 条")
        return results

    def translate_subtitles(
        self,
        segments: List[Dict],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[Dict]:
        """
        翻译字幕片段

        Args:
            segments: 字幕片段列表，每个片段包含:
                - start: 开始时间（秒）
                - end: 结束时间（秒）
                - text: 文本内容
            progress_callback: 进度回调函数 callback(stage: str, progress: float)

        Returns:
            List[Dict]: 翻译后的字幕片段列表，增加:
                - translated_text: 翻译后的文本
        """
        if not segments:
            return []

        self.logger.info(f"开始翻译字幕: {len(segments)} 个片段")

        # 进度回调
        if progress_callback:
            progress_callback("translate", 0.0)

        # 提取文本
        texts = [seg.get('text', '') for seg in segments]

        # 定义内部进度回调
        def batch_progress(p):
            if progress_callback:
                progress_callback("translate", p * 0.9)  # 留10%给后处理

        # 批量翻译
        translations = self.translate_batch(texts, progress_callback=batch_progress)

        # 组合结果
        result = []
        for seg, translated in zip(segments, translations):
            new_seg = seg.copy()
            new_seg['translated_text'] = translated
            result.append(new_seg)

        # 进度回调
        if progress_callback:
            progress_callback("translate", 1.0)

        self.logger.info(f"字幕翻译完成: {len(result)} 个片段")
        return result

    def generate_bilingual_srt(
        self,
        segments: List[Dict],
        layout: Literal['stacked', 'original_first', 'translation_first'] = 'stacked',
        output_path: Optional[str] = None
    ) -> str:
        """
        生成双语字幕SRT文件

        Args:
            segments: 翻译后的字幕片段列表（需包含translated_text）
            layout: 布局方式
                - 'stacked': 上下排列（原文在上，译文在下）
                - 'original_first': 仅显示原文在上
                - 'translation_first': 译文在上，原文在下
            output_path: 输出文件路径，None则返回内容字符串

        Returns:
            str: SRT内容或文件路径
        """
        self.logger.info(f"生成双语字幕: 布局={layout}")

        srt_lines = []

        for i, seg in enumerate(segments, start=1):
            # 序号
            srt_lines.append(str(i))

            # 时间轴
            start_time = self._format_timestamp(seg['start'])
            end_time = self._format_timestamp(seg['end'])
            srt_lines.append(f"{start_time} --> {end_time}")

            # 文本内容
            original = seg.get('text', '').strip()
            translated = seg.get('translated_text', '').strip()

            if layout == 'stacked':
                # 原文在上，译文在下
                if original:
                    srt_lines.append(original)
                if translated:
                    srt_lines.append(translated)
            elif layout == 'original_first':
                # 仅原文
                if original:
                    srt_lines.append(original)
            elif layout == 'translation_first':
                # 译文在上，原文在下
                if translated:
                    srt_lines.append(translated)
                if original:
                    srt_lines.append(original)

            # 空行分隔
            srt_lines.append('')

        content = '\n'.join(srt_lines)

        # 保存到文件
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.logger.info(f"双语字幕已保存: {output_path}")
            return output_path

        return content

    def generate_translation_only_srt(
        self,
        segments: List[Dict],
        output_path: Optional[str] = None
    ) -> str:
        """
        生成仅译文字幕SRT文件

        Args:
            segments: 翻译后的字幕片段列表（需包含translated_text）
            output_path: 输出文件路径，None则返回内容字符串

        Returns:
            str: SRT内容或文件路径
        """
        self.logger.info("生成仅译文字幕")

        srt_lines = []

        for i, seg in enumerate(segments, start=1):
            # 序号
            srt_lines.append(str(i))

            # 时间轴
            start_time = self._format_timestamp(seg['start'])
            end_time = self._format_timestamp(seg['end'])
            srt_lines.append(f"{start_time} --> {end_time}")

            # 仅译文
            translated = seg.get('translated_text', '').strip()
            if translated:
                srt_lines.append(translated)
            else:
                # 如果没有译文，使用原文
                srt_lines.append(seg.get('text', '').strip())

            # 空行分隔
            srt_lines.append('')

        content = '\n'.join(srt_lines)

        # 保存到文件
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.logger.info(f"译文字幕已保存: {output_path}")
            return output_path

        return content

    def _format_timestamp(self, seconds: float) -> str:
        """
        格式化时间戳为SRT格式

        Args:
            seconds: 秒数

        Returns:
            str: SRT时间格式 (HH:MM:SS,mmm)
        """
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        millis = td.microseconds // 1000

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def detect_language(text: str) -> str:
        """
        简单的语言检测

        Args:
            text: 待检测文本

        Returns:
            str: 语言代码 (en/zh/ja/unknown)
        """
        if not text:
            return 'unknown'

        # 统计字符类型
        chinese_count = 0
        japanese_count = 0
        english_count = 0

        for char in text:
            code = ord(char)
            # 中文字符范围
            if 0x4E00 <= code <= 0x9FFF:
                chinese_count += 1
            # 日文假名范围
            elif 0x3040 <= code <= 0x30FF:
                japanese_count += 1
            # 英文字母
            elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
                english_count += 1

        total = chinese_count + japanese_count + english_count
        if total == 0:
            return 'unknown'

        # 根据占比判断
        if japanese_count > 0:
            return 'ja'
        elif chinese_count / total > 0.3:
            return 'zh'
        elif english_count / total > 0.5:
            return 'en'
        else:
            return 'zh'  # 默认中文

    def release(self):
        """释放模型资源"""
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None

        if hasattr(self, '_tokenizer') and self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if hasattr(self, 'device') and self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        if hasattr(self, 'logger'):
            self.logger.debug("翻译模型资源已释放")

    def __del__(self):
        """析构函数"""
        try:
            self.release()
        except Exception:
            pass  # 忽略析构时的异常


# 便捷函数
def create_translator(
    src_lang: str = 'en',
    tgt_lang: str = 'zh',
    device: Optional[str] = None
) -> SubtitleTranslator:
    """
    创建翻译器实例

    Args:
        src_lang: 源语言 (en/zh/ja)
        tgt_lang: 目标语言 (en/zh/ja)
        device: 设备 (cuda/cpu/None自动)

    Returns:
        SubtitleTranslator: 翻译器实例
    """
    return SubtitleTranslator(src_lang=src_lang, tgt_lang=tgt_lang, device=device)


def get_supported_languages() -> Dict[str, str]:
    """
    获取支持的语言列表

    Returns:
        Dict[str, str]: 语言代码到名称的映射
    """
    return LANGUAGE_NAMES.copy()


def get_supported_directions() -> List[str]:
    """
    获取支持的翻译方向

    Returns:
        List[str]: 支持的翻译方向列表 (如 ['en-zh', 'zh-en'])
    """
    return list(TRANSLATION_MODELS.keys())


# 测试代码
if __name__ == "__main__":
    """测试翻译功能"""
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "=" * 60)
    print("  字幕翻译模块测试")
    print("=" * 60)

    # 测试字幕数据（模拟识别结果）
    test_segments = [
        {'start': 0.0, 'end': 3.0, 'text': 'Hello everyone, welcome to my channel.'},
        {'start': 3.0, 'end': 6.0, 'text': 'Today we are going to learn Python programming.'},
        {'start': 6.0, 'end': 9.0, 'text': 'Artificial intelligence is changing our world.'},
        {'start': 9.0, 'end': 12.0, 'text': 'Thank you for watching!'},
    ]

    # 创建翻译器
    print("\n--- 创建翻译器 ---")
    translator = create_translator(src_lang='en', tgt_lang='zh')

    # 进度回调
    def progress_callback(stage: str, progress: float):
        print(f"  {stage}: {progress*100:.1f}%")

    # 翻译字幕
    print("\n--- 翻译字幕 ---")
    translated_segments = translator.translate_subtitles(
        test_segments,
        progress_callback=progress_callback
    )

    # 显示结果
    print("\n--- 翻译结果 ---")
    for seg in translated_segments:
        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s]")
        print(f"    原文: {seg['text']}")
        print(f"    译文: {seg['translated_text']}")

    # 生成双语字幕
    print("\n--- 生成双语字幕 ---")
    bilingual_content = translator.generate_bilingual_srt(
        translated_segments,
        layout='stacked'
    )
    print(bilingual_content[:500])

    # 语言检测测试
    print("\n--- 语言检测测试 ---")
    test_texts = [
        "Hello, how are you?",
        "你好，最近怎么样？",
        "こんにちは",
        "Hello 你好",
    ]
    for text in test_texts:
        lang = SubtitleTranslator.detect_language(text)
        print(f"  '{text}' -> {lang}")

    # 释放资源
    translator.release()

    print("\n" + "=" * 60)
    print("  ✅ 翻译模块测试完成!")
    print("=" * 60)
