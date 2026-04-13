#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM翻译模块 - 使用Qwen2.5本地大模型进行翻译

基于技术决策（2025-12-15）：
- 替代MarianMT，解决无上下文理解导致的翻译质量问题
- 使用transformers + bitsandbytes 4bit量化加载模型
- 支持硬件分级配置

分级配置：
- 低配(2GB+): Qwen2.5-0.5B-Instruct (4bit) ~0.4GB显存
- 标准(4GB): Qwen2.5-3B-Instruct (4bit) ~2GB显存
- 高配(6GB+): Qwen2.5-7B-Instruct (4bit) ~4GB显存

作者: Benson Laur
日期: 2025-12-21
"""

import os
import sys
import gc
from pathlib import Path
from typing import List, Dict, Optional, Callable, Literal
from datetime import timedelta

# 设置镜像（放在导入transformers之前）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请先安装依赖: pip install transformers bitsandbytes accelerate")
    sys.exit(1)

from src.utils.logger import get_logger


# 翻译模型配置
TRANSLATION_MODELS = {
    'low': {
        'name': 'Qwen2.5-0.5B-Instruct',
        'hf_name': 'Qwen/Qwen2.5-0.5B-Instruct',
        'vram': '~0.5GB',
        'quality': '中等',
    },
    'standard': {
        'name': 'Qwen2.5-3B-Instruct',
        'hf_name': 'Qwen/Qwen2.5-3B-Instruct',
        'vram': '~2GB',
        'quality': '良好',
    },
    'high': {
        'name': 'Qwen2.5-7B-Instruct',
        'hf_name': 'Qwen/Qwen2.5-7B-Instruct',
        'vram': '~4GB',
        'quality': '优秀',
    },
}

# 语言名称映射
LANGUAGE_NAMES = {
    'en': '英文',
    'zh': '中文',
    'ja': '日文',
    'ko': '韩文',
    'fr': '法文',
    'de': '德文',
    'es': '西班牙文',
    'ru': '俄文',
}


class LLMTranslator:
    """
    基于LLM的字幕翻译器

    使用Qwen2.5本地大模型进行高质量翻译，支持上下文理解
    """

    def __init__(
        self,
        model_tier: Literal['low', 'standard', 'high'] = 'standard',
        src_lang: str = 'zh',
        tgt_lang: str = 'en',
        use_gpu: bool = True,
        load_in_4bit: bool = True,
    ):
        """
        初始化LLM翻译器

        Args:
            model_tier: 模型等级 (low/standard/high)
            src_lang: 源语言代码
            tgt_lang: 目标语言代码
            use_gpu: 是否使用GPU
            load_in_4bit: 是否使用4bit量化（推荐开启以节省显存）
        """
        self.logger = get_logger()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model_tier = model_tier
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.load_in_4bit = load_in_4bit

        # 确定模型配置
        if model_tier not in TRANSLATION_MODELS:
            raise ValueError(f"不支持的模型等级: {model_tier}")

        self.model_config = TRANSLATION_MODELS[model_tier]
        self.model_name = self.model_config['hf_name']

        self.logger.info(
            f"LLM翻译器初始化: {self.model_config['name']}, "
            f"{LANGUAGE_NAMES.get(src_lang, src_lang)}→{LANGUAGE_NAMES.get(tgt_lang, tgt_lang)}, "
            f"GPU={self.use_gpu}, 4bit={self.load_in_4bit}"
        )

        # 模型和分词器（延迟加载）
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """延迟加载分词器"""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """加载LLM模型"""
        self.logger.info(f"正在加载LLM模型: {self.model_name}")

        try:
            # 加载分词器
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # 配置量化
            if self.load_in_4bit and self.use_gpu:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map='auto',
                    trust_remote_code=True
                )
            else:
                # 无量化或CPU模式
                device = 'cuda' if self.use_gpu else 'cpu'
                dtype = torch.float16 if self.use_gpu else torch.float32
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map='auto' if self.use_gpu else None,
                    trust_remote_code=True
                )
                if not self.use_gpu:
                    self._model = self._model.to(device)

            if self.use_gpu:
                vram = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"LLM模型加载成功，显存使用: {vram:.2f} GB")
            else:
                self.logger.info("LLM模型加载成功 (CPU模式)")

        except Exception as e:
            self.logger.error(f"LLM模型加载失败: {e}")
            raise

    def _build_prompt(self, text: str) -> str:
        """构建翻译提示词"""
        src_name = LANGUAGE_NAMES.get(self.src_lang, self.src_lang)
        tgt_name = LANGUAGE_NAMES.get(self.tgt_lang, self.tgt_lang)

        return f"将下面的{src_name}翻译成{tgt_name}，只输出翻译结果：\n{text}"

    def _build_batch_prompt(self, texts: List[str]) -> str:
        """构建批量翻译提示词"""
        src_name = LANGUAGE_NAMES.get(self.src_lang, self.src_lang)
        tgt_name = LANGUAGE_NAMES.get(self.tgt_lang, self.tgt_lang)

        text_lines = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])

        return f"""将下面的{src_name}字幕翻译成{tgt_name}。
要求：
1. 保持原文的语气和风格
2. 每行翻译对应原文的编号
3. 只输出翻译结果，格式为"编号. 翻译"

原文：
{text_lines}

翻译："""

    def translate_text(self, text: str) -> str:
        """
        翻译单条文本

        Args:
            text: 待翻译文本

        Returns:
            str: 翻译结果
        """
        if not text or not text.strip():
            return ""

        try:
            prompt = self._build_prompt(text)
            messages = [{'role': 'user', 'content': prompt}]

            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([chat_text], return_tensors='pt')

            if self.use_gpu:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            self.logger.error(f"翻译失败: {e}")
            return text  # 返回原文

    def translate_batch(
        self,
        texts: List[str],
        batch_size: int = 10,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[str]:
        """
        批量翻译文本

        Args:
            texts: 待翻译文本列表
            batch_size: 每批处理的文本数量
            progress_callback: 进度回调函数

        Returns:
            List[str]: 翻译结果列表
        """
        if not texts:
            return []

        self.logger.info(f"开始批量翻译: {len(texts)} 条文本")

        results = []
        total = len(texts)

        # 分批处理
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]

            # 过滤空文本
            non_empty = [(j, t) for j, t in enumerate(batch) if t and t.strip()]

            if non_empty:
                try:
                    # 构建批量翻译提示
                    non_empty_texts = [t for _, t in non_empty]
                    prompt = self._build_batch_prompt(non_empty_texts)
                    messages = [{'role': 'user', 'content': prompt}]

                    chat_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.tokenizer([chat_text], return_tensors='pt')

                    if self.use_gpu:
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            temperature=0.1,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )

                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )

                    # 解析结果
                    translations = self._parse_batch_result(response, len(non_empty_texts))

                    # 重建完整结果
                    batch_results = [""] * len(batch)
                    for (orig_idx, _), trans in zip(non_empty, translations):
                        batch_results[orig_idx] = trans

                    results.extend(batch_results)

                except Exception as e:
                    self.logger.error(f"批量翻译失败: {e}")
                    # 回退到单条翻译
                    for text in batch:
                        results.append(self.translate_text(text))
            else:
                results.extend([""] * len(batch))

            # 进度回调
            if progress_callback:
                progress = min(i + batch_size, total) / total
                progress_callback(progress)

        self.logger.info(f"批量翻译完成: {len(results)} 条")
        return results

    def _parse_batch_result(self, result_text: str, expected_count: int) -> List[str]:
        """解析批量翻译结果"""
        lines = result_text.strip().split('\n')
        translations = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 移除编号前缀
            if len(line) > 2 and line[0].isdigit() and line[1] == '.':
                line = line[2:].strip()
            elif len(line) > 3 and line[:2].isdigit() and line[2] == '.':
                line = line[3:].strip()

            if line:
                translations.append(line)

        # 补齐或截断
        if len(translations) < expected_count:
            translations.extend([""] * (expected_count - len(translations)))
        elif len(translations) > expected_count:
            translations = translations[:expected_count]

        return translations

    def translate_subtitles(
        self,
        segments: List[Dict],
        batch_size: int = 10,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[Dict]:
        """
        翻译字幕片段

        Args:
            segments: 字幕片段列表
            batch_size: 批量大小
            progress_callback: 进度回调 callback(stage, progress)

        Returns:
            List[Dict]: 翻译后的字幕片段
        """
        if not segments:
            return []

        self.logger.info(f"开始翻译字幕: {len(segments)} 个片段")

        if progress_callback:
            progress_callback("translate", 0.0)

        # 提取文本
        texts = [seg.get('text', '') for seg in segments]

        # 批量翻译进度回调
        def batch_progress(p):
            if progress_callback:
                progress_callback("translate", p * 0.9)

        # 批量翻译
        translations = self.translate_batch(
            texts,
            batch_size=batch_size,
            progress_callback=batch_progress
        )

        # 组合结果
        result = []
        for seg, translated in zip(segments, translations):
            new_seg = seg.copy()
            new_seg['translated_text'] = translated
            result.append(new_seg)

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
        """生成双语字幕SRT文件"""
        srt_lines = []

        for i, seg in enumerate(segments, start=1):
            srt_lines.append(str(i))

            start_time = self._format_timestamp(seg['start'])
            end_time = self._format_timestamp(seg['end'])
            srt_lines.append(f"{start_time} --> {end_time}")

            original = seg.get('text', '').strip()
            translated = seg.get('translated_text', '').strip()

            if layout == 'stacked':
                if original:
                    srt_lines.append(original)
                if translated:
                    srt_lines.append(translated)
            elif layout == 'original_first':
                if original:
                    srt_lines.append(original)
            elif layout == 'translation_first':
                if translated:
                    srt_lines.append(translated)
                if original:
                    srt_lines.append(original)

            srt_lines.append('')

        content = '\n'.join(srt_lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
        """生成仅译文字幕"""
        srt_lines = []

        for i, seg in enumerate(segments, start=1):
            srt_lines.append(str(i))

            start_time = self._format_timestamp(seg['start'])
            end_time = self._format_timestamp(seg['end'])
            srt_lines.append(f"{start_time} --> {end_time}")

            translated = seg.get('translated_text', '').strip()
            if translated:
                srt_lines.append(translated)
            else:
                srt_lines.append(seg.get('text', '').strip())

            srt_lines.append('')

        content = '\n'.join(srt_lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"译文字幕已保存: {output_path}")
            return output_path

        return content

    def _format_timestamp(self, seconds: float) -> str:
        """格式化时间戳为SRT格式"""
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

        chinese_count = 0
        japanese_count = 0
        english_count = 0

        for char in text:
            code = ord(char)
            if 0x4E00 <= code <= 0x9FFF:
                chinese_count += 1
            elif 0x3040 <= code <= 0x30FF:
                japanese_count += 1
            elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
                english_count += 1

        total = chinese_count + japanese_count + english_count
        if total == 0:
            return 'unknown'

        if japanese_count > 0:
            return 'ja'
        elif chinese_count / total > 0.3:
            return 'zh'
        elif english_count / total > 0.5:
            return 'en'
        else:
            return 'zh'

    def release(self):
        """释放模型资源"""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        self.logger.debug("LLM模型资源已释放")

    def __del__(self):
        """析构函数"""
        try:
            self.release()
        except Exception:
            pass


# 便捷函数
def create_llm_translator(
    model_tier: Literal['low', 'standard', 'high'] = 'standard',
    src_lang: str = 'zh',
    tgt_lang: str = 'en',
    use_gpu: bool = True
) -> LLMTranslator:
    """
    创建LLM翻译器实例

    Args:
        model_tier: 模型等级 (low/standard/high)
        src_lang: 源语言
        tgt_lang: 目标语言
        use_gpu: 是否使用GPU

    Returns:
        LLMTranslator: 翻译器实例
    """
    return LLMTranslator(
        model_tier=model_tier,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        use_gpu=use_gpu
    )


def get_available_tiers() -> Dict[str, Dict]:
    """获取可用的模型等级配置"""
    return TRANSLATION_MODELS.copy()


# 测试代码
if __name__ == "__main__":
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("\n" + "=" * 60)
    print("  LLM翻译模块测试 (Qwen2.5 + transformers)")
    print("=" * 60)

    # 检查GPU
    print(f"\nGPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # MarianMT翻错的测试案例
    test_cases = [
        ('省力的人生路径', 'MarianMT误译: a way to save your life'),
        ('什么时间干什么事情', 'MarianMT误译: What time is it?'),
        ('考公', 'MarianMT误译: going to the exam'),
        ('非常的省力非常的顺遂', 'MarianMT误译: Very, very, very productive'),
    ]

    try:
        print("\n--- 创建翻译器 (standard: Qwen2.5-3B) ---")
        translator = create_llm_translator(
            model_tier='standard',
            src_lang='zh',
            tgt_lang='en'
        )

        print("\n--- 翻译质量对比 ---")
        for zh_text, marian_note in test_cases:
            result = translator.translate_text(zh_text)
            print(f"\n原文: {zh_text}")
            print(f"  {marian_note}")
            print(f"  Qwen2.5: {result}")

        # 字幕翻译测试
        print("\n--- 字幕翻译测试 ---")
        test_segments = [
            {'start': 0.0, 'end': 3.0, 'text': '大家好，欢迎来到我的频道'},
            {'start': 3.0, 'end': 6.0, 'text': '今天我们来学习Python编程'},
            {'start': 6.0, 'end': 9.0, 'text': '人工智能正在改变我们的世界'},
        ]

        translated = translator.translate_subtitles(test_segments)
        for seg in translated:
            print(f"  [{seg['start']:.1f}s] {seg['text']}")
            print(f"         → {seg['translated_text']}")

        # 显存使用
        if torch.cuda.is_available():
            print(f"\n显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        translator.release()
        print("\n资源已释放")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("  测试完成!")
    print("=" * 60)
