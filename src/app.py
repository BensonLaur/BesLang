#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bes语坊 (BesLang) - 主程序入口

产品名称：Bes语坊 / BesLang
核心理念：语言转文字，文字变语音
作者：Benson Laur
邮箱：BensonLaur@163.com
版本：v0.2.0 (Phase 1 - Week 3)
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# ============================================================
# 注意：transformers 预热已移至 main.py 的 tkinter 启动画面中
# 这样可以在显示启动画面的同时进行预热，提升用户体验
# ============================================================

# 导入 PyQt6（预热已在 main.py 中完成）
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QMenuBar, QMenu, QStatusBar, QMessageBox, QTabWidget,
    QPushButton, QTextEdit, QFileDialog, QComboBox, QGroupBox,
    QRadioButton, QButtonGroup, QProgressBar, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QDragEnterEvent, QDropEvent

# 导入日志系统
from utils.logger import get_logger

# 导入字幕生成模块
from core.subtitle import SubtitleGenerator

# 导入翻译模块（使用Qwen2.5 LLM翻译器，替代MarianMT）
from core.llm_translator import LLMTranslator, create_llm_translator

# 获取全局日志实例
logger = get_logger()


class SubtitleWorker(QThread):
    """字幕生成工作线程（避免UI卡顿）"""

    # 信号定义
    progress_updated = pyqtSignal(str, float)  # (阶段, 进度)
    finished = pyqtSignal(str, str)  # 完成信号(输出路径, 模式)
    error = pyqtSignal(str)  # 错误信号(错误信息)
    # 请求在主线程创建翻译器的信号
    request_translator = pyqtSignal(str, str)  # (src_lang, tgt_lang)

    def __init__(
        self,
        video_path: str,
        model_name: str,
        language: str,
        output_mode: str = 'single',  # single/bilingual/translation_only
        target_lang: str = 'zh'
    ):
        super().__init__()
        self.video_path = video_path
        self.model_name = model_name
        self.language = language
        self.output_mode = output_mode
        self.target_lang = target_lang
        # 翻译器将由主线程传入
        self.translator = None

    def run(self):
        """执行字幕生成任务"""
        try:
            logger.info(f"开始字幕生成任务: {self.video_path}")
            logger.info(f"配置: 模型={self.model_name}, 语言={self.language}, 模式={self.output_mode}")

            # 创建字幕生成器
            generator = SubtitleGenerator(
                model_name=self.model_name,
                language=self.language if self.language != "auto" else None
            )

            # 步骤1: 提取音频
            self._progress_callback("extract_audio", 0.0)
            audio_path = generator.extract_audio(
                self.video_path,
                progress_callback=self._progress_callback
            )

            # 步骤2: 识别语音
            self._progress_callback("recognize", 0.0)
            segments = generator.recognize(
                audio_path,
                progress_callback=self._progress_callback
            )

            # 生成基础输出路径
            base_path = Path(self.video_path).with_suffix('')

            if self.output_mode == 'single':
                # 单语字幕 - 直接保存
                output_path = str(base_path) + '.srt'
                self._progress_callback("save_srt", 0.0)
                generator.save_srt(segments, output_path)
                self._progress_callback("save_srt", 1.0)

                # 释放Whisper模型
                del generator
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.finished.emit(output_path, 'single')

            else:
                # 需要翻译 - 先释放Whisper模型以腾出显存
                logger.info("释放Whisper模型以腾出显存...")
                del generator
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Whisper模型已释放")

                self._progress_callback("translate", 0.0)

                # 确定翻译方向
                src_lang = self.language if self.language != 'auto' else 'en'
                tgt_lang = self.target_lang

                # 自动检测源语言
                if self.language == 'auto' and segments:
                    sample_text = ' '.join([s['text'] for s in segments[:5]])
                    detected = LLMTranslator.detect_language(sample_text)
                    if detected != 'unknown':
                        src_lang = detected

                # 如果源语言和目标语言相同，自动调整目标语言
                if src_lang == tgt_lang:
                    if src_lang == 'zh':
                        tgt_lang = 'en'
                    else:
                        tgt_lang = 'zh'
                    logger.warning(f"源语言和目标语言相同，自动调整: {src_lang} -> {tgt_lang}")

                logger.info(f"翻译方向: {src_lang} -> {tgt_lang}")

                # 创建LLM翻译器（使用Qwen2.5-3B，4bit量化，显存约2GB）
                self.translator = create_llm_translator(
                    model_tier='standard',  # Qwen2.5-3B
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    use_gpu=True
                )

                # 翻译字幕
                translated_segments = self.translator.translate_subtitles(
                    segments,
                    progress_callback=self._progress_callback
                )

                # 保存字幕
                self._progress_callback("save_srt", 0.0)

                if self.output_mode == 'bilingual':
                    # 双语字幕
                    output_path = str(base_path) + '_bilingual.srt'
                    self.translator.generate_bilingual_srt(
                        translated_segments,
                        layout='stacked',
                        output_path=output_path
                    )
                    self.finished.emit(output_path, 'bilingual')

                elif self.output_mode == 'translation_only':
                    # 仅译文
                    output_path = str(base_path) + f'_{self.target_lang}.srt'
                    self.translator.generate_translation_only_srt(
                        translated_segments,
                        output_path=output_path
                    )
                    self.finished.emit(output_path, 'translation_only')

                self._progress_callback("save_srt", 1.0)

                # 释放翻译器资源
                self.translator.release()

            logger.info(f"字幕生成任务完成: {output_path}")

            # 清理临时文件
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass

        except Exception as e:
            error_msg = str(e)
            self.error.emit(error_msg)
            logger.error(f"字幕生成任务失败: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())

    def _progress_callback(self, stage: str, progress: float):
        """进度回调函数"""
        self.progress_updated.emit(stage, progress)


class SmartSubtitleTab(QWidget):
    """智能字幕标签页"""

    def __init__(self):
        super().__init__()
        logger.debug("初始化智能字幕标签页")
        self.video_path = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 1. 标题区域
        title_label = QLabel("🎙️ 智能字幕生成")
        title_font = QFont("Microsoft YaHei", 16, QFont.Weight.Bold)
        title_label.setFont(title_font)

        subtitle_label = QLabel("语言 → 文字：单语识别、双语翻译")
        subtitle_label.setStyleSheet("color: #666; margin-bottom: 10px;")

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)

        # 2. 文件选择区域
        file_group = self.create_file_selection_group()
        layout.addWidget(file_group)

        # 3. 配置选项区域
        config_group = self.create_config_group()
        layout.addWidget(config_group)

        # 4. 开始按钮
        self.btn_start = QPushButton("🚀 开始生成字幕")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:disabled {
                background-color: #9CA3AF;
            }
        """)
        self.btn_start.clicked.connect(self.start_generation)
        self.btn_start.setEnabled(False)
        layout.addWidget(self.btn_start)

        # 5. 进度显示区域
        progress_group = self.create_progress_group()
        layout.addWidget(progress_group)

        # 6. 结果预览区域
        result_group = self.create_result_group()
        layout.addWidget(result_group)

        # 添加弹性空间
        layout.addStretch()

        self.setLayout(layout)

    def create_file_selection_group(self) -> QGroupBox:
        """创建文件选择区域"""
        group = QGroupBox("📁 选择视频文件")
        layout = QVBoxLayout()

        # 文件路径显示
        self.label_file_path = QLabel("未选择文件")
        self.label_file_path.setStyleSheet("""
            padding: 10px;
            background-color: #F3F4F6;
            border-radius: 5px;
            color: #6B7280;
        """)
        layout.addWidget(self.label_file_path)

        # 选择按钮
        btn_select = QPushButton("浏览文件...")
        btn_select.clicked.connect(self.select_video_file)
        layout.addWidget(btn_select)

        group.setLayout(layout)
        return group

    def create_config_group(self) -> QGroupBox:
        """创建配置选项区域"""
        group = QGroupBox("⚙️ 配置选项")
        layout = QVBoxLayout()

        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("识别模型:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["base (推荐)", "tiny (快速)", "small (高质量)", "medium (更高质量)", "large (最高质量)"])
        model_layout.addWidget(self.combo_model)
        model_layout.addStretch()
        layout.addLayout(model_layout)

        # 语言选择
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("识别语言:"))
        self.combo_language = QComboBox()
        self.combo_language.addItems(["自动检测", "中文 (zh)", "英文 (en)", "日文 (ja)"])
        lang_layout.addWidget(self.combo_language)
        lang_layout.addStretch()
        layout.addLayout(lang_layout)

        # 分隔线
        layout.addSpacing(10)

        # 输出选项标签
        output_label = QLabel("📤 输出选项:")
        output_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(output_label)

        # 单语字幕选项
        self.radio_single = QRadioButton("单语字幕（仅识别，不翻译）")
        self.radio_single.setChecked(True)
        self.radio_single.toggled.connect(self.on_output_mode_changed)
        layout.addWidget(self.radio_single)

        # 双语字幕选项
        self.radio_bilingual = QRadioButton("双语字幕（识别 + 翻译）")
        self.radio_bilingual.toggled.connect(self.on_output_mode_changed)
        layout.addWidget(self.radio_bilingual)

        # 仅译文选项
        self.radio_translation = QRadioButton("仅译文（翻译后不保留原文）")
        self.radio_translation.toggled.connect(self.on_output_mode_changed)
        layout.addWidget(self.radio_translation)

        # 翻译目标语言选择（默认隐藏）
        self.target_lang_layout = QHBoxLayout()
        self.target_lang_layout.addSpacing(20)
        self.label_target_lang = QLabel("译为:")
        self.combo_target_lang = QComboBox()
        self.combo_target_lang.addItems(["中文 (zh)", "英文 (en)"])
        self.target_lang_layout.addWidget(self.label_target_lang)
        self.target_lang_layout.addWidget(self.combo_target_lang)
        self.target_lang_layout.addStretch()

        # 将翻译选项包装成widget以便显示/隐藏
        self.target_lang_widget = QWidget()
        self.target_lang_widget.setLayout(self.target_lang_layout)
        self.target_lang_widget.setVisible(False)
        layout.addWidget(self.target_lang_widget)

        group.setLayout(layout)
        return group

    def on_output_mode_changed(self):
        """输出模式切换时的处理"""
        # 显示/隐藏翻译目标语言选择
        need_translation = self.radio_bilingual.isChecked() or self.radio_translation.isChecked()
        self.target_lang_widget.setVisible(need_translation)

    def create_progress_group(self) -> QGroupBox:
        """创建进度显示区域"""
        group = QGroupBox("📊 进度")
        layout = QVBoxLayout()

        # 当前阶段标签
        self.label_current_stage = QLabel("等待开始...")
        self.label_current_stage.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.label_current_stage)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 阶段详情
        self.label_stage_extract = QLabel("⏸  提取音频")
        self.label_stage_recognize = QLabel("⏸  识别语音")
        self.label_stage_translate = QLabel("⏸  翻译字幕")
        self.label_stage_save = QLabel("⏸  保存字幕")

        layout.addWidget(self.label_stage_extract)
        layout.addWidget(self.label_stage_recognize)
        layout.addWidget(self.label_stage_translate)
        layout.addWidget(self.label_stage_save)

        # 默认隐藏翻译阶段
        self.label_stage_translate.setVisible(False)

        group.setLayout(layout)
        return group

    def create_result_group(self) -> QGroupBox:
        """创建结果预览区域"""
        group = QGroupBox("📄 结果预览")
        layout = QVBoxLayout()

        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setPlaceholderText("字幕内容将在这里显示...")
        self.text_result.setMinimumHeight(200)
        layout.addWidget(self.text_result)

        group.setLayout(layout)
        return group

    def select_video_file(self):
        """选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv);;所有文件 (*.*)"
        )

        if file_path:
            self.video_path = file_path
            self.label_file_path.setText(file_path)
            self.label_file_path.setStyleSheet("""
                padding: 10px;
                background-color: #DBEAFE;
                border-radius: 5px;
                color: #1E40AF;
            """)
            self.btn_start.setEnabled(True)
            logger.info(f"选择视频文件: {file_path}")

    def start_generation(self):
        """开始生成字幕"""
        if not self.video_path:
            QMessageBox.warning(self, "错误", "请先选择视频文件！")
            return

        # 禁用开始按钮
        self.btn_start.setEnabled(False)
        self.btn_start.setText("⏳ 正在生成...")

        # 确定输出模式
        if self.radio_single.isChecked():
            output_mode = 'single'
        elif self.radio_bilingual.isChecked():
            output_mode = 'bilingual'
        else:
            output_mode = 'translation_only'

        # 显示/隐藏翻译阶段
        need_translation = output_mode != 'single'
        self.label_stage_translate.setVisible(need_translation)

        # 重置进度显示
        self.progress_bar.setValue(0)
        self.label_current_stage.setText("正在初始化...")
        self.label_stage_extract.setText("⏸  提取音频")
        self.label_stage_recognize.setText("⏸  识别语音")
        self.label_stage_translate.setText("⏸  翻译字幕")
        self.label_stage_save.setText("⏸  保存字幕")
        self.text_result.clear()

        # 获取配置
        model_text = self.combo_model.currentText()
        model_name = model_text.split()[0]  # "base (推荐)" -> "base"

        lang_text = self.combo_language.currentText()
        if lang_text == "自动检测":
            language = "auto"
        else:
            language = lang_text.split()[-1].strip("()")  # "中文 (zh)" -> "zh"

        # 获取目标语言
        target_lang_text = self.combo_target_lang.currentText()
        target_lang = target_lang_text.split()[-1].strip("()")  # "中文 (zh)" -> "zh"

        logger.info(f"开始生成字幕: 模型={model_name}, 语言={language}, 模式={output_mode}, 目标语言={target_lang}")

        # 创建并启动工作线程
        # 注意：翻译器现在在工作线程中创建（因为已在程序启动时预热了transformers）
        self.worker = SubtitleWorker(
            self.video_path,
            model_name,
            language,
            output_mode=output_mode,
            target_lang=target_lang
        )
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.error.connect(self.on_generation_error)
        self.worker.start()

    def on_progress_updated(self, stage: str, progress: float):
        """进度更新回调"""
        # 判断是否需要翻译
        need_translation = self.label_stage_translate.isVisible()

        # 根据是否需要翻译调整进度分配
        if need_translation:
            # 有翻译: 提取10%, 识别40%, 翻译40%, 保存10%
            if stage == "extract_audio":
                total_progress = int(progress * 10)  # 0-10%
                self.label_current_stage.setText("正在提取音频...")
                if progress >= 1.0:
                    self.label_stage_extract.setText("✅ 提取音频 (完成)")
            elif stage == "recognize":
                total_progress = 10 + int(progress * 40)  # 10-50%
                self.label_current_stage.setText("正在识别语音...")
                self.label_stage_extract.setText("✅ 提取音频 (完成)")
                if progress >= 1.0:
                    self.label_stage_recognize.setText("✅ 识别语音 (完成)")
            elif stage == "translate":
                total_progress = 50 + int(progress * 40)  # 50-90%
                self.label_current_stage.setText("正在翻译字幕...")
                self.label_stage_recognize.setText("✅ 识别语音 (完成)")
                if progress >= 1.0:
                    self.label_stage_translate.setText("✅ 翻译字幕 (完成)")
            elif stage == "save_srt":
                total_progress = 90 + int(progress * 10)  # 90-100%
                self.label_current_stage.setText("正在保存字幕...")
                self.label_stage_translate.setText("✅ 翻译字幕 (完成)")
                if progress >= 1.0:
                    self.label_stage_save.setText("✅ 保存字幕 (完成)")
            else:
                total_progress = 0
        else:
            # 无翻译: 提取20%, 识别70%, 保存10%
            if stage == "extract_audio":
                total_progress = int(progress * 20)  # 0-20%
                self.label_current_stage.setText("正在提取音频...")
                if progress >= 1.0:
                    self.label_stage_extract.setText("✅ 提取音频 (完成)")
            elif stage == "recognize":
                total_progress = 20 + int(progress * 70)  # 20-90%
                self.label_current_stage.setText("正在识别语音...")
                self.label_stage_extract.setText("✅ 提取音频 (完成)")
                if progress >= 1.0:
                    self.label_stage_recognize.setText("✅ 识别语音 (完成)")
            elif stage == "save_srt":
                total_progress = 90 + int(progress * 10)  # 90-100%
                self.label_current_stage.setText("正在保存字幕...")
                self.label_stage_recognize.setText("✅ 识别语音 (完成)")
                if progress >= 1.0:
                    self.label_stage_save.setText("✅ 保存字幕 (完成)")
            else:
                total_progress = 0

        self.progress_bar.setValue(total_progress)

    def on_generation_finished(self, output_path: str, output_mode: str):
        """生成完成回调"""
        self.label_current_stage.setText("✅ 生成完成！")
        self.progress_bar.setValue(100)

        # 恢复按钮状态
        self.btn_start.setEnabled(True)
        self.btn_start.setText("🚀 开始生成字幕")

        # 根据模式显示不同的成功信息
        mode_names = {
            'single': '单语字幕',
            'bilingual': '双语字幕',
            'translation_only': '译文字幕'
        }
        mode_name = mode_names.get(output_mode, '字幕')

        # 显示结果
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.text_result.setPlainText(content)

            QMessageBox.information(
                self,
                "成功",
                f"{mode_name}生成成功！\n\n文件保存在:\n{output_path}"
            )
            logger.info(f"{mode_name}生成任务成功完成")

        except Exception as e:
            logger.error(f"读取字幕文件失败: {e}")

    def on_generation_error(self, error_msg: str):
        """生成错误回调"""
        self.label_current_stage.setText("❌ 生成失败")

        # 恢复按钮状态
        self.btn_start.setEnabled(True)
        self.btn_start.setText("🚀 开始生成字幕")

        QMessageBox.critical(
            self,
            "错误",
            f"字幕生成失败！\n\n错误信息:\n{error_msg}"
        )
        logger.error(f"字幕生成失败: {error_msg}")


class MainWindow(QMainWindow):
    """Bes语坊主窗口"""

    def __init__(self):
        super().__init__()
        logger.info("=" * 60)
        logger.info("Bes语坊 (BesLang) 正在启动...")
        logger.info("版本: v0.2.0 | 阶段: Phase 1 - Week 3")
        logger.info("=" * 60)

        self.init_ui()
        self.setup_statusbar_timer()

        logger.info("主窗口初始化完成")

    def init_ui(self):
        """初始化用户界面"""
        # 1. 设置窗口基本属性
        self.setWindowTitle("Bes语坊 - BesLang Language Workshop")
        self.setGeometry(100, 100, 1200, 800)

        # 2. 居中显示窗口
        self.center_window()

        # 3. 创建菜单栏
        self.create_menu_bar()

        # 4. 创建中央区域（欢迎信息）
        self.create_central_widget()

        # 5. 创建状态栏
        self.create_status_bar()

    def center_window(self):
        """将窗口居中显示"""
        screen = QApplication.primaryScreen().geometry()
        window_geometry = self.frameGeometry()
        center_point = screen.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")

        # 退出操作
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("退出程序")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")

        # 关于操作
        about_action = QAction("关于(&A)", self)
        about_action.setStatusTip("关于Bes语坊")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_central_widget(self):
        """创建中央区域 - 标签页"""
        # 创建标签页控件
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # 添加智能字幕标签页
        self.subtitle_tab = SmartSubtitleTab()
        self.tab_widget.addTab(self.subtitle_tab, "🎙️ 智能字幕")

        # 添加欢迎/关于标签页
        welcome_tab = self.create_welcome_tab()
        self.tab_widget.addTab(welcome_tab, "🏠 关于")

        logger.debug("标签页创建完成")

    def create_welcome_tab(self) -> QWidget:
        """创建欢迎标签页"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 创建欢迎标题
        title_label = QLabel("欢迎使用 Bes语坊")
        title_font = QFont("Microsoft YaHei", 32, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 创建副标题（核心理念）
        subtitle_label = QLabel("语言转文字，文字变语音")
        subtitle_font = QFont("Microsoft YaHei", 18)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #666; margin-top: 20px;")

        # 创建功能说明
        features_label = QLabel(
            "核心功能：\n\n"
            "🎙️ 语言 → 文字：单语识别、双语翻译\n"
            "🔊 文字 → 语音：音色克隆、智能配音、文本朗读"
        )
        features_font = QFont("Microsoft YaHei", 12)
        features_label.setFont(features_font)
        features_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        features_label.setStyleSheet("color: #888; margin-top: 40px; line-height: 1.8;")

        # 创建版本信息
        version_label = QLabel("v0.2.0 - Week 3 (单语+双语字幕)")
        version_font = QFont("Microsoft YaHei", 10)
        version_label.setFont(version_font)
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("color: #aaa; margin-top: 60px;")

        # 添加到布局
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addWidget(features_label)
        layout.addWidget(version_label)

        widget.setLayout(layout)
        return widget

    def create_status_bar(self):
        """创建状态栏"""
        self.statusbar = self.statusBar()

        # 状态栏左侧显示固定信息
        self.statusbar.showMessage("就绪")

        # 状态栏右侧显示当前时间（将在timer中更新）
        self.time_label = QLabel()
        self.statusbar.addPermanentWidget(self.time_label)

        # 初始化时间显示
        self.update_time()

    def setup_statusbar_timer(self):
        """设置状态栏时间更新定时器"""
        # 创建定时器，每秒更新一次时间
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # 1000毫秒 = 1秒

    def update_time(self):
        """更新状态栏的时间显示"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(f"当前时间: {current_time}")

    def show_about(self):
        """显示关于对话框"""
        about_text = (
            "<h2>Bes语坊 (BesLang)</h2>"
            "<p><b>版本：</b>v0.2.0</p>"
            "<p><b>核心理念：</b>语言转文字，文字变语音</p>"
            "<p><b>Slogan：</b>语言工坊，倍速创作</p>"
            "<br>"
            "<p><b>开发者：</b>Benson Laur</p>"
            "<p><b>邮箱：</b>BensonLaur@163.com</p>"
            "<br>"
            "<p><b>技术栈：</b></p>"
            "<ul>"
            "<li>GUI框架：PyQt6</li>"
            "<li>语音识别：OpenAI Whisper</li>"
            "<li>机器翻译：MarianMT</li>"
            "<li>音色克隆：OpenVoice</li>"
            "<li>推理引擎：ONNX Runtime</li>"
            "</ul>"
            "<br>"
            "<p style='color: #666;'>100%本地化，隐私优先，买断制</p>"
        )

        QMessageBox.about(self, "关于 Bes语坊", about_text)

    def closeEvent(self, event):
        """窗口关闭事件"""
        reply = QMessageBox.question(
            self,
            "确认退出",
            "确定要退出 Bes语坊 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            logger.info("用户确认退出")
            logger.info("Bes语坊 (BesLang) 已关闭")
            logger.info("=" * 60)
            event.accept()
        else:
            logger.debug("用户取消退出操作")
            event.ignore()


def main():
    """主函数"""
    try:
        # 创建应用程序实例
        app = QApplication(sys.argv)

        # 设置应用程序元信息
        app.setApplicationName("Bes语坊")
        app.setOrganizationName("Benson Laur")
        app.setOrganizationDomain("github.com/BensonLaur/BesLang")

        # 创建并显示主窗口
        window = MainWindow()
        window.show()

        logger.info("应用程序已启动，进入事件循环")

        # 进入事件循环
        exit_code = app.exec()

        logger.info(f"应用程序退出，退出码: {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logger.exception(f"程序运行时发生严重错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
