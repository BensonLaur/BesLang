#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bes语坊日志系统

功能：
- 同时输出到控制台和文件
- 不同级别用不同颜色显示（控制台）
- 日志文件按日期自动滚动
- 支持单例模式，全局统一日志对象
"""

import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path


# ANSI颜色代码（用于控制台输出）
class ColorCode:
    """终端颜色代码"""
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（仅用于控制台）"""

    # 不同日志级别对应的颜色
    LEVEL_COLORS = {
        logging.DEBUG: ColorCode.CYAN,
        logging.INFO: ColorCode.GREEN,
        logging.WARNING: ColorCode.YELLOW,
        logging.ERROR: ColorCode.RED,
        logging.CRITICAL: ColorCode.BOLD + ColorCode.RED,
    }

    def format(self, record):
        """格式化日志记录"""
        # 保存原始的levelname
        original_levelname = record.levelname

        # 获取日志级别对应的颜色
        level_color = self.LEVEL_COLORS.get(record.levelno, ColorCode.RESET)

        # 给日志级别名称添加颜色
        record.levelname = f"{level_color}{record.levelname}{ColorCode.RESET}"

        # 调用父类的格式化方法
        formatted = super().format(record)

        # 恢复原始的levelname（避免影响其他handler）
        record.levelname = original_levelname

        return formatted


class Logger:
    """
    Bes语坊日志管理器（单例模式）

    使用方法：
        from utils.logger import get_logger

        logger = get_logger()
        logger.debug("调试信息")
        logger.info("普通信息")
        logger.warning("警告信息")
        logger.error("错误信息")
        logger.critical("严重错误")
    """

    _instance = None

    def __new__(cls):
        """单例模式：确保只有一个日志实例"""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化日志系统"""
        if self._initialized:
            return

        # 创建日志目录
        self.log_dir = Path(__file__).parent.parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # 创建logger对象
        self.logger = logging.getLogger("BesLang")
        self.logger.setLevel(logging.DEBUG)

        # 避免重复添加handler
        if not self.logger.handlers:
            # 添加控制台处理器
            self._add_console_handler()

            # 添加文件处理器
            self._add_file_handler()

        self._initialized = True

    def _add_console_handler(self):
        """添加控制台处理器（带颜色）"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # 使用带颜色的格式化器
        console_formatter = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        """添加文件处理器（按日期自动滚动）"""
        # 日志文件路径
        log_file = self.log_dir / "app.log"

        # 创建时间滚动文件处理器
        # when='midnight': 每天午夜创建新日志文件
        # interval=1: 每1天滚动一次
        # backupCount=30: 保留30天的日志文件
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)

        # 设置日志文件名后缀（日期格式）
        file_handler.suffix = "%Y-%m-%d.log"

        # 文件格式化器（不需要颜色，使用普通Formatter）
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(file_handler)

    def debug(self, message: str):
        """调试信息"""
        self.logger.debug(message)

    def info(self, message: str):
        """普通信息"""
        self.logger.info(message)

    def warning(self, message: str):
        """警告信息"""
        self.logger.warning(message)

    def error(self, message: str):
        """错误信息"""
        self.logger.error(message)

    def critical(self, message: str):
        """严重错误"""
        self.logger.critical(message)

    def exception(self, message: str):
        """异常信息（会自动记录堆栈跟踪）"""
        self.logger.exception(message)


# 全局日志实例（单例）
_global_logger = None


def get_logger() -> Logger:
    """
    获取全局日志实例

    Returns:
        Logger: 日志管理器实例

    Example:
        >>> from utils.logger import get_logger
        >>> logger = get_logger()
        >>> logger.info("程序启动")
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger


# 测试代码
if __name__ == "__main__":
    """测试日志系统"""
    logger = get_logger()

    logger.debug("这是一条调试信息（DEBUG）")
    logger.info("这是一条普通信息（INFO）")
    logger.warning("这是一条警告信息（WARNING）")
    logger.error("这是一条错误信息（ERROR）")
    logger.critical("这是一条严重错误（CRITICAL）")

    # 测试异常日志
    try:
        1 / 0
    except Exception as e:
        logger.exception("捕获到异常")

    print("\n日志已同时输出到控制台和文件：logs/app.log")
