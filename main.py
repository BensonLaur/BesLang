#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bes语坊 (BesLang) - 本地化AI语言处理工坊
主程序启动入口

作者: Benson Laur
邮箱: BensonLaur@163.com
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def log(message: str):
    """简单的日志函数（在 logger 模块导入前使用）"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} | STARTUP | {message}")


def show_splash_and_warmup():
    """
    显示 tkinter 启动画面并进行 transformers 预热

    使用 tkinter 而非 PyQt6 的原因：
    - tokenizers (Rust) 必须在 PyQt6 导入之前预热
    - tkinter 是 Python 内置库，不会与 Rust 代码冲突

    注意：为避免 Tcl_AsyncDelete 错误，预热在主线程中执行，
    tkinter 只负责显示静态画面，不使用多线程。
    """
    import tkinter as tk
    from tkinter import ttk
    import gc

    log("开始显示启动画面...")

    # 创建 tkinter 窗口
    root = tk.Tk()
    root.title("Bes语坊")
    root.overrideredirect(True)  # 无边框
    root.attributes('-topmost', True)  # 置顶

    # 窗口尺寸和居中
    width, height = 480, 340
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")

    # 墨蓝色背景
    bg_color = '#1E3A8A'
    root.configure(bg=bg_color)

    # 品牌名称
    title_label = tk.Label(
        root,
        text="Bes 语坊",
        font=("Microsoft YaHei", 36, "bold"),
        fg="white",
        bg=bg_color
    )
    title_label.pack(pady=(50, 5))

    # 英文名
    en_label = tk.Label(
        root,
        text="BesLang",
        font=("Arial", 18),
        fg="white",
        bg=bg_color
    )
    en_label.pack(pady=(0, 15))

    # Slogan
    slogan_label = tk.Label(
        root,
        text="语言工坊，倍速创作",
        font=("Microsoft YaHei", 14),
        fg='#06B6D4',  # 青瓷色
        bg=bg_color
    )
    slogan_label.pack(pady=(0, 5))

    # 版本号
    version_label = tk.Label(
        root,
        text="v0.2.0",
        font=("Microsoft YaHei", 10),
        fg='#9CA3AF',
        bg=bg_color
    )
    version_label.pack(pady=(0, 25))

    # 进度条容器（用于居中）
    progress_frame = tk.Frame(root, bg=bg_color)
    progress_frame.pack(fill=tk.X, padx=60)

    # 配置进度条样式
    style = ttk.Style()
    style.theme_use('default')
    style.configure(
        "Custom.Horizontal.TProgressbar",
        troughcolor='#1E3A8A',      # 背景色（与窗口一致）
        background='#06B6D4',        # 进度条颜色（青瓷色）
        bordercolor='#3B5998',       # 边框色
        lightcolor='#06B6D4',
        darkcolor='#06B6D4',
        thickness=8
    )

    # 进度条
    progress_bar = ttk.Progressbar(
        progress_frame,
        style="Custom.Horizontal.TProgressbar",
        orient=tk.HORIZONTAL,
        length=360,
        mode='determinate'
    )
    progress_bar.pack(pady=(0, 15))

    # 状态标签（底部）
    status_label = tk.Label(
        root,
        text="正在启动...",
        font=("Microsoft YaHei", 10),
        fg='white',
        bg=bg_color
    )
    status_label.pack()

    log("启动画面已显示")

    warmup_error = None
    current_step = [0]  # 用列表包装以便在闭包中修改

    def update_progress(step: int, status: str):
        """更新进度条和状态文字"""
        # 进度映射：共5个步骤
        progress_map = {
            1: 20,   # 准备环境
            2: 40,   # 加载核心组件
            3: 70,   # 初始化AI引擎
            4: 90,   # 配置完成
            5: 100,  # 启动完成
        }
        progress_bar['value'] = progress_map.get(step, 0)
        status_label.config(text=status)
        root.update()

    def do_warmup_steps():
        """分步执行预热，每步更新进度"""
        nonlocal warmup_error

        try:
            # 步骤1：准备环境
            update_progress(1, "正在准备环境...")
            log("步骤1: 准备环境")
            root.after(200)  # 短暂延迟让用户看到进度
            root.update()

            # 步骤2：加载核心组件
            update_progress(2, "正在加载核心组件...")
            log("步骤2: 加载核心组件")
            root.update()

            # 步骤3：初始化AI引擎（实际的预热工作）
            update_progress(3, "正在初始化AI引擎...")
            log("步骤3: 初始化AI引擎 (transformers预热)")

            # 使用 Qwen2.5 的 AutoTokenizer 进行预热
            # 这会触发 tokenizers (Rust) 的初始化，避免与 PyQt6 冲突
            from transformers import AutoTokenizer
            log("transformers 模块导入成功")

            try:
                log("开始加载 tokenizer...")
                # 尝试从缓存加载，首次运行会自动下载
                _dummy_tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-3B-Instruct",
                    local_files_only=True
                )
                del _dummy_tokenizer
                log("Qwen2.5 tokenizer 预热完成")
            except Exception as e:
                log(f"tokenizer 预热跳过（首次使用时下载）: {e}")

            # 步骤4：配置完成
            update_progress(4, "正在完成配置...")
            log("步骤4: 完成配置")
            root.update()

            # 步骤5：启动完成
            update_progress(5, "启动完成！")
            log("步骤5: 启动完成")
            root.update()

        except Exception as e:
            log(f"预热过程出错: {e}")
            warmup_error = str(e)
            status_label.config(text="初始化遇到问题，继续启动...")
            root.update()

        # 短暂延迟让用户看到 100% 完成状态
        log("准备关闭启动画面...")
        root.after(400, root.destroy)

    # 显示窗口后立即开始预热流程
    root.after(100, do_warmup_steps)

    # 运行 tkinter 主循环
    root.mainloop()

    log("启动画面已关闭")

    # 彻底清理 tkinter 资源
    log("清理 tkinter 资源...")
    try:
        del root
        del title_label
        del en_label
        del slogan_label
        del version_label
        del status_label
        del progress_bar
        del progress_frame
    except Exception:
        pass

    gc.collect()
    log("tkinter 资源清理完成")

    return warmup_error


def main():
    """主程序启动入口"""
    try:
        log("=" * 50)
        log("Bes语坊 启动流程开始")
        log("=" * 50)

        # 第1步：显示启动画面并预热 transformers
        # 这必须在导入 PyQt6 之前完成
        warmup_error = show_splash_and_warmup()

        if warmup_error:
            log(f"警告: 预热过程出错: {warmup_error}")

        # 第2步：导入并运行 PyQt6 应用
        log("开始导入 PyQt6 应用模块...")
        from app import main as app_main
        log("PyQt6 应用模块导入完成")

        log("启动 PyQt6 主应用...")
        app_main()

    except ImportError as e:
        print("=" * 60)
        print("错误：无法导入应用程序模块")
        print("=" * 60)
        print(f"详细信息: {e}")
        print()
        print("请确保:")
        print("1. src/app.py 文件存在")
        print("2. 已安装所有依赖: pip install -r requirements.txt")
        print("3. 已激活虚拟环境: .\\venv\\Scripts\\activate")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print("=" * 60)
        print("错误：程序启动失败")
        print("=" * 60)
        print(f"详细信息: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
