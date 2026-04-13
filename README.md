# BesLang

> 一款本地化的视频字幕识别与翻译工具。完全在你的电脑上运行,不依赖任何在线服务,保护隐私,无需订阅。

**BesLang 延续了作者 2017 年开源的 [BesLyric](https://github.com/BensonLaur/BesLyric-for-X) 项目的命名传承——同一位开发者对"用工具解决自己真实需求"这件事的第二次尝试。**

---

## 这个项目是什么

作者本人曾想把英文视频搬到中文平台,发现现有方案全部依赖云端服务、需要订阅、隐私不保。BesLang 就是为了解决这个问题——一个本地运行、一键式、买断制的字幕识别和翻译工具。

**核心能力**:
- 🎙️ **语音识别**:基于 OpenAI Whisper,支持任意时长和分辨率的视频
- 🌐 **智能翻译**:基于本地运行的 Qwen 小模型,高质量中英互译
- 📄 **多种输出**:单语字幕 / 双语对照 / 纯译文,支持 SRT、VTT、TXT 格式
- 🔒 **完全本地**:不联网、不上传、不订阅——你的数据只在你的电脑上
- 💻 **跨平台(规划中)**:当前支持 Windows,未来将覆盖 macOS 和 Linux

---

## 快速开始

### 环境要求

- Python 3.10+
- Windows 10/11(其他平台的打包版本正在开发中)
- 推荐 4GB+ GPU 显存(支持 CPU 运行,但速度较慢)
- FFmpeg(用于视频处理)

### 安装

```powershell
# 克隆仓库
git clone https://github.com/BensonLaur/BesLang.git
cd BesLang

# 创建虚拟环境(推荐)
python -m venv venv
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 运行

```powershell
python main.py
```

拖拽视频文件到界面,选择模型和语言,一键生成字幕。

> **提示**:首次运行会自动下载所需的 AI 模型,请保持网络连接。

---

## 双轨理念:开源版 + Pro 版

BesLang 采用 **Open Core** 模式,分为两个互补的版本:

### BesLang(本仓库)- 开源免费

由作者用爱发电持续维护,遵循 **GPL 3.0 协议**。基础版本包含**完整的**字幕识别与翻译能力,**没有任何时长、分辨率、功能阉割**——对个人用户来说就是一个完整可用的工具。

### BesLang Pro - 未来推出

面向有专业需求的用户,提供批量处理、术语库管理、专业字幕编辑、深度 UI 定制、自动化集成、商业授权等增值能力。Pro 版采用**一次付费终身使用**的买断制,预计 2026 年下半年推出。

**如果你是需要专业工作流的重度用户,或者想为这个项目补充电力支持作者持续投入,欢迎未来了解 Pro 版本。**

无论你使用哪个版本,都欢迎加入 BesLang 的用户社区——每一个用它生成字幕的人,都是这个项目存在意义的一部分。

---

## 贡献与反馈

当前阶段**不接受代码类 PR**,但非常欢迎以下形式的参与:

- 🐛 [提交 Bug 报告](https://github.com/BensonLaur/BesLang/issues)
- 💡 [提出功能建议](https://github.com/BensonLaur/BesLang/issues)
- 📝 分享你的使用场景和反馈
- ⭐ 如果这个工具对你有用,给个 star 就是最好的支持

> **关于 PR**:由于本项目采用双重许可(Dual Licensing)——同一份代码既以 GPL 3.0 开源,未来也将用于闭源的 BesLang Pro——为避免贡献者版权的复杂性,当前阶段暂不接受代码合并。感谢你的理解。

---

## 关于作者

**Benson Laur** — 一位从 2017 年就开始做独立桌面工具的开发者。

- BesLyric(2017~):网易云音乐滚动歌词制作工具,已稳定运行 9 年
- BesLang(2026~):本地化视频字幕识别与翻译工具

独立开发者的作品不是用来"颠覆行业"的,它们存在的意义是:**被真实的人用上,解决真实的问题,并且长期被维护**。

---

## 许可

本项目以 **GPL 3.0** 协议开源。详见 [LICENSE](LICENSE) 文件。

本项目所有核心代码均由 Benson Laur 独立开发。作者作为版权持有人,保留以商业许可对同一代码进行重新授权用于 BesLang Pro 商业产品的权利(双重许可)。

---

## English

**BesLang** is a fully local video subtitle recognition and translation tool. Powered by OpenAI Whisper and Qwen LLM, it runs entirely on your machine — no cloud, no subscription, no privacy concerns.

This is a Python MVP (v0.x). A native C++ rewrite is planned for late 2026, which will also introduce the commercial **BesLang Pro** edition for professional users.

See the Chinese section above for detailed usage and project philosophy. For questions and issues, feel free to open a GitHub issue in English or Chinese.

**License**: GPL 3.0
