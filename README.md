# 三种生成对抗网络对比：GAN / WGAN / WGAN-GP（MNIST & Fashion-MNIST）

本项目对比了三种常见的生成对抗网络：GAN、WGAN 与 WGAN-GP，在 MNIST 与 Fashion-MNIST 两个数据集上的训练稳定性与生成质量。代码、实验流程与结果可通过根目录的 `asg6_GroupO.ipynb`（及导出的 `asg6_GroupO.py`）和 `code` 文件夹中的独立脚本复现。

- 主要 Notebook：`asg6_GroupO.ipynb`
- 同步脚本版本：`asg6_GroupO.py`
- 独立最小可运行脚本（TensorFlow/Keras）：
  - MNIST：`code/GAN/GAN.py`
  - Fashion-MNIST：`code/GAN/Fashion GAN.py`
- 结果可视化样例保存在：
  - `code/GAN/outputs original/`
  - `code/GAN/outputs fashion/`
  - `code/WGAN/outputs(mnist)/`
  - `code/WGAN/outputs(fashion-mnist)/`
  - `code/WGAN-GP/WGAN-outputs/`


## 项目概述
- 目标：比较 GAN、WGAN、WGAN-GP 三种方法的训练稳定性与图像质量，缓解原始 GAN 的训练不稳定与模式坍塌问题，为更可靠的生成模型提供参考。
- 数据：MNIST 与 Fashion-MNIST，均为 28×28×1 灰度图，像素归一化到 [-1, 1]。
- 输入/输出：随机噪声 $z ∈ R^{100}$ → 生成器输出 28×28×1 图像；判别器/critic 输出实数或概率。
- 损失与差异：
  - GAN：二元交叉熵近似 JS 散度，易出现梯度消失与不稳定。
  - WGAN：优化 Wasserstein 距离（Earth-Mover 距离），使用权重裁剪（weight clipping）约束 1-Lipschitz。
  - WGAN-GP：以梯度惩罚（gradient penalty）代替裁剪，更稳定地满足 1-Lipschitz 约束。

Notebook 中包含问题描述、实现、可视化与结果分析；独立脚本提供“开箱即跑”的最小示例。


## 目录结构
- `asg6_GroupO.ipynb`：完整实验流程（问题描述、实现、对比、可视化、分析）。
- `asg6_GroupO.py`：Notebook 的脚本导出版，包含多个分段训练（GAN/WGAN/WGAN-GP，MNIST/Fashion-MNIST）。
- `requirements.txt`：Python 依赖（用于快速安装）。
- `code/`：独立脚本与结果图
  - `GAN/`
    - `GAN.py`：MNIST 上的 GAN/WGAN/WGAN-GP 最小脚本（通过注释切换）。
    - `Fashion GAN.py`：Fashion-MNIST 上的对应最小脚本。
    - `outputs original/`、`outputs fashion/`：示例结果图（Epoch 0/5/10/14 与最终报告图）。
  - `WGAN/outputs(mnist|fashion-mnist)/`：WGAN 对比结果图。
  - `WGAN-GP/WGAN-outputs/`：WGAN-GP 对比结果图（含 MNIST 与 Fashion-MNIST）。


## 环境与依赖
- Python 3.9–3.11（建议 3.10/3.11）
- 主要依赖：
  - tensorflow / keras
  - numpy
  - matplotlib
  - tqdm

Apple Silicon（M1/M2/M3）可选安装：
- `tensorflow-macos` 与 `tensorflow-metal` 可显著加速训练。

安装示例（建议使用虚拟环境）：
```zsh
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖（推荐）
pip install -r requirements.txt

# 通用 CPU 版本（x86/Intel）
pip install "tensorflow>=2.12,<2.16" numpy matplotlib tqdm

# Apple Silicon（M1/M2/M3）推荐
pip install numpy matplotlib tqdm
pip install tensorflow-macos tensorflow-metal
```

注：不同 Mac/芯片与 Python 版本组合可能略有差异，若遇到安装问题请参考 TensorFlow 官方安装指南。


## 快速开始
你可以选择以下任一方式复现实验：

- 方式 A：运行 Notebook（推荐用于阅读与一次性复现）
  1. 打开 `asg6_GroupO.ipynb`。
  2. 按章节依次执行。注意：其中 WGAN-GP 有一段面向 Colab/Google Drive 的保存路径（`/content/drive/...`），如在本地运行，请修改或注释 Drive 挂载与输出目录（见下方“注意事项”）。

- 方式 B：运行独立最小脚本（简单直跑）
  - MNIST（默认执行 train_gan，可切换为 WGAN/WGAN-GP）
    ```zsh
    cd code/GAN
    python GAN.py
    ```
  - Fashion-MNIST（默认执行 train_gan，可切换为 WGAN/WGAN-GP）
    ```zsh
    cd code/GAN
    python "Fashion GAN.py"
    ```
  在脚本文件末尾取消/注释如下调用即可切换模型：
  ```python
  train_gan()
  # train_wgan()
  # train_wgan_gp()
  ```

- 方式 C：运行导出脚本 `asg6_GroupO.py`（包含多段实验）
  ```zsh
  python asg6_GroupO.py
  ```
  依据脚本内的注释，按需注释/放开不同行的训练函数（如 `train_wgan_mnist()`、`train_wgan_fashion_mnist()`、`train_wgan_gp(...)`）。


## 训练细节
- 输入噪声维度：100
- 图像分辨率：28×28×1（tanh 输出，对应 [-1,1] 归一化）
- Batch size：128
- 优化器：Adam(lr=2e-4, beta1=0.5)
- 判别器/critic 更新：
  - 简易脚本中：WGAN/WGAN-GP 每步 3 次（`GAN/GAN.py`）
  - Notebook/导出脚本：WGAN/WGAN-GP 每步 5 次（`asg6_GroupO.(ipynb|py)`）
- WGAN：权重裁剪 `clip_value=0.01`
- WGAN-GP：梯度惩罚系数 `lambda_gp=10`
- 训练轮数：15（保存 Epoch 0/5/10/14 的可视化与最终报告图）

可在各脚本类 `GANTrainer` 初始化参数中修改上述超参数，并视需要提升 `epochs`。


## 输出与结果查看
脚本会在当前工作目录下创建 `outputs/` 并保存：
- `MODEL_epoch{N}.png`：固定噪声下的 5×5 生成网格（如 `GAN_epoch10.png`、`WGAN_MNIST_epoch5.png`）。
- `MODEL_final_report.png`：包含 G/D 损失曲线与样本“像素标准差”多样性曲线。

本仓库已包含若干参考结果：
- `code/.../GAN/outputs original/`、`outputs fashion/`
- `code/.../WGAN/outputs(mnist|fashion-mnist)/`
- `code/.../WGAN-GP/WGAN-outputs/`

Notebook 末尾还提供了一个 HTML 拼板可视化（按目录读取图片并分组展示）。


## 复现实验建议
- 先在 MNIST 上验证（GAN→WGAN→WGAN-GP）是否能稳定收敛，再切换到 Fashion-MNIST。
- 调整 critic 次数（如 3→5）、`clip_value`（WGAN）与 `lambda_gp`（WGAN-GP），能显著影响稳定性与质量。
- 如果需要更客观的对比，建议加入 FID/Inception Score 等指标；目前项目以可视化与损失曲线为主。


## 注意事项
- Notebook 的 WGAN-GP 某段使用了 Google Drive 保存：
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  OUTPUT_DIR = "/content/drive/MyDrive/WGAN_outputs"
  ```
  本地运行请注释上述挂载语句，并将 `OUTPUT_DIR` 改为本地路径（例如：`OUTPUT_DIR = "./outputs"`）。
- `code/GAN/Fashion GAN.py` 文件名包含空格，命令行运行时需要加引号（README 已示例）。
- 未显式固定随机种子，结果会有轻微随机性。如需更可复现的结果，可在脚本开头设置 `tf.random.set_seed(...)` 与 Python/NumPy 随机种子。


## 背景与参考
- Notebook 中已包含“问题描述/技术描述/结果分析”等文字说明，概括如下：
  - 高层问题：原始 GAN 训练不稳定/模式坍塌；Wasserstein 系列（WGAN/WGAN-GP）改善距离度量与 Lipschitz 约束，获得更稳定的梯度和训练过程。
  - 技术要点：噪声→图像的生成映射；WGAN 使用权重裁剪，WGAN-GP 使用梯度惩罚；在 MNIST 与 Fashion-MNIST 上对比训练曲线与生成样例。
  - 初步结论：视觉上三者均可生成可辨样本；WGAN/WGAN-GP 对稳定性更友好，但对实现与超参较敏感；建议引入多次独立重复与 FID 等指标进行更严谨的量化评估。
- 数据集来自 `tf.keras.datasets`（自动下载并缓存）。


## 致谢
- 感谢CISC7026[smorad, Morad]老师对本项目的认可，并且给予了 100 分的肯定。


## 许可
本项目仅用于课程学习与学术交流，若需二次分发或商用，请先与作者沟通确定许可方式。
