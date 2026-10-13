# NGBoost：AudioMNIST 噪声鲁棒性实验

本仓库从 [audiomnist-noise-robustness](https://github.com/wengyinz1232-create/audiomnist-noise-robustness) 提取 NGBoost 实验代码、运行依赖、数据清单及两次已完成实验的结果。来源提交：`3b6528bd7da32c772a63d4c7df449cd0faf67ecb`。

本次迁移保留原训练与选参逻辑，没有重新训练或改变超参数。实验是 **10 类数字语音分类**：MFCC+delta 特征、干净语音与 10 dB 白噪声增强、外层 10 折和内层 3 折说话者隔离验证，同时保留 LightGBM 配对基线及其公共依赖。

## 文件与数据

| 路径 | 内容 |
|---|---|
| `compare_ngboost.py` | NGBoost 建模、内层轮数选择、逐折训练与结果核验 |
| `report_ngboost.py` | 从已有结果生成中文报告和图表 |
| `reproduce.py` | 准备独立运行目录、下载原始数据及提取特征 |
| `AudioMNIST_完整版/` | 原训练公共模块、依赖、数据清单和上游许可证 |
| `AudioMNIST_完整版/data/` | 12,000 条样本清单、标签、说话者信息、特征名称与数据哈希 |
| `reproduced_results/ngboost-38150310147/` | 原 50/100/200 轮实验的预测、逐折结果和报告 |
| `reproduced_results/ngboost-38332286883/` | 扩展 200/400/800 轮实验的预测、逐折结果和报告 |
| `.github/workflows/ngboost.yml` | 手动启动的完整 NGBoost 配对实验工作流 |
| `docs/NGBOOST_EXPERIMENT.md` | 原实验协议、来源和详细说明 |

原仓库没有提交原始音频 `.wav` / ZIP、训练特征缓存或拟合模型 `.pkl`，本仓库也不包含这些文件。已复制的 `.npz` 是逐折预测证据，不是训练特征缓存。原音频来自 [AudioMNIST](https://github.com/soerenab/AudioMNIST)，固定版本 `630d7dab4c040882834de6fa21baf9a60372accd`；运行准备命令会下载约 1 GB 原始数据并重建特征。数据许可见 `AudioMNIST_完整版/data_raw/SOURCE_LICENSE`。

## 安装和检查

使用 Python 3.12，在仓库根目录执行：

```bash
python -m pip install -r requirements-ngboost.txt
python -m unittest discover -s tests -v
python reproduce.py self-test
```

## 运行扩展实验

```bash
python reproduce.py prepare --work-dir .ngboost-extended800
python compare_ngboost.py plan --protocol extended800
python compare_ngboost.py fold --work-dir .ngboost-extended800 --fold 1 --protocol extended800
```

将 `--fold` 依次设置为 2 至 10，完成全部十折后：

```bash
python compare_ngboost.py summarize --work-dir .ngboost-extended800 --protocol extended800
```

也可在本仓库 Actions 中选择 **NGBoost paired comparison → Run workflow**，协议选择 `extended800`。工作流只支持手动启动，上传代码不会自动启动完整训练。

历史实验使用 `--protocol legacy200`，请使用独立运行目录。两个协议分别只在 `[50, 100, 200]` 和 `[200, 400, 800]` 中选择提升轮数；学习率固定为 0.03，树深度固定为 3，最小叶子样本数为 20，行列采样比例均为 0.5。

## 查看已有结果

- [扩展 200/400/800 实验中文报告](reproduced_results/ngboost-38332286883/NGBOOST_RESULTS_ZH.md)
- [历史 50/100/200 实验中文报告](reproduced_results/ngboost-38150310147/NGBOOST_RESULTS_ZH.md)
- [完整实验说明](docs/NGBOOST_EXPERIMENT.md)

两次结果均为原仓库历史实验，不能视为在本仓库重新训练的结果。历史实验十折均选择 200，扩展实验十折均选择 800，均命中候选上界；这些结果不能表述为已找到全局最优参数。报告中的运行链接保留原仓库地址以追溯来源。

迁移文件和 SHA-256 校验值记录于 `MIGRATION_MANIFEST.json`。
