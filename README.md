# ML2026

Kaggle House Prices 房价回归作业，以及使用 Arbor 免 Key 工作流进行的后续实验。

## 项目

- [`ml/`](ml/)：数据探索、折内特征工程、模型比较、Optuna 搜索、残差集成和提交工具。
- [`ml/House_Prices_实验分析.ipynb`](ml/House_Prices_实验分析.ipynb)：中文实验分析与保存的真实输出。
- [`ml/arbor/`](ml/arbor/)：隔离评测、三个候选模型、实验记录与官方只读前端。
- [`ml/output/`](ml/output/)：轻量分析结果、图表、历史提交及线上评分记录。

已核验最佳 Kaggle Public Score 为 **0.11477**，初始分数为 **0.12333**。Arbor 第三个实验本地略有改善，但线上为 **0.11480**，因此没有替换最佳提交。目标低于 0.1 尚未实现。

## 数据与运行

原始比赛数据、训练模型、完整检查点、临时文件和交付压缩包不纳入 Git。请从[官方比赛数据页](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)下载 `train.csv`、`test.csv`、`sample_submission.csv` 和 `data_description.txt`，放入 `ml/data/`。

训练环境的精确版本见 [`ml/requirements-server.txt`](ml/requirements-server.txt)。原实验使用 Python 3.8；GPU 配置需根据本机资源调整。CPU 环境可修改实验配置中的 XGBoost/CatBoost 设备选项。

```bash
cd ml
python -m pip install -r requirements-server.txt
python experiment.py --run screen
```

完整训练、调参与复现步骤见 [`ml/README.md`](ml/README.md)。完整模型与逐折预测需要重新训练生成；仓库只保留实验汇总，避免误用不完整的模型检查点。重建报告另需 `reportlab`、`pypdf`、`Pillow`，Notebook 另需 IPython/Jupyter。

Kaggle 认证由 `KAGGLE_API_TOKEN` 环境变量提供。仓库不包含访问凭据。
