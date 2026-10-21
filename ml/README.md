# House Prices 作业



官方比赛：<https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques>

## GitHub 仓库说明

本仓库上传代码、Notebook、配置和轻量分析结果。以下关于 `data/`、`models/`、`experiments/`、PDF 报告与参考文件的描述对应原始服务器完整工作区；Git 中不包含原始数据、模型、完整逐折检查点和 PDF 文件。运行前请按根目录 README 准备数据，并通过下述命令重新生成训练产物。仓库保留的实验汇总不是可恢复训练的检查点。

Arbor 后续三个实验单独整理在 [`arbor/`](arbor/)，包括源码、评测协议和真实结果；其最佳候选线上 0.11480，未替换当前最优 0.11477。

项目位于服务器 `/mnt/data/bts/repos/homework/ml`。初始提交 0.12333；当前实际最优成绩请查看 `output/selected_submission.json`，不要用本地 CV 替代线上评分。`submission.csv` 保留实际已验证最优提交。

## 文件

- `House_Prices_实验分析.ipynb`：可运行的中文分析说明，附真实实验输出。
- `output/pdf/House_Prices_机器学习实验报告_边通书.pdf`：课程报告，含完整实验附录。
- `experiment.py`：折内填补、变换、编码、缩放，模型训练和逐折检查点。
- `optimize.py`：40 次 Optuna 搜索，SQLite study 与 CSV 搜索轨迹。
- `analyze.py`：EDA、图、受约束集成与权重交叉拟合。
- `finalize.py`：冻结权重复验、完整样本审计、多种子集成和校准诊断。
- `transaction_branch.py`：大面积 Partial 交易分支与按折留出验证。
- `train_final.py`：全量训练、保存模型、重新加载后校验预测。
- `models/`：12 个全量训练推理管线；它们对应 fullfit 候选。官方最优提交可能是折模型版本，以选中记录为准。
- `experiments/`：参数、OOF、预测、逐折耗时、特征重要性、错误日志、Optuna study。
- `output/`：分析 CSV/JSON、图片、每版提交和真实 Kaggle 评分记录。
- `experiments/baseline/`：用户原始训练代码与提交备份。`train.py` 仍是原始基线代码，运行它会写回基线提交。

## 环境

服务器已有 `/home/xzr/bin/python`，Python 3.8；精确依赖见 `requirements-server.txt`。通常不需要新建环境或重复安装。每个实验使用 8 个 CPU 线程，GPU 4、5 是实验当时空闲设备；重跑前检查 `nvidia-smi` 并按空闲情况调整配置中的设备编号。

报告可在 macOS 使用带 `reportlab/pypdf/pandas/Pillow` 的 Python 重建：`python build_report.py`。脚本使用系统中文字体并嵌入 PDF；其他系统可替换字体路径。Notebook 读取同目录已有数据与结果，默认不重新运行全部参数搜索。

## 完整复现

以下命令均在 `ml` 目录执行。已完成的逐折检查点会被复用。重新做独立新运行时应换 `--run` 名称，避免把已有检查点误当成新结果。

```bash
ML_PYTHON=/home/xzr/bin/python
$ML_PYTHON experiment.py --run screen
$ML_PYTHON experiment.py --run tune --config tune_config.json
$ML_PYTHON optimize.py --trials 40
$ML_PYTHON experiment.py --run optuna_final --config optuna_best_config.json
```

注意：Optuna SQLite study 会继续添加试验。若要保持本次结果，不要再次运行搜索；本次最优参数已经保存在 `optuna_best_config.json`。Optuna 种子与剪枝轨迹已保存，GPU 算法可能存在细微数值差异。

`experiments/combined/` 用符号链接指向上述已完成模型，仅用于汇总相同 seed=42 的 OOF 结果。可重新建立链接后计算固定权重：

```bash
$ML_PYTHON - <<'PY'
from pathlib import Path
root = Path('experiments/combined')
root.mkdir(exist_ok=True)
for run in ['screen', 'tune', 'optuna_final']:
    for d in Path('experiments', run).iterdir():
        if not (d / 'metrics.json').exists():
            continue
        link = root / d.name
        if not link.exists():
            link.symlink_to(d.resolve(), target_is_directory=True)
PY
$ML_PYTHON analyze.py --run combined --label optimized
```

保存的 `frozen_config.json` 与 `fixed_iterations.json` 是本次复验选用的四个模型和固定轮数。它们来自选中模型五折早停轮数的中位数。运行复验与最终预测：

```bash
$ML_PYTHON experiment.py --run confirmation --seed 2026 --config frozen_config.json --fixed fixed_iterations.json
$ML_PYTHON experiment.py --run all_rows_audit --seed 73 --config frozen_config.json --fixed fixed_iterations.json --all-rows
$ML_PYTHON finalize.py
$ML_PYTHON transaction_branch.py
$ML_PYTHON train_final.py
```

`finalize.py` 依赖已保存的本次 `optimized.json` 权重；重新优化可能改变权重及后续结果。精确复现已交付提交时，应使用归档配置和既有结果，不改变选择协议。初步 v2 集成的权重和组成另存于 `output/preliminary.json`。

## Kaggle 认证和提交

`kaggle_access.py` 从 `KAGGLE_API_TOKEN` 环境变量读取 Token，不保存凭据。推荐使用交互式隐藏输入，避免把密钥写入命令历史：

```bash
read -s KAGGLE_API_TOKEN
export KAGGLE_API_TOKEN
python kaggle_access.py list
python kaggle_access.py submit-http --file output/submission_transaction_branch.csv --description 'Train-only transaction branch plus multiseed blend'
unset KAGGLE_API_TOKEN
```

服务器可访问 Kaggle API，但本次连接上传存储服务时发生挂起；文件上传在本地网络完成。每次提交后检查 `status=complete` 和实际 `publicScore`；上传成功不等于评分完成。不要重复提交已有候选以浪费每日额度。

## 结果解释

主模型选择 CV 使用删除两条已知大面积低价异常的 1458 行；完整审计保留全部 1460 行参与评分，只在各普通模型训练折排除异常。交易分支用同折训练集中的大面积 Partial 销售拟合，在两条异常记录之间做留出验证，绝不使用被验证行自身标签。分支只有两个支持样本，且是在探索后提出，不能解释为独立确认的普遍交易规则。

权重交叉拟合、多随机划分和多种子 OOF 都是选择偏差或稳定性诊断。基础模型在同一批标签上做过选择，因此这些数字不是无偏嵌套验证，也不是隐藏测试集成绩。官方目标低于 0.1 尚未实现；报告和 Notebook 均保留未达标的真实状态。没有使用外部房价标签、测试集答案或按测试 Id 查价格。
