# Arbor 实验与前端

此目录导出了本次三个真实实验的源码、评估记录和只读前端，不依赖嵌套 Git 仓库。实验详情见 `SUMMARY.zh-CN.md` 和 `REPORT.md`。最佳新候选 Kaggle 分数为 **0.11480**，未超过此前 **0.11477**，因此没有替换最终提交。配置中的本地指标来自开发交叉验证，没有独立、从未使用过的本地测试集。

## 查看前端

使用 Python 3.10 或更新版本，在此目录执行：

```bash
python3 monitor/serve.py
```

打开终端输出的本地地址，默认是 http://127.0.0.1:8765 。前端读取本目录 `.arbor/sessions/house_prices_keyless_20261013` 下保存的实验树。`monitor/LICENSE` 保留了原项目的 Apache 许可证。

## 重现实验

先按照项目根目录 README 从 Kaggle 下载原始比赛文件到 `ml/data/`，再在本目录执行：

```bash
mkdir -p data
cp ../data/train.csv ../data/test.csv ../data/sample_submission.csv ../data/data_description.txt data/
```

数据必须保持原始字节，评估器会按 `protected_manifest.json` 校验。训练环境依赖见 `../requirements-server.txt`；前端可以使用单独的较新 Python 环境。

`baseline/` 保存原基线的 OOF 和提交结果；根目录 `solution.py` 是返回基线的初始方案。要运行第三个实验，可以先备份根目录 `solution.py`，将 `experiments/3/solution.py` 复制到该位置，再执行 `python evaluate.py --node-id node3`。具体参数以 `python evaluate.py --help` 为准。不要修改受保护评估器来绕过校验。

实验报告及提示中的服务器绝对路径记录了原始运行环境，迁移时以当前目录为准。完整训练检查点与原始数据未纳入 Git，运行评估会重新产生计算结果。
