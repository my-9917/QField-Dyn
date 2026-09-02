# 第三方依赖、数据与授权披露

QField-Dyn 源代码以 Apache License 2.0 发布。下列第三方软件通过 Python 依赖安装，不在本仓库中复制其源代码；使用者应同时遵守各项目许可证及其随附第三方声明。

| 依赖 | 固定版本 | 许可证 | 来源 |
|---|---:|---|---|
| PyTorch | 2.5.1+cu121 | BSD-3-Clause；包含上游第三方声明 | https://github.com/pytorch/pytorch |
| NumPy | 2.2.6 | BSD-3-Clause；包含上游兼容组件 | https://github.com/numpy/numpy |
| SciPy | 1.15.3 | BSD-3-Clause；包含上游兼容组件 | https://github.com/scipy/scipy |
| h5py | 3.15.1 | BSD-3-Clause | https://github.com/h5py/h5py |
| RDKit | 2026.3.4 | BSD-3-Clause | https://github.com/rdkit/rdkit |
| MDAnalysis | 2.9.0 | LGPL-3.0-or-later；开发者贡献通常为 LGPL-2.1-or-later | https://github.com/MDAnalysis/mdanalysis |
| PyYAML | 6.0.3 | MIT | https://github.com/yaml/pyyaml |
| NetworkX | 3.4.2 | BSD-3-Clause | https://github.com/networkx/networkx |
| PyTorch Geometric | 2.8.0.post1 | MIT | https://github.com/pyg-team/pytorch_geometric |

## 数据与模型

- 训练数据：MISATO version 1.0.0，Zenodo record 7711953，DOI `10.5281/zenodo.7711953`。
- 数据范围：只使用赛事规定的 competition-train 13,066 个复合物及其训练侧量子化学信息；competition-validation 只用于模型选择和本地结果报告。
- MISATO 源码：上游 GitHub 仓库标注 LGPL-2.1。
- MISATO 数据：Zenodo 记录的 Rights/License 字段未给出具体许可证。本仓库不再分发 MISATO 原始轨迹、官方评测输入、未来轨迹或由 MISATO 坐标导出的样例文件。
- 模型权重与统计量：`artifacts/` 中的文件由 competition-train 训练或训练侧拟合得到，仅用于作品检查、核验和复现；其使用仍受赛事规则及上游数据权利约束。

## 已有项目与外部资源

- NeuralMD 仅作为任务相关文献参考；本项目未复制或改写其代码，未使用其权重、势函数或训练数据。
- 本项目未使用外部商业 API、外部势函数、外部基础模型或 MISATO 以外的外部 MD 数据。
- 本仓库未包含 GOAI 匿名评测体系的未来轨迹，也不通过体系标识、序列或结构检索能够还原评测答案的外部数据。
