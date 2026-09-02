# QField-Dyn 可复现推理包

QField-Dyn（Quantum-Information-Enhanced Field Dynamics）根据 `GOAI_eval_public` 中每个体系的观测轨迹生成 T1–T4 未来轨迹。蛋白坐标始终固定，模型只预测配体运动；输出为包含完整体系原子、单位为 nm 的 XTC 文件。

## 科学问题与意义

本项目研究固定蛋白口袋条件下的小分子未来轨迹预测。该任务不仅要求坐标接近参考轨迹，还要求保持键长、键角、立体结构和碰撞合理性，并恢复涨落、速度相关与接触转换等动力学特征。单纯追求平均坐标误差容易产生静态、过度平滑或长程漂移，因此模型选择同时考虑 Geo、Phys、Dyn、Stab 与防静态表现。

## 方法贡献与结果口径

QField-Dyn 使用量子化学监督学习配体电子相关表示，并与固定蛋白口袋和观测轨迹共同编码。T1–T4 共享结构表示、运动分解、几何约束和物理条件，根据观测长度、帧间隔与预测跨度选择相应的时间尺度更新。项目进一步将 RMSF、速度分布、接触距离、接触转换和时间相关纳入训练或候选选择，避免轨迹退化为均值结构。

完整本地代理结果、参照方法和适用边界见 [`RESULTS.md`](RESULTS.md)，原始汇总与逐体系数值位于 [`results/`](results/)，训练数据划分与复现链见 [`TRAINING.md`](TRAINING.md)。这些结果不是官方分数；公开 T4 不含未来真值，因此只报告格式、物理风险和观测统计，不报告未知未来准确率。

## 1. 环境安装

主要安装路径为 Python 3.10 虚拟环境：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-public.txt
```

固定依赖包括 PyTorch 2.5.1+cu121、NumPy 2.2.6、SciPy 1.15.3、h5py 3.15.1、RDKit 2026.3.4 和 MDAnalysis 2.9.0。本组合已在 NVIDIA A800、Python 3.10.21、CUDA 12.1 环境完成全量推理。

## 2. 评测数据

将组委会评测目录原样放在本包根目录：

```text
QField-Dyn/
├── GOAI_eval_public/
│   ├── README.md
│   ├── protocol.json
│   ├── T1/
│   ├── T2/
│   ├── T3/
│   └── T4/
├── README.md
└── run.sh
```

程序只读取 PDB、观测 XTC、`meta.json`、`ids.txt`、`README.md` 和 `protocol.json` 所定义的公开输入，不读取待预测未来帧。

## 3. 方法与关键配置

QField-Dyn 是任务条件的多时间尺度轨迹框架。所有档次共享固定口袋、配体状态、观测轨迹、刚体/内部运动分解、结构几何约束和相对物理条件，再根据观测长度、时间间隔和预测长度调用相应的动力学专家。

- T1：组合量子条件轨迹、联合动力学和历史轨迹候选，根据观测段的速度相关、接触转换和运动强度选择结果。
- T2：组合固定幅度候选与历史候选，利用 80 个观测帧估计时间相关目标。
- T3：使用双专家时域融合、扭转几何约束，以及由 20 个观测帧估计的平移和旋转涨落。
- T4：1000 ps 间隔和 490 步超出已有未来监督范围，使用同一框架中的观测历史随机动力学专家，重采样成对平移/旋转创新，并加入质心均值回归、刚体碰撞约束和 XTC 写回后的极端重叠修正。

量子信息增强来自训练侧 GFN2-xTB 水环境原子电荷、电子亲和能和化学硬度监督；推理阶段由输入结构生成电子相关表示，不查询评测体系量子标签。关键配置位于 `config/model.json`。T1–T3 种子为 20260816，T4 种子为 20260825。

## 4. 权重与配置

推理所需的 13 项模型权重、统计量和配置全部位于包内 `artifacts/` 与 `config/model.json`，无需联网下载。权重由 MISATO competition-train 训练或训练侧拟合得到。

## 5. 一键推理

放置评测数据并完成环境安装后，仅执行：

```bash
bash run.sh
```

脚本不接收参数，也不要求人工交互。流程依次为：环境与文件检查 → 数据读取 → 预处理 → 模型加载 → 推理 → 几何与碰撞处理 → XTC 写出 → 格式核对。入口在启动 Python 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，保证 CUDA 确定性路径可运行。

## 6. 输出

成功后得到：

```text
GOAI_pred_TEAMID/
├── T1/   # 30 条，每条 10 帧
├── T2/   # 30 条，每条 20 帧
├── T3/   # 30 条，每条 80 帧
└── T4/   # 5 条，每条 490 帧
```

验证摘要写入根目录 `reproduction_verification.json`。输出 XTC 只包含未来 `n_pred` 帧，原子数、顺序、时间与盒信息遵循评测包定义。

## 7. 运行检查

`run.sh` 在推理前检查 Python 环境、CUDA、评测输入和 13 项模型产物；推理后检查 95 条文件的档次、命名、帧数、原子数、有限坐标、时间、盒子以及非配体原子固定性。

## 8. 硬件与耗时

远程 NVIDIA A800 全量实测约 2 分钟生成并核对 95 个体系。进程峰值内存约 1.54 GB，PyTorch 峰值分配/保留显存约 39.0/52.4 MB；建议至少提供 1 个 CPU 核、4 GB 可用内存和 2 GB 可用显存。该建议不是多型号硬件上测得的最低边界。

## 9. 训练数据、训练代码与外部资源

监督训练只使用赛事规定的 MISATO competition-train 13,066 个复合物及其训练侧量子化学信息。competition-validation 只用于模型选择和本地结果报告；不使用 competition-test、独立核验体系未来轨迹或评测体系量子标签。训练代码位于 `training/`，训练依赖位于 `requirements-training.txt`。

MISATO 数据：Zenodo record 7711953，version 1.0.0，DOI `10.5281/zenodo.7711953`。MISATO GitHub 源码仓库标注 LGPL-2.1；Zenodo 数据记录的 Rights/License 字段未显示具体许可证，数据使用与再分发遵循 Zenodo 记录及赛事要求。本项目未使用外部势函数、外部基础模型或 MISATO 以外的外部 MD 数据。

项目仓库：`https://github.com/my-9917/QField-Dyn`  
源码许可证：`Apache-2.0`  
固定版本：`goai-finals-2026` tag  
第三方依赖与数据授权：见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

仓库不包含 MISATO 原始轨迹、官方评测输入、未来轨迹或由 MISATO 坐标导出的样例文件。NeuralMD 仅作为任务相关文献参考，未复用其代码、权重或数据。本项目未使用外部商业 API、外部势函数、外部基础模型或 MISATO 以外的外部 MD 数据。

## 10. 已知限制与常见问题

- 公开 T4 不含未来真值。当前结果只能证明输出格式有效、明确物理错误消失且观测统计未明显退化，不能证明未知未来上的 Geo、Dyn、Stab 或总分一定提高。
- 0.4 Å 是极端固定环境重叠的局部触发阈值，不是官方 Phys 评分公式。
- T4 已作为完整加分档提交，因此复现流程会生成全部 5 条轨迹。
- XTC 使用 nm；PDB 与模型内部单位转换由 `tools/public_io.py` 完成，不需要人工转换。
- 公开 T4 的 PDB 元素列为空时，MDAnalysis 可能给出元素猜测警告，不影响原子顺序、坐标或输出核对。
- 若 `GOAI_pred_TEAMID` 已存在，请在干净副本中重新运行，避免混合新旧结果。
