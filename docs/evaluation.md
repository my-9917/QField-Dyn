# 评价协议

已报告结果来自90个不同的非训练MISATO复合物，T1/T2/T3各30个，每案例生成一条轨迹。成员、训练ID排除、种子和权重见[manifest](../results/truth90/manifest.json)。划分按复合物ID去重；蛋白家族和配体骨架独立性尚未量化。

| 维度 | 主要实现 |
|---|---|
| Geo | 配体重原子逐帧RMSD、均值、末帧和时间曲线；坐标固定于共同蛋白参考 |
| Phys | 键长、键角、自碰撞、环境近接触、手性相关几何，报告受检合法帧比例与严重程度 |
| Dyn | 原子RMSF、接触Brier、回转半径、接触和运动分布、位移与时间相关误差 |
| Stab | 早中晚分段误差、运动幅度和漂移；T4单独记录长程生成的数值与几何表现 |
| 概率评分 | Feature ES、coordinate ES等；单路径使用确定性形式，多路径校准须另行评价 |

`src/evaluation/`保留逐项实现。Phys v2采用训练几何分布的尾部标定，不等同于力场能量，也不把正常热涨落直接解释为断键。主要方法对比使用共同配体重原子，全原子诊断另列。

正式参照为最后5帧最小二乘速度的Linear外推及NeuralMD作者预训练ODE。确定性基线按单路径评分。NeuralMD与QField-Dyn训练条件不同，此实验属于各自训练方案的应用比较。

`all_prefix_metrics.csv`保存完整逐案例指标；`metric_summary.csv`汇总分档均值；`paired_system_comparisons.csv`提供体系配对差异；`case_failures_and_differences.csv`保留失败与不利结果。置信区间以复合物为单位、2,000次bootstrap计算。

## 从授权数据重新评价

先以`prepare_data.py`建立原生坐标缓存，再按发布manifest的成员与档位建立本机执行manifest。以下入口核对验证归属并绑定本机缓存和公开推理权重；发布表格的原始权重身份保留在`original_model`。

```bash
python tools/prepare_evaluation_manifest.py --cache /path/to/prepared_cache \
  --output outputs/execution_manifest.json
cp configs/phys_calibration.json models/phys_calibration.json
```

```bash
CUDA_VISIBLE_DEVICES=0 python runtime/evaluate_truth_cases.py \
  --root models --manifest outputs/execution_manifest.json \
  --statistics configs/feature_statistics_v2.json \
  --output outputs/evaluation --shard 0 --shards 1
```

该入口的`--root`需要同时包含`E3.pt`、`epoch_02.pt`和`phys_calibration.json`。生成阶段仅接收观测字段，评分阶段读取未来标签。

## NeuralMD

作者仓库：https://github.com/chao1224/NeuralMD ，使用commit `a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe`。按其文档获取ODE seed22权重与训练名单；本仓库提供迁移适配代码，保持原Euler规则与时间缩放。

```text
external/neuralmd/
  vendor/NeuralMD-a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe/
  assets/model.pth
  assets/train_MD.txt
```

设置`QFIELD_NEURALMD_ASSETS`可使用其他目录。权重SHA256为`4f35d8fea9a8f38e4f2fb576cf74279a0359a7b614f926b1d453937d7a548351`。依赖安装遵循该commit的作者说明，包括其几何图模型依赖和`torchdiffeq`。

```bash
python runtime/evaluate_neuralmd.py \
  --case-manifest /path/to/execution_manifest.json \
  --statistics configs/feature_statistics_v2.json \
  --calibration configs/ligand_geometry_calibration_v1.json \
  --output outputs/neuralmd --device cuda --paths 1
```

## 参数与推理时间

报告参数数量和同型号A800、同输入的完整无缓存推理。模型加载、预处理、生成和文件写出计入冷启动延迟，精确重放与评分另计。QField-Dyn生成全配体原子且进行积分精度控制；NeuralMD生成重原子并使用作者积分设置。两者算法成本与原子范围分别披露，当前结果不支持QField-Dyn速度优势。

90例新增验证与原64体系多路径模型选择/128体系独立检验属于不同实验设计；本tag的结果表对应90例实验。T4缺少匹配的490 ns未来真值，报告范围为生成运动和数值/几何表现。
