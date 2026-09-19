# 训练与检查点

## 已执行日程

主模型沿9,846体系CFM预训练、512体系适应、2,048体系三轮production继续训练。每轮production包含6,144个CFM前缀、1,536次更新；其中128个轮换体系的384个前缀采用稀疏32步Heun双路径rollout。三轮学习率为`1e-4 / 1e-4 / 3e-5`，按预设三轮结束，`converged=false`。

冻结主模型后，128个训练体系上的几何适配器完成四轮、384次更新。此tag采用主模型第三轮与适配器第二轮的实测权重。实际损失及优化统计见[训练记录](../results/training_records/README.md)。

CFM是基础训练项；实际rollout的几何、运动和坐标ES监督通过共享编码器、可微量子读出与轨迹流反向传播。读出包含36个可训练角，储层内部固定。适配器训练冻结主模型，优化最终坐标损失与重建图中的参数。

## 入口与复现条件

数据和训练代码随本仓库提供，MISATO原始轨迹与AMBER拓扑由数据来源获取。模型文件可直接用于推理；完整训练复现还需要各阶段起点检查点、资格化数据名单、训练侧标定及batch计划。发布的推理权重不构成完整的历史优化器快照链。

| 入口 | 输入与作用 |
|---|---|
| `prepare_data.py` | 原生MD HDF5、拓扑、资格名单和训练标定，建立T1/T2/T3缓存 |
| `prepare_core_training.py` | 合格训练人口、512体系已有缓存，构造2048体系及rollout子集 |
| `prepare_rotating_epoch.py` | 训练人口、检查点和训练标定，生成逐轮batch与学习率计划 |
| `train_semiflexible.py` | 配置、缓存、起点或恢复检查点，执行分布式主模型训练 |
| `cache_adapter_proposals.py` | 适配训练manifest与冻结主模型，生成带种子的训练proposal |
| `train_trajectory_adapter.py` | proposal、训练几何、统计量与工程检查，执行适配训练 |

配置中使用本机的明确数据路径。`prepare_data.py`需要`eligibility`、`md`、`calibration`、`systems`、`membership_seed`和`workers`字段。资格JSON包含`passed`、`failures`、`eligible`；每个成员有`id`、`partition`和`topology`。

主模型命令形式：

```bash
torchrun --standalone --nproc_per_node=2 runtime/train_semiflexible.py \
  --config /path/to/training_config.json --cache /path/to/prepared_cache \
  --initial /path/to/stage_initial.pt --training-plan /path/to/epoch_plan.json \
  --output outputs/training_epoch
```

`--resume`恢复参数、优化器、随机流和训练进度；恢复时必须采用对应配置与缓存身份。实际训练配置和训练计划由保存的检查点及运行材料确定。训练入口保留完整epoch与阶段快照，临时文件写入完成后再替换检查点。

适配训练的梯度/结构检查由`verify_covalent_reconstruction.py`、`verify_trajectory_adapter.py`及真实prefix检查完成。重新训练使用新的输出目录，保留该tag的权重与结果作为固定参照。
