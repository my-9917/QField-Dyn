# 输入与推理

先运行`python tools/build_runtime.py --output runtime`。所有命令从仓库根目录执行。

## 输入目录

```text
observations/
  protocol.json
  T1/
    manifest.jsonl
    example/
      meta.json
      top.pdb
      obs.xtc
```

`protocol.json`可只包含需要运行的档位：

```json
{"tiers":{"T1":{"n_obs":10,"n_pred":10,"dt_ps":80,"ligand_resname":"MOL","n_systems":1}}}
```

`T1/manifest.jsonl`每行一个案例：

```json
{"id":"example","top":"T1/example/top.pdb","obs":"T1/example/obs.xtc","n_obs":10,"n_pred":10,"dt_ps":80,"ligand_resname":"MOL"}
```

`meta.json`包含相同任务字段以及`id`、`tier`、`n_atoms`、`obs_index_0based`和`pred_index_0based`。T1两个区间为`[0,10]`和`[10,20]`，右端点不包含；`n_atoms`等于PDB总原子数。

PDB与XTC原子顺序一致，元素和配体CONECT连接完整。观测从0 ps开始，时间间隔匹配所选档位。输入包括蛋白、配体及支持的单原子离子；水在上游准备阶段处理。未来坐标不作为输入。

## 命令

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh /absolute/path/observations outputs/example
```

输出按`T1/<id>_pred.xtc`至`T4/<id>_pred.xtc`组织，生成证据保存在`outputs/reproduction/`，检查汇总为`reproduction_verification.json`。`--verify-replay`会额外执行一次相同随机种子的完整生成。完整生成时间与重放时间分别记录。

赛事包支持无参数运行`bash run.sh`：默认输入为`GOAI_eval_public/`，默认输出为相邻的`../GOAI_pred_xxxxxm429/`。默认Python为`.venv/bin/python`，可通过`PYTHON`环境变量指定环境。

只运行某个案例可直接使用底层入口：

```bash
CUDA_VISIBLE_DEVICES=0 python runtime/predict_trajectory_adapter.py \
  --base models/E3.pt --adapter models/epoch_02.pt \
  --geometry-calibration configs/ligand_geometry_calibration_v1.json \
  --t4-geometry-calibration configs/phys_calibration.json \
  --public-root /absolute/path/observations --case example \
  --output outputs/example --seed 2026091101 --verify-replay
```

90例有真值实验使用单独登记的评价种子`2026091407`，见`results/truth90/manifest.json`。改变种子属于不同的随机生成样本。

## 权重

| 文件 | SHA256 |
|---|---|
| `models/E3.pt` | `cc88c35d54510f5a6e19321ec87d374f7b7b00bbc03f02b0d8a6850ac25963f7` |
| `models/epoch_02.pt` | `cbd20f667e18c1125c880d89e6554638c7c40dadd1680b24539d31f3d66e5287` |

检查点包含模型spec，加载时核对适配器绑定的主模型身份。使用受信任的本项目检查点。

公开推理检查点移除了训练调度与本机路径；全部模型张量逐值保持一致。原始与导出文件的SHA256对应关系见models/provenance.json。

公开源码与推理检查点在A800上完成真实T1案例验证：10帧坐标、时间戳与原版本逐元素一致，精确重放通过；64/128步最终坐标RMS差为0.00079456 Å。该检查验证源码整理和检查点导出后的计算一致性，预测质量由90案例指标表评估。
