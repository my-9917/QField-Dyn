# 训练与全流程复现

本目录给出最终模型使用的训练链：量子编码器、基础坐标动力学、量子条件轨迹模型、T1/T2 联合动力学与历史模型、双专家门控，以及 T3 的几何与涨落参数拟合。监督拟合只使用 competition-train；competition-validation 只用于模型选择和结果报告，test future 不进入缓存或训练目标。

代码包内的 `artifacts/` 是正式推理使用的冻结训练产物，可用于精确复现提交推理。下列命令用于从随机初始化复现训练流程；GPU 算子和重新拟合可能产生不同的 checkpoint，因此不承诺逐字节得到同一权重。新训练产物只有在 train-internal 上同时通过 Geo、Phys、Dyn、Stab 与防静态检查后，才可替换冻结模型。

以下命令均从代码包根目录运行，并使用独立虚拟环境，不使用 Conda base。

## 1. 环境和数据

    python -m venv .venv
    .venv/bin/python -m pip install -r requirements-training.txt

公开 MISATO 文件按以下结构放置：

    data/
      MD.hdf5
      QM.hdf5
      topology_train/<complex_id>/production.top.gz
      topology_validation/<complex_id>/production.top.gz

competition-train 和 competition-validation 的 ID 列表位于 splits/train.txt 与 splits/validation.txt。训练链不读取 competition-test 轨迹。

## 2. 构建清单和结构缓存

    .venv/bin/python -m tools.build_manifest --md data/MD.hdf5 --train splits/train.txt --validation splits/validation.txt --output reproduced_data/manifest.csv

    .venv/bin/python -m tools.build_graph_cache --md data/MD.hdf5 --manifest reproduced_data/manifest.csv --topology-root data/topology_train --topology-name production.top.gz --split train --output reproduced_data/graph_train.hdf5
    .venv/bin/python -m tools.build_graph_cache --md data/MD.hdf5 --manifest reproduced_data/manifest.csv --topology-root data/topology_validation --topology-name production.top.gz --split validation --output reproduced_data/graph_validation.hdf5

    .venv/bin/python -m tools.build_coordinate_cache --md data/MD.hdf5 --manifest reproduced_data/manifest.csv --split train --topology-root data/topology_train --output reproduced_data/coordinates_train.hdf5
    .venv/bin/python -m tools.build_coordinate_cache --md data/MD.hdf5 --manifest reproduced_data/manifest.csv --split validation --topology-root data/topology_validation --output reproduced_data/coordinates_validation.hdf5

坐标缓存先消除周期性边界跳变，再把受体对齐到第 1 帧。验证缓存含公开 validation future，只能用于验证；正式测试推理只能读取组委会提供的观测帧。

## 3. 训练量子编码器

    .venv/bin/python -m training.create_quantum_split --graph-cache reproduced_data/graph_train.hdf5 --manifest reproduced_data/manifest.csv --output reproduced_data/quantum_split.csv --summary reproduced_results/quantum_split.json

    .venv/bin/python -m tools.build_quantum_statistics --md data/MD.hdf5 --qm data/QM.hdf5 --graph-cache reproduced_data/graph_train.hdf5 --manifest reproduced_data/manifest.csv --internal-split reproduced_data/quantum_split.csv --output reproduced_data/quantum_statistics.hdf5 --summary reproduced_results/quantum_statistics.json

    .venv/bin/python -m training.train_quantum_encoder --config config/quantum_training.yaml

    .venv/bin/python -m tools.cache_quantum_features --config config/quantum_training.yaml --checkpoint reproduced_artifacts/quantum/quantum_encoder.pt --md data/MD.hdf5 --graph-cache reproduced_data/graph_train.hdf5 --manifest reproduced_data/manifest.csv --split train --output reproduced_data/quantum_features_train.hdf5

    .venv/bin/python -m tools.cache_quantum_features --config config/quantum_training.yaml --checkpoint reproduced_artifacts/quantum/quantum_encoder.pt --md data/MD.hdf5 --graph-cache reproduced_data/graph_validation.hdf5 --manifest reproduced_data/manifest.csv --split validation --output reproduced_data/quantum_features_validation.hdf5

量子编码器从配体结构学习训练集内的原子电荷和分子电子性质监督。轨迹模型只读取编码器由输入结构推断的特征，不按复合物 ID 查询 validation/test 量子标签。

## 4. 训练轨迹模型

    .venv/bin/python -m training.train_base_dynamics --config config/base_pretraining.yaml
    .venv/bin/python -m training.train_base_dynamics --config config/base_training.yaml
    .venv/bin/python -m training.train_trajectory_model --config config/trajectory_training.yaml

基础预训练采用 4 步 rollout；继续训练和量子条件轨迹训练采用 8 步 rollout，并联合优化坐标、键长和键角误差。

T1 使用与双专家门控相同的 competition-train 内部划分，并在冻结轨迹模型上继续两阶段训练：

    .venv/bin/python -m training.create_expert_split --manifest reproduced_data/manifest.csv --output reproduced_data/expert_split.csv
    .venv/bin/python -m training.train_joint_trajectory_model --config config/trajectory_short_stage1.yaml
    .venv/bin/python -m training.train_joint_trajectory_model --config config/trajectory_short_stage2.yaml

T1 使用 10 步 rollout，增加 RMSF、速度分布、接触距离、接触转换、速度时间相关和回转半径目标。第二阶段提高坐标、角度和时间相关权重，产出 reproduced_artifacts/trajectory_model_t1.pt。推理时只转移该模型预测的刚体运动与可旋转键变化，并逐帧拒绝增加配体内部碰撞或蛋白碰撞的修正。

在此基础上训练读取前 10 帧历史的 T1 轨迹模型，并从 competition-train 的 fit 子集拟合 T1 时间相关映射和中等时滞目标：

    .venv/bin/python -m training.train_history_joint_trajectory_model --config config/history_training_t1.yaml
    .venv/bin/python -m tools.fit_t1_multilag_mapping --manifest reproduced_data/manifest.csv --split reproduced_data/expert_split.csv --coordinate-cache reproduced_data/coordinates_train.hdf5 --graph-cache reproduced_data/graph_train.hdf5 --output reproduced_artifacts/velocity_correlation_t1.csv
    .venv/bin/python -m tools.fit_t1_phase_targets --manifest reproduced_data/manifest.csv --partition reproduced_data/expert_split.csv --coordinates reproduced_data/coordinates_train.hdf5 --graphs reproduced_data/graph_train.hdf5 --per-system reproduced_results/t1_phase_targets_per_system.csv --summary reproduced_artifacts/t1_phase_targets.csv

第 2 个 epoch 的历史权重 `reproduced_artifacts/history_t1/history_trajectory_t1_epoch2.pt` 先在 1,307 个 train-internal 体系上通过 Geo、Phys、Dyn、Stab 与防静态联合检查，再进行一次 competition-validation 检查。正式推理只读取前 10 帧：先要求历史候选改善观测映射的 lag-1 相关性且不降低运动强度和接触转换，再恢复原预测质心；最后要求接触距离分布更接近观测段，且 lag-2/4 相关性更接近 fit 子集的目标。选择过程不读取 future。

T2 使用同一组联合目标和 20 步 rollout，并从 competition-train fit 子集拟合观测段到未来段的速度时间相关映射：

    .venv/bin/python -m training.train_joint_trajectory_model --config config/trajectory_medium.yaml
    .venv/bin/python -m tools.fit_velocity_correlation --manifest reproduced_data/manifest.csv --split reproduced_data/expert_split.csv --coordinate-cache reproduced_data/coordinates_train.hdf5 --graph-cache reproduced_data/graph_train.hdf5 --output reproduced_artifacts/velocity_correlation.csv

T2 推理同时运行基础轨迹模型和 T2 联合模型，构造 0、0.25、0.50、0.75、1.0 五个固定幅度。只用已观测 80 帧估计目标时间相关，并选择最接近的候选；不读取真实 future。

在此基础上训练历史条件轨迹模型：

    .venv/bin/python -m training.train_history_joint_trajectory_model --config config/history_training_t2.yaml

三个 epoch 分别保存。模型选择不按训练总损失，而是在 competition-train 的独立 internal_validation 上联合检查 Geo、Phys、Dyn、Stab 与防静态指标。最终使用第 1 个 epoch 的历史增量参数，并恢复已经验证的 T2 骨干：

    .venv/bin/python -m tools.restore_history_backbone --base reproduced_artifacts/trajectory_model_t2.pt --history reproduced_artifacts/history_t2/history_trajectory_t2_epoch1.pt --output reproduced_artifacts/history_trajectory_t2.pt

推理时，历史模型生成第二个候选，经扭转几何投影和不增加碰撞筛选后，只在其速度相关更接近由观测 80 帧估计的目标、且接触转换不低于冻结输出时采用。选择过程不读取 future。

## 5. 训练双专家门控并拟合 T3 参数

    .venv/bin/python -m tools.cache_expert_inputs --config config/trajectory_training.yaml --checkpoint reproduced_artifacts/trajectory_model.pt --layer-cache reproduced_data/quantum_features_train.hdf5 --output reproduced_data/expert_inputs.hdf5

    .venv/bin/python -m training.train_expert_gate --config config/expert_training.yaml

    .venv/bin/python -m training.fit_geometry --q-cache reproduced_data/expert_inputs.hdf5 --ligand-coordinate-cache reproduced_data/coordinates_train.hdf5 --graph-cache reproduced_data/graph_train.hdf5 --split reproduced_data/expert_split.csv --coefficients reproduced_artifacts/geometry_coefficients.csv --metrics reproduced_results/geometry_fit.csv

    .venv/bin/python -m training.fit_displacement_limit --q-cache reproduced_data/expert_inputs.hdf5 --coordinate-cache reproduced_data/coordinates_train.hdf5 --graph-cache reproduced_data/graph_train.hdf5 --internal-split reproduced_data/expert_split.csv --expert-checkpoint reproduced_artifacts/expert_gate.pt --geometry-coefficients reproduced_artifacts/geometry_coefficients.csv --output reproduced_results/displacement_limit.csv

    .venv/bin/python -m training.fit_translation_scale --internal-split reproduced_data/expert_split.csv --ligand-coordinate-cache reproduced_data/coordinates_train.hdf5 --graph-cache reproduced_data/graph_train.hdf5 --output reproduced_results/translation_fit.csv

    .venv/bin/python -m training.fit_rotation_scale --internal-split reproduced_data/expert_split.csv --ligand-coordinate-cache reproduced_data/coordinates_train.hdf5 --graph-cache reproduced_data/graph_train.hdf5 --output reproduced_results/rotation_fit.csv

    .venv/bin/python -m training.assemble_model_config --displacement-metrics reproduced_results/displacement_limit.csv --translation-metrics reproduced_results/translation_fit.csv --rotation-metrics reproduced_results/rotation_fit.csv --max-rms-displacement 0.9950557351 --output reproduced_artifacts/model.json

`0.9950557351 Å` 是正式模型在 train-only 校准后冻结的位移上限。命令行显式保留该部署参数，避免重新拟合时无意改变已验证的推理边界。

## 6. 组装独立推理包

完整部署需要十三项训练产物：quantum_encoder.pt、quantum_statistics.hdf5、trajectory_model.pt、trajectory_model_t1.pt、trajectory_model_t2.pt、history_trajectory_t1.pt、history_trajectory_t2.pt、velocity_correlation_t1.csv、t1_phase_targets.csv、velocity_correlation.csv、expert_gate.pt、geometry_coefficients.csv 和 model.json。

    .venv/bin/python -m tools.assemble_package --output reproduced_package

组装工具会复制最终 T1/T2 权重、历史权重、时间相关统计量以及 T3 所需权重，不覆盖本包提供的参考模型。

## 7. 批量预测和评价

    .venv/bin/python -m tools.predict_dataset --project-directory reproduced_package --manifest reproduced_data/manifest.csv --split validation --graph-cache reproduced_data/graph_validation.hdf5 --coordinate-cache reproduced_data/coordinates_validation.hdf5 --task T1 --output reproduced_results/predictions_t1.hdf5

    .venv/bin/python -m tools.evaluate_dataset --manifest reproduced_data/manifest.csv --dataset-split validation --ligand-coordinate-cache reproduced_data/coordinates_validation.hdf5 --graph-cache reproduced_data/graph_validation.hdf5 --predictions reproduced_results/predictions_t1.hdf5 --prediction-mode stored --model QField-Dyn --task T1 --output reproduced_results/validation_t1.csv --per-complex reproduced_results/validation_t1_per_complex.csv

将 task 改为 T2 或 T3 即可生成对应档次结果。

官方公开评测包的 T1-T4 统一生成命令为：

    .venv/bin/python -m tools.predict_public --data GOAI_eval_public --output reproduced_public_results --tiers T1 T2 T3 T4 --device cuda

该入口按每个体系的 meta.json 读取观测帧数、预测帧数和时间间隔，输出 `{id}_pred.xtc`；T4 使用独立长程生成路径。

成功标准：训练损失和输出坐标均为有限值；量子内部验证、轨迹 validation loss 和专家内部验证完成；T1/T2/T3 分别输出 10/20/80 帧；评价报告同时包含 Geo、Phys、Dyn、Stab 与防静态指标。完成训练不等于可以替换冻结模型，只有联合比较通过才允许进入正式推理。若出现非有限损失，直接停止并保留训练日志，不跳过批次或降低标准。
