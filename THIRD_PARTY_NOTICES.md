# 第三方软件、数据与模型

本仓库源码沿用Apache License 2.0。依赖通过各自发行渠道安装，其许可证与随附声明以所安装版本为准。

主要依赖包括PyTorch、NumPy、SciPy、h5py、RDKit、MDAnalysis、Matplotlib与Pillow。依赖版本见`requirements.txt`及`requirements-lock.txt`。MDAnalysis用于拓扑与轨迹I/O；其上游许可证独立于本仓库代码许可证。可选化学输入诊断`observed_chemistry.py`需要Open Babel，主推理链不调用该诊断。

## 数据

训练及90案例验证使用MISATO，数据来源：https://doi.org/10.5281/zenodo.7711953 。仓库发布划分说明、验证成员ID与派生统计，原始MD轨迹、AMBER拓扑、比赛匿名输入和未来真值从授权来源获取。本模型权重和训练统计由本项目训练产生，使用时同时遵守赛事与上游数据条款。

## NeuralMD

NeuralMD作者仓库：https://github.com/chao1224/NeuralMD 。本次对比使用其公开预训练ODE seed22权重，源代码commit `a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe`，权重SHA256见`docs/evaluation.md`。

本仓库的`scripts/baselines/`提供输入转换、推理调用和共同指标评价；NeuralMD模型实现、作者权重及其依赖从上游获取。双方训练条件分别披露，本次比较未按同数据同训练轮数重新训练NeuralMD。

## 软件执行范围

QField-Dyn量子状态由经典计算机上的PyTorch复数张量模拟。发布结果不包含真实量子设备或有限shots训练。预测只读取输入观测与拓扑；未来真值用于独立的评分步骤。
