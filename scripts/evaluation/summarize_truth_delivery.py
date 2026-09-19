"""Package-facing conclusions from completed fixed-case tables and measured timings."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--timing-review', type=Path, required=True)
    a = p.parse_args(); out = a.root / 'tables'
    complete = json.loads((out / 'completion.json').read_text())
    assert complete['completed'] and complete['prefixes'] == 90
    with (out / 'metric_summary.csv').open(encoding='utf-8-sig') as f:
        summary = list(csv.DictReader(f))
    with (out / 'paired_system_comparisons.csv').open(encoding='utf-8-sig') as f:
        paired = list(csv.DictReader(f))
    with (out / 'all_prefix_metrics.csv').open(encoding='utf-8-sig') as f:
        entries = list(csv.DictReader(f))
    means = {(r['tier'], r['method'], r['metric']): float(r['mean']) for r in summary}
    values = {(r['id'], r['tier'], r['method'], r['metric']): float(r['value']) for r in entries}
    metrics = [
        ('Geo：平均RMSD（Å，低优）', 'Geo.mean_rmsd_angstrom'),
        ('Phys v2：受检合法帧比例（高优）', 'Phys_v2_common_heavy.assessed_tail_valid_frame_fraction'),
        ('Dyn：RMSF MAE（Å，低优）', 'Dyn.rmsf_mae_angstrom'),
        ('Dyn：口袋接触Brier（低优）', 'Dyn.contacts.observed_pocket.brier'),
        ('Stab：后段RMSD（Å，低优）', 'Stab.late.rmsd_angstrom'),
        ('单路径Feature ES（低优）', 'Probability.feature_energy_score')]
    methods = ['QField-Dyn', 'Linear', 'NeuralMD']
    lines = ['# QField-Dyn：90个蛋白-配体复合物的轨迹预测结果', '',
        '在90个不同的非训练MISATO验证复合物上评价QField-Dyn，T1/T2/T3各30例。采用主模型第三轮与结构适配器第二轮权重，每例生成一条轨迹。积分精度采用相邻分辨率最终坐标RMS差0.01 Å标准，坐标以实际XTC文件解码结果计分。', '',
        '模型相对Linear的整体位置、运动误差及接触预测改善；相对NeuralMD，运动幅度误差较小，T3受检几何保持较好，位置和接触误差较大。方法优势按具体指标表述。', '',
        '## 完整四维比较', '', '| 指标 | QField-Dyn | Linear | NeuralMD |', '|---|---:|---:|---:|']
    for label, key in metrics:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{means["all", m, key]:.6f}' for m in methods) + ' |')
    lines += ['', '各档位完整表见`results.md`；逐体系数据、误差曲线和全部原始/最终几何诊断见CSV。均值按体系等权，三个档位各30例。', '',
        '## 配对差异与改善范围', '',
        '下表为QField-Dyn减去参照；误差负值较优，合法帧正值较优。95%区间以体系为单位、2,000次bootstrap计算。', '',
        '| 参照 | 指标 | 平均差 | 95%区间 | QField-Dyn较优案例 |', '|---|---|---:|---|---:|']
    for reference in ['Linear', 'NeuralMD']:
        for label, key in metrics:
            r = next(r for r in paired if r['tier'] == 'all' and r['reference'] == reference and r['metric'] == key)
            fraction = float(r['lower_fraction'])
            if key.startswith('Phys_'): fraction = 1 - fraction - float(r['equal_fraction'])
            lines.append(f'| {reference} | {label} | {float(r["mean_difference"]):.6f} | [{float(r["ci95_low"]):.6f}, {float(r["ci95_high"]):.6f}] | {round(fraction * int(r["systems"]))}/{r["systems"]} |')
    lines += ['', '## 物理与结论范围', '',
        f'共同重原子Phys v2下，真值在固定蛋白参考中的受检合法帧比例为{means["all", "QField-Dyn", "Reference_Phys_v2_common_heavy.assessed_tail_valid_frame_fraction"]:.6f}。该指标使用预先训练标定的几何尾部阈值，原严格阈值结果另列。几何代理覆盖键、角、平面/环及碰撞；化学手性检查的实际覆盖和力场能量缺口按原指标记录。', '',
        'NeuralMD采用作者预训练权重，两模型训练条件不同；双方训练名单与本次90例的复合物ID交集为空。Geo/Dyn/Stab/ES和主要Phys比较采用共同配体重原子，QField-Dyn全原子结果另列。QField-Dyn比较包含结构修正。', '',
        '每案例一条轨迹，Feature ES按单实现形式解释。RMSF误差下降反映运动幅度恢复的改善；运动方向和接触时序分别由位置及接触指标评价。本研究结果覆盖本次90个验证复合物。T4属于长时程生成示例，其未来预测准确性缺少对应真值支持。']
    issues = []
    for identifier, tier in sorted({(r['id'], r['tier']) for r in entries}):
        row = dict(id=identifier, tier=tier)
        for label, key in [('common_valid', 'assessed_tail_valid_frame_fraction'), ('environment_valid', 'fixed_environment_valid_frame_fraction')]:
            row[label] = values[identifier, tier, 'QField-Dyn', 'Phys_v2_common_heavy.' + key]
            row['truth_' + label] = values[identifier, tier, 'QField-Dyn', 'Reference_Phys_v2_common_heavy.' + key]
        for key in ['Geo.mean_rmsd_angstrom', 'Dyn.rmsf_mae_angstrom', 'Dyn.contacts.observed_pocket.brier']:
            row[key] = values[identifier, tier, 'QField-Dyn', key]
            for reference in ['Linear', 'NeuralMD']:
                row[key + '_minus_' + reference] = row[key] - values[identifier, tier, reference, key]
        row['contains_assessed_geometry_failure'] = row['common_valid'] < 1
        issues.append(row)
    with (out / 'case_failures_and_differences.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(issues[0])); w.writeheader(); w.writerows(issues)
    lines += ['', f'{sum(r["contains_assessed_geometry_failure"] for r in issues)}/90个案例包含至少一个受检几何异常帧，逐例结果见`case_failures_and_differences.csv`。', '', '## 推理时间和参数量', '']
    timing = json.loads(a.timing_review.read_text()); assert timing['completed']
    lines += ['同型号A800、同一份观测输入的三组实际无缓存完整管线计时如下。每档一次，共享GPU；QField-Dyn输出全配体与固定复合物XTC，NeuralMD输出配体重原子XTC，保留各自数值算法。', '',
        '| 方法 | 档位 | 参数量 | 完整冷启动（秒） |', '|---|---|---:|---:|']
    for row in timing['rows']:
        method = 'QField-Dyn' if row['method'].startswith('QField-Dyn') else 'NeuralMD'
        lines.append(f'| {method} | {row["tier"]} | {row["parameters"]:,} | {row["cold_pipeline_seconds"]:.3f} |')
    (out / 'same_hardware_timing_review.json').write_text(json.dumps(timing, indent=2))
    with (out / 'timing_per_case.csv').open(encoding='utf-8-sig') as f: timings = list(csv.DictReader(f))
    lines += ['', '90例模型生成耗时统计如下：QField-Dyn在A800上运行，NeuralMD在CPU上运行，分别报告实际执行成本。该表生成耗时含必要积分精度检查，重放、评分及文件导出单独执行；cold列为加载、输入准备与生成之和。', '',
        '| 方法/设备 | 档位 | 例数 | 平均生成（秒） | 中位生成（秒） | P95生成（秒） | 平均cold（秒） |', '|---|---|---:|---:|---:|---:|---:|']
    for method, warm, cold in [('QField-Dyn/A800', 'qmem_generation_seconds', 'qmem_cold_generation_seconds'), ('NeuralMD/CPU', 'neuralmd_cpu_generation_seconds', 'neuralmd_cpu_cold_seconds')]:
        for tier in ['T1', 'T2', 'T3']:
            subset = [r for r in timings if r['tier'] == tier]; v = np.array([float(r[warm]) for r in subset])
            lines.append(f'| {method} | {tier} | {len(v)} | {v.mean():.3f} | {np.median(v):.3f} | {np.quantile(v, .95):.3f} | {np.mean([float(r[cold]) for r in subset]):.3f} |')
    (out / '完整结果与交付.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(completed=True, cases=90, geometry_cases_with_anomalies=sum(r['contains_assessed_geometry_failure'] for r in issues))))


if __name__ == '__main__':
    main()
