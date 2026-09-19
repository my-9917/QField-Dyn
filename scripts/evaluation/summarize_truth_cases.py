"""Complete paired tables from the frozen 90-case manifest, including every case."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from evaluate_neuralmd import heavy_record
from evaluate_naive_baselines import predict
from experiment_tables import tables,write_csv
from physical_quality import score_quality
from projection_encoding import encode_coordinates
from semiflexible_scores import physics_scores


def main():
    p=argparse.ArgumentParser()
    for key in ('root','geometry-calibration','phys-calibration'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if a.wait:
        while not all((a.root/f'qmem/review_{i}.json').exists() for i in range(4)):
            execution=json.loads((a.root/'execution.json').read_text())
            assert 'finished_utc' not in execution or execution['completed'],execution
            time.sleep(10)
    manifest=json.loads((a.root/'manifest.json').read_text());tasks=manifest['tasks']
    assert len(tasks)==90 and manifest['cases_by_tier']==dict(T1=30,T2=30,T3=30)
    reports=[json.loads((a.root/f'qmem/review_{i}.json').read_text()) for i in range(4)]
    assert all(r['completed'] for r in reports)
    neural=json.loads((a.root/'neuralmd/review_0.json').read_text());assert neural['completed'] and len(neural['rows'])==90
    calibration=json.loads(a.geometry_calibration.read_text());phys_calibration=json.loads(a.phys_calibration.read_text())
    rows=[];timing=[];status=[];curves=[]
    for task in tasks:
        artifact=torch.load(a.root/'qmem'/(task['key']+'.pt'),map_location='cpu',weights_only=False)
        assert artifact['protocol']==manifest
        other=torch.load(a.root/'neuralmd'/(task['key']+'.pt'),map_location='cpu',weights_only=False)
        record=torch.load(task['record'],map_location='cpu',weights_only=False)
        row=artifact['row'];assert row['exact_replay_passed'] and row['numerical_resolution_passed']
        paths=np.asarray(artifact['encoded_paths']);heavy=np.asarray(record['inputs']['ligand_graph']['atomic_numbers'])>1
        _,er=encode_coordinates(paths,record);common=heavy_record(er,heavy,calibration)
        linear,_=encode_coordinates(predict(record['inputs']['X_obs'],80,task['n_pred'])[0]['Linear'][None],record)
        coordinates={'QField-Dyn':paths[:,:,heavy],'Linear':linear[:,:,heavy],'NeuralMD':other['coordinates'][None]}
        assert np.array_equal(np.flatnonzero(heavy),other['heavy_indices'])
        row['baselines']['NeuralMD']=other['row']['metrics']
        # Preserve full-atom physical diagnostics in QField-Dyn/Linear; compare all methods on heavy atoms separately.
        row['baselines']['NeuralMD'].pop('Phys');row['baselines']['NeuralMD'].pop('Reference_Phys')
        for method,x in coordinates.items():
            metric=row['metrics'] if method=='QField-Dyn' else row['baselines'][method]
            metric['Phys_common_heavy']=physics_scores(x,common['geometry'])[0]
            metric['Reference_Phys_common_heavy']=physics_scores(np.asarray(common['X_future'])[None],common['geometry'])[0]
            metric['Phys_v2_common_heavy']=score_quality(x,common,phys_calibration,reference=np.asarray(common['X_future']))
            metric['Reference_Phys_v2_common_heavy']=score_quality(np.asarray(common['X_future'])[None],common,phys_calibration)
            err=np.sqrt(np.square(x[0]-np.asarray(common['X_future'])).sum(-1).mean(-1))
            curves.append(dict(id=task['id'],tier=task['tier'],method=method,lead_times_ps=(np.arange(1,len(err)+1)*80).tolist(),rmsd_angstrom=err.tolist()))
        raw=np.asarray(artifact['sampling']['raw_ligand'])[None]
        row['metrics']['Raw_Phys']=physics_scores(raw,record['geometry'])[0]
        row['metrics']['Raw_Phys_v2']=score_quality(raw,record,phys_calibration)
        row['metrics']['Reference_Phys_v2']=row['reference_phys_v2']
        timing.append(dict(id=task['id'],tier=task['tier'],frames=task['n_pred'],
            qmem_generation_seconds=row['timing']['model_generation_seconds'],
            qmem_cold_generation_seconds=row['timing']['cold_generation_seconds'],
            qmem_replay_seconds=row['timing']['replay_seconds'],
            neuralmd_cpu_generation_seconds=other['row']['complete_model_inference_seconds'],
            neuralmd_cpu_cold_seconds=other['row']['cold_start_inference_seconds']))
        status.append(dict(id=task['id'],tier=task['tier'],finite=row['finite'],replay=row['exact_replay_passed'],
            resolution=row['numerical_resolution_passed'],integration_steps=artifact['sampling']['integration_steps'],
            final_valid_fraction=row['phys_v2']['assessed_tail_valid_frame_fraction'],
            reference_valid_fraction=row['reference_phys_v2']['assessed_tail_valid_frame_fraction']))
        rows.append(row)
    output=a.root/'tables';report=tables(rows,output,'truth90_validation')
    write_csv(output/'timing_per_case.csv',timing);write_csv(output/'case_quality.csv',status)
    (output/'frame_curves.json').write_text(json.dumps(curves))
    (output/'complete_rows.json').write_text(json.dumps(rows,allow_nan=False))
    (output/'evaluation_protocol.json').write_text(json.dumps(manifest,indent=2))
    (output/'comparison_protocol.md').write_text('''# 有真值90案例评价口径

90个不同的MISATO验证复合物，T1/T2/T3各30个；模型及NeuralMD作者训练名单交集为空。每个案例一条固定种子轨迹，全部成员在预测前固定。模型为已完成训练的E3+Adapter E2主候选，数值精度与精确重放逐例验收。原64体系模型选择和128独立检验的完成状态另行记录。

正式参照为Linear最后5帧和NeuralMD作者预训练模型。双方训练条件不同，本表报告迁移应用表现。Geo、Dyn、Stab和ES采用共同配体重原子；Phys_common_heavy与Phys_v2_common_heavy分别报告原阈值和训练标定尾部阈值下的共同重原子几何指标。QField-Dyn/Linear全原子物理诊断另列，NeuralMD对应全原子项记NA。Phys v2属于受检几何代理指标，化学身份覆盖与力场能量范围按原报告保留。比较包含本模型的结构修正。真实未来仅进入评价。

单条路径的ES按确定性形式解释，分布覆盖率留给多路径实验。所有体系及失败按固定名单报告。误差和违例越低越好，合法帧比例越高越好，运动幅度必须结合真值解释。CPU NeuralMD与GPU QField-Dyn逐例时间分别报告，同硬件速度比较见已完成的独立测速表。

T4展示490帧长时程生成轨迹；未来准确性等待合格真值，当前T1–T3结果按80 ps帧间隔报告。
''',encoding='utf-8')
    report['numerical_and_replay_passed']=all(r['replay'] and r['resolution'] for r in status)
    report['methods']=['QField-Dyn','Linear','NeuralMD'];report['T4']='motion demonstration; future accuracy awaits long-trajectory truth'
    (output/'completion.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
