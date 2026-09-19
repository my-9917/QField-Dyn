"""Export all scalar metrics and paired system comparisons as CSV and Markdown."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from compare_generation import flatten_metrics


KEYS = {
    'Geo_RMSD_A': 'Geo.mean_rmsd_angstrom',
    'Geo_coordinate_ES_A': 'Geo.coordinate_energy_score_angstrom',
    'Phys_v1_valid': 'Phys.assessed_geometry_valid',
    'Phys_v2_valid': 'Phys_v2.assessed_tail_valid_frame_fraction',
    'Phys_bond_count': 'Phys.bond_violations',
    'Phys_angle_count': 'Phys.angle_violations',
    'Phys_environment_count': 'Phys.ligand_environment_overlap_violations',
    'Phys_common_heavy_valid': 'Phys_common_heavy.assessed_geometry_valid',
    'Phys_common_heavy_environment_count': 'Phys_common_heavy.ligand_environment_overlap_violations',
    'Phys_v2_common_heavy_valid': 'Phys_v2_common_heavy.assessed_tail_valid_frame_fraction',
    'Phys_v2_common_heavy_environment_valid': 'Phys_v2_common_heavy.fixed_environment_valid_frame_fraction',
    'Dyn_RMSF_MAE_A': 'Dyn.rmsf_mae_angstrom',
    'Dyn_predicted_RMSF_A': 'Dyn.predicted_mean_atom_rmsf_angstrom',
    'Dyn_contact_Brier': 'Dyn.contacts.observed_pocket.brier',
    'Stab_late_RMSD_A': 'Stab.late.rmsd_angstrom',
    'Feature_ES': 'Probability.feature_energy_score',
}


def write_csv(path, rows):
    assert rows
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def tables(rows, output, split):
    output.mkdir(parents=True,exist_ok=True)
    assert len({(r['id'],r['tier']) for r in rows}) == len(rows)
    ids=sorted({r['id'] for r in rows})
    references=[name for name in ('Linear','NeuralMD') if all(name in r['baselines'] for r in rows)]
    methods=['QField-Dyn',*references]
    entries=[]
    for row in sorted(rows,key=lambda r:(r['id'],r['tier'])):
        for method in methods:
            model=method=='QField-Dyn'
            metrics=dict(row['metrics'] if model else row['baselines'][method])
            if 'phys_v2' in row and (model or method in row['baseline_phys_v2']):
                metrics['Phys_v2']=row['phys_v2'] if model else row['baseline_phys_v2'][method]
            for metric,value in flatten_metrics(metrics).items():
                entries.append(dict(split=split,id=row['id'],tier=row['tier'],method=method,metric=metric,value=value))
    write_csv(output/'all_prefix_metrics.csv',entries)
    names=sorted({e['metric'] for e in entries});aggregates=[];paired=[];rng=np.random.default_rng(202609191437)
    lookup={(e['id'],e['tier'],e['method'],e['metric']):e['value'] for e in entries}
    for tier in ('all','T1','T2','T3'):
        tiers=('T1','T2','T3') if tier=='all' else (tier,)
        for metric in names:
            systems={}
            for method in methods:
                scores={}
                for identifier in ids:
                    values=[lookup[(identifier,t,method,metric)] for t in tiers if (identifier,t,method,metric) in lookup]
                    if values:scores[identifier]=float(np.mean(values))
                systems[method]=scores;values=np.array(list(scores.values()))
                if len(values):aggregates.append(dict(split=split,tier=tier,method=method,metric=metric,
                    mean=float(values.mean()),median=float(np.median(values)),standard_deviation=float(values.std(ddof=1)) if len(values)>1 else 0.,systems=len(values)))
            for reference in references:
                shared=sorted(systems['QField-Dyn'].keys()&systems[reference].keys())
                delta=np.array([systems['QField-Dyn'][i]-systems[reference][i] for i in shared])
                if not len(delta):continue
                bootstrap=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(-1)
                paired.append(dict(split=split,tier=tier,reference=reference,metric=metric,systems=len(delta),
                    mean_difference=float(delta.mean()),median_difference=float(np.median(delta)),
                    lower_fraction=float((delta<0).mean()),equal_fraction=float((delta==0).mean()),
                    ci95_low=float(np.quantile(bootstrap,.025)),ci95_high=float(np.quantile(bootstrap,.975))))
    write_csv(output/'metric_summary.csv',aggregates);write_csv(output/'paired_system_comparisons.csv',paired)
    index={(r['tier'],r['method'],r['metric']):r['mean'] for r in aggregates}
    lines=[f'# {split}: complete experimental tables','',
           f'{len(ids)} systems; {len(rows)} cases; {rows[0]["paths"]} model paths per case. Equal systems within each tier.',
           'Paired comparisons resample systems. A lower difference is favorable only for error/severity metrics; valid fractions favor higher values.','']
    for tier in ('T1','T2','T3','all'):
        lines += [f'## {tier}','','| Metric | '+' | '.join(methods)+' |','|---|'+'---:|'*len(methods)]
        for label,key in KEYS.items():
            values=[index.get((tier,method,key)) for method in methods]
            lines.append('| '+label+' | '+' | '.join('NA' if x is None else f'{x:.6g}' for x in values)+' |')
        lines.append('')
    (output/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    report=dict(completed=True,split=split,systems=len(ids),prefixes=len(rows),model_paths=rows[0]['paths'],
                baseline_independent_paths=1,defined_metrics=len(names),aggregation='equal systems, equal defined tiers',
                uncertainty_unit='system',source_ids=ids,negative_results_preserved=True)
    (output/'manifest.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--shards',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',required=True)
    a=p.parse_args();reports=[json.loads((a.input/f'review_{i}.json').read_text()) for i in range(a.shards)]
    assert all(r['completed'] for r in reports)
    rows=[row for report in reports for row in report['rows']]
    print(json.dumps(tables(rows,a.output,a.split)))


if __name__=='__main__':main()
