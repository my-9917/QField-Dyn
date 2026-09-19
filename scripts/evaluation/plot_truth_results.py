"""Observed test errors and physical validity on the fixed labelled 90 cases."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['Linear','NeuralMD','QField-Dyn']
COLORS=['#7b848e','#668fae','#568778']
METRICS=[('Geo.mean_rmsd_angstrom','Geo: RMSD (Å)'),
    ('Phys_v2_common_heavy.assessed_tail_valid_frame_fraction','Phys v2: valid-frame fraction'),
    ('Dyn.rmsf_mae_angstrom','Dyn: RMSF MAE (Å)'),
    ('Dyn.contacts.observed_pocket.brier','Dyn: contact Brier'),
    ('Stab.late.rmsd_angstrom','Stab: late-window RMSD (Å)'),
    ('Probability.feature_energy_score','Single-path feature ES')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--tables',type=Path,required=True);a=p.parse_args()
    completed=json.loads((a.tables/'completion.json').read_text());assert completed['prefixes']==90
    with (a.tables/'all_prefix_metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for row in rows:
        if row['method'] == 'QMem-SemiFlexFlow': row['method'] = 'QField-Dyn'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7,'legend.frameon':False})
    rng=np.random.default_rng(20260920);summaries=[]
    fig,axes=plt.subplots(2,3,figsize=(10,6.4));fig.subplots_adjust(left=.08,right=.99,bottom=.19,top=.90,hspace=.45,wspace=.32)
    for panel,(ax,(metric,title)) in enumerate(zip(axes.flat,METRICS)):
        for offset,(method,color) in enumerate(zip(METHODS,COLORS)):
            means=[];lower=[];upper=[]
            for tier in ('T1','T2','T3'):
                values=np.array([float(r['value']) for r in rows if r['tier']==tier and r['method']==method and r['metric']==metric])
                assert len(values)>0
                boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(-1)
                mean=float(values.mean());lo,hi=np.quantile(boot,[.025,.975])
                means.append(mean);lower.append(lo);upper.append(hi)
                summaries.append(dict(tier=tier,metric=metric,method=method,mean=mean,ci95_low=float(lo),ci95_high=float(hi),cases=len(values)))
            x=np.arange(3)+(offset-1)*.22;means=np.asarray(means)
            ax.errorbar(x,means,yerr=[means-np.asarray(lower),np.asarray(upper)-means],fmt='o',color=color,capsize=3,markersize=4,label=method)
        ax.set_xticks(range(3),['T1','T2','T3']);ax.set_ylabel(title);ax.grid(axis='y',color='#e6e9eb',linewidth=.6)
        ax.set_ylim(bottom=0)
        if metric.startswith('Phys_'):ax.set_ylim(-.03,1.03)
        ax.text(-.18,1.05,chr(97+panel),transform=ax.transAxes,fontweight='bold',fontsize=12)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.09),ncol=3)
    fig.suptitle('Trajectory prediction on 90 labelled validation complexes',fontsize=12,y=.975)
    fig.text(.08,.055,'30 distinct complexes per tier; common ligand heavy atoms. 95% complex-bootstrap intervals.',fontsize=8)
    fig.text(.08,.029,'One trajectory per case. NeuralMD uses author-pretrained weights; training conditions differ between models.',fontsize=7)
    for suffix in ('svg','pdf','png'):fig.savefig(a.tables/f'comparison.{suffix}',dpi=250,facecolor='white')
    plt.close(fig)
    curves=json.loads((a.tables/'frame_curves.json').read_text())
    for row in curves:
        if row['method'] == 'QMem-SemiFlexFlow': row['method'] = 'QField-Dyn'
    fig,axes=plt.subplots(1,3,figsize=(10,3.2))
    for ax,tier in zip(axes,('T1','T2','T3')):
        for method,color in zip(METHODS,COLORS):
            selected=[r for r in curves if r['tier']==tier and r['method']==method];assert len(selected)==30
            ax.plot(np.asarray(selected[0]['lead_times_ps'])/1000,np.mean([r['rmsd_angstrom'] for r in selected],axis=0),color=color,label=method)
        ax.set_title(tier);ax.set_xlabel('Prediction lead time (ns)');ax.set_ylabel('Mean RMSD (Å)');ax.set_ylim(bottom=0)
    axes[-1].legend(fontsize=7);fig.tight_layout()
    for suffix in ('svg','pdf','png'):fig.savefig(a.tables/f'error_curves.{suffix}',dpi=250,facecolor='white')
    plt.close(fig)
    (a.tables/'figure_source_data.json').write_text(json.dumps(summaries,indent=2))
    (a.tables/'figure_caption.md').write_text('''Ninety distinct labelled validation complexes; 30 per task, selected before generation with zero complex-ID overlap with both models' training lists. All panels use common ligand heavy atoms. Points show equal-complex means with 2,000 complex-bootstrap resamples for 95% intervals. Contact scores include only complexes with a defined observed pocket, with counts in the source data. Lower errors and higher valid-frame fractions indicate better performance. QField-Dyn uses the third-epoch main model and second-epoch structural adapter; NeuralMD uses an author-pretrained checkpoint under different training conditions. Each method outputs one trajectory per case. Feature ES has the single-realization deterministic-form interpretation. Curves show mean framewise RMSD. Source data include all 90 complexes.\n''',encoding='utf-8')
    print(json.dumps(dict(completed=True,methods=METHODS,figures=['comparison','error_curves'])))


if __name__=='__main__':main()
