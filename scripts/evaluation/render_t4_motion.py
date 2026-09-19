"""Render all saved long-range frames as a motion demonstration with physical time."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
from competition_io import public_cases, read_observation
from semiflexible_inputs import observation_inputs
from trajectory_delivery import read_xtc


def main():
    p = argparse.ArgumentParser()
    for key in ('artifact', 'xtc', 'public-root', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--version-label', required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    saved = torch.load(a.artifact, map_location='cpu', weights_only=False)
    row, meta = next((r, m) for r, m in public_cases(a.public_root) if m['id'] == a.case)
    case = read_observation(a.public_root, row, meta)
    inputs, _, transform = observation_inputs(case)
    graph = inputs['ligand_graph']; heavy = np.asarray(graph['atomic_numbers']) > 1
    xtc = read_xtc(a.xtc)
    all_ligand = xtc['coordinates_angstrom'][:, case['ligand_indices']] - transform['origin']
    np.testing.assert_array_equal(all_ligand, saved['ligand'])
    assert len(all_ligand) == 490 and np.isfinite(all_ligand).all()
    xyz = all_ligand[:, heavy]; observed = np.asarray(inputs['X_obs'])[-1, heavy]
    centre = observed.mean(0)
    _, _, rotation = np.linalg.svd(observed - centre, full_matrices=False)
    if np.linalg.det(rotation) < 0: rotation[-1] *= -1
    xyz = (xyz - centre) @ rotation.T; last = (observed - centre) @ rotation.T
    protein = np.asarray(inputs['P0'])
    keep = (np.asarray(inputs['protein_topology']['atomic_numbers']) > 1) & (np.linalg.norm(protein[:, None] - observed[None], axis=-1).min(-1) < 7)
    pocket = (protein[keep] - centre) @ rotation.T
    reverse = np.full(len(heavy), -1, dtype=int); reverse[heavy] = np.arange(heavy.sum())
    bonds = np.asarray(graph['bonds'], dtype=int); bonds = reverse[bonds[heavy[bonds].all(-1)]]
    lower = np.minimum(xyz.min((0, 1)), last.min(0)) - 2
    upper = np.maximum(xyz.max((0, 1)), last.max(0)) + 2
    middle = (lower + upper) / 2; radius = max(float((upper - lower).max() / 2), 4)
    times = np.arange(meta['n_obs'], meta['n_obs'] + 490) * meta['dt_ps'] / 1000
    com = xyz.mean(1); displacement = np.linalg.norm(com - last.mean(0), axis=-1)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig = plt.figure(figsize=(9, 6.5), dpi=110)
    fig.suptitle(f'QField-Dyn  |  {a.case}  |  Long-range trajectory', fontsize=15, y=.975)
    clock = fig.text(.5, .925, '', ha='center')
    ax = fig.add_axes([.10, .24, .8, .65], projection='3d')
    ax.set_proj_type('ortho'); ax.view_init(24, -62); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.set_xlim(middle[0]-radius, middle[0]+radius); ax.set_ylim(middle[1]-radius, middle[1]+radius); ax.set_zlim(middle[2]-radius, middle[2]+radius)
    ax.scatter(*pocket.T, s=7, c='#b8c1c8', alpha=.2, depthshade=False)
    ax.add_collection3d(Line3DCollection(last[bonds], colors='#8c979e', linewidths=1, linestyles='dashed', alpha=.45))
    lines = Line3DCollection(xyz[0][bonds], colors='#348375', linewidths=2.4); ax.add_collection3d(lines)
    atoms = ax.scatter(*xyz[0].T, s=24, c='#348375', edgecolors='white', linewidths=.3, depthshade=False)
    trail, = ax.plot([], [], [], color='#7caba1', lw=1.2, alpha=.65)
    chart = fig.add_axes([.13, .12, .75, .13]); chart.plot(times, displacement, c='#348375', lw=1.3)
    cursor = chart.axvline(times[0], c='#9aa4aa', lw=1)
    chart.set(xlabel='Physical time (ns)', ylabel='COM displacement (Å)', xlim=(times[0], times[-1]))
    chart.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .033, 'Generated motion demonstration • Fixed pocket • Dashed: last observation • All 490 frames', ha='center', fontsize=8)
    fig.text(.5, .013, a.version_label + ' • Future-reference accuracy: unassessed', ha='center', fontsize=7.5)

    def update(i):
        clock.set_text(f'Frame {i+1}/490     Physical time {times[i]:.0f} ns')
        atoms._offsets3d = tuple(xyz[i].T); lines.set_segments(xyz[i][bonds])
        trail.set_data_3d(*com[:i+1].T); cursor.set_xdata([times[i], times[i]])
        return []

    target = a.output / f'{a.case}.gif'
    FuncAnimation(fig, update, frames=490, interval=100, blit=False).save(target, writer=PillowWriter(fps=10), dpi=110)
    for i in (0, 244, 489):
        update(i); fig.savefig(a.output / f'{a.case}_frame_{i+1:03d}.png', dpi=110)
    plt.close(fig)
    with Image.open(target) as check: assert check.n_frames == 490
    report = dict(completed=True, model='QField-Dyn', case=a.case, frames=490, physical_dt_ps=meta['dt_ps'],
                  first_time_ns=float(times[0]), last_time_ns=float(times[-1]), frame_interpolation=False,
                  fixed_camera=True, playback_fps=10, source_xtc=str(a.xtc), source_artifact=str(a.artifact),
                  version=a.version_label, future_reference_accuracy='unassessed', file=target.name,
                  base_sha256=saved['config']['base_sha256'], adapter_sha256=saved['config']['adapter_sha256'])
    (a.output / f'{a.case}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
