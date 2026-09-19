"""Fixed-camera trajectory GIFs from the first preregistered case of each tier."""
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
from trajectory_delivery import read_xtc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selection', type=Path)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    manifest = json.loads((a.root / 'manifest.json').read_text())
    selection = (json.loads(a.selection.read_text()) if a.selection else
                 dict(tasks=[next(t for t in manifest['tasks'] if t['tier'] == tier) for tier in ['T1', 'T2', 'T3']],
                      rule='first case per tier in frozen manifest', caption='First frozen case per tier; illustrative, not selected by performance.'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    reviews = []
    for task in selection['tasks']:
        tier = task['tier']
        record = torch.load(task['record'], map_location='cpu', weights_only=False)
        saved = torch.load(a.root / 'qmem' / (task['key'] + '.pt'), map_location='cpu', weights_only=False)
        neural = torch.load(a.root / 'neuralmd' / (task['key'] + '.pt'), map_location='cpu', weights_only=False)
        inputs = record['inputs']; graph = inputs['ligand_graph']
        heavy = np.asarray(graph['atomic_numbers']) > 1
        ligand_indices = np.asarray(graph['source_atom_indices'])[heavy]
        directory = a.root / 'trajectories' / (task['evaluation_case'] + '_' + task['id'])
        origin = np.asarray(record['transform']['origin'])
        truth_file = read_xtc(directory / 'truth.xtc')
        predicted_file = read_xtc(directory / 'pred.xtc')
        truth = truth_file['coordinates_angstrom'][:, ligand_indices] - origin
        predicted = predicted_file['coordinates_angstrom'][:, ligand_indices] - origin
        np.testing.assert_array_equal(predicted, saved['encoded_paths'][0][:, heavy])
        actual = [truth, predicted, np.asarray(neural['coordinates'])]
        labels = ['Ground truth', 'QField-Dyn', 'NeuralMD']
        colors = ['#596774', '#348375', '#648daa']
        observed = np.asarray(inputs['X_obs'])[-1, heavy]
        centre = observed.mean(0)
        # One display-only rigid rotation, determined exclusively by the observed ligand.
        _, _, rotation = np.linalg.svd(observed - centre, full_matrices=False)
        if np.linalg.det(rotation) < 0: rotation[-1] *= -1
        paths = [(x - centre) @ rotation.T for x in actual]
        last = (observed - centre) @ rotation.T
        protein = np.asarray(inputs['P0'])
        protein_heavy = np.asarray(inputs['protein_topology']['atomic_numbers']) > 1
        pocket = protein_heavy & (np.linalg.norm(protein[:, None] - observed[None], axis=-1).min(-1) < 7)
        protein_points = (protein[pocket] - centre) @ rotation.T
        reverse = np.full(len(heavy), -1, dtype=int); reverse[heavy] = np.arange(heavy.sum())
        bonds = np.asarray(graph['bonds'], dtype=int)
        bonds = reverse[bonds[heavy[bonds].all(-1)]]
        union = np.concatenate([x.reshape(-1, 3) for x in paths] + [last])
        lower, upper = union.min(0) - 2, union.max(0) + 2
        middle = (lower + upper) / 2; radius = max((upper - lower).max() / 2, 4)
        errors = [np.sqrt(np.square(x - truth).sum(-1).mean(-1)) for x in actual[1:]]
        n = len(truth); lead = np.arange(1, n + 1) * task['dt_ps'] / 1000
        absolute = np.arange(task['n_obs'], task['n_obs'] + n) * task['dt_ps'] / 1000
        fig = plt.figure(figsize=(12, 6.5), dpi=110)
        fig.suptitle(f'{tier}  |  {task["id"]}  |  Ligand trajectory in a fixed protein pocket', x=.5, y=.974, fontsize=15)
        clock = fig.text(.5, .916, '', ha='center', fontsize=11)
        artists = []
        for j, (label, color, path) in enumerate(zip(labels, colors, paths)):
            ax = fig.add_axes([.015 + j * .33, .27, .32, .61], projection='3d')
            ax.set_proj_type('ortho'); ax.view_init(elev=24, azim=-62)
            ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
            ax.set_xlim(middle[0] - radius, middle[0] + radius)
            ax.set_ylim(middle[1] - radius, middle[1] + radius)
            ax.set_zlim(middle[2] - radius, middle[2] + radius)
            ax.scatter(*protein_points.T, s=7, c='#c5cbd0', alpha=.20, depthshade=False)
            ax.add_collection3d(Line3DCollection(last[bonds], colors='#9ba3aa', linewidths=1, linestyles='dashed', alpha=.48))
            lines = Line3DCollection(path[0][bonds], colors=color, linewidths=2.5)
            ax.add_collection3d(lines)
            atoms = ax.scatter(*path[0].T, s=23, c=color, edgecolors='white', linewidths=.35, depthshade=False)
            ax.set_title(label, color=color, fontsize=13, pad=-3)
            score = ax.text2D(.5, -.01, '', ha='center', transform=ax.transAxes, fontsize=10)
            artists.append((atoms, lines, score))
        chart = fig.add_axes([.11, .12, .79, .13])
        for label, color, error in zip(labels[1:], colors[1:], errors):
            chart.plot(lead, error, color=color, lw=1.6, label=label)
        cursor = chart.axvline(lead[0], color='#b6bdc2', lw=1)
        markers = [chart.plot([lead[0]], [e[0]], 'o', ms=4, color=c)[0] for e, c in zip(errors, colors[1:])]
        chart.set_xlim(0, lead[-1]); chart.set_ylim(0, max(max(e) for e in errors) * 1.15)
        chart.set_xlabel('Prediction lead time (ns)', fontsize=9); chart.set_ylabel('RMSD (Å)', fontsize=9)
        chart.tick_params(labelsize=8); chart.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
        fig.text(.5, .042, 'Common ligand heavy atoms • Gray: fixed pocket • Dashed: last observation • Same fixed camera and scale', ha='center', fontsize=8)
        fig.text(.5, .017, selection['caption'] + ' QField-Dyn E3 + Adapter E2; NeuralMD author-pretrained transfer.', ha='center', fontsize=7.5)

        def update(frame):
            clock.set_text(f'Frame {frame + 1}/{n}     Absolute time {absolute[frame]:.3f} ns     Lead {lead[frame]:.3f} ns')
            for j, ((atoms, lines, score), path) in enumerate(zip(artists, paths)):
                atoms._offsets3d = tuple(path[frame].T)
                lines.set_segments(path[frame][bonds])
                score.set_text('Reference MD trajectory' if j == 0 else f'Frame RMSD: {errors[j - 1][frame]:.3f} Å')
            cursor.set_xdata([lead[frame], lead[frame]])
            for marker, error in zip(markers, errors): marker.set_data([lead[frame]], [error[frame]])
            return []

        animation = FuncAnimation(fig, update, frames=n, interval=200, blit=False, repeat=True)
        filename = a.output / (task['key'] + '.gif')
        animation.save(filename, writer=PillowWriter(fps=5), dpi=110)
        for frame in sorted({0, n // 2, n - 1}):
            update(frame); fig.savefig(a.output / f'{task["key"]}_frame_{frame + 1:03d}.png', dpi=110)
        plt.close(fig)
        with Image.open(filename) as check:
            assert check.n_frames == n
            durations = []
            for frame in range(n): check.seek(frame); durations.append(check.info['duration'])
            assert durations == [200] * n
            size = check.size
        review = dict(case=task['key'],selection=selection['rule'],frames=n,
                      physical_dt_ps=80,playback_fps=5,frame_interpolation=False,
                      common_ligand_heavy_atoms=int(heavy.sum()),size_pixels=size,
                      first_absolute_ns=float(absolute[0]),last_absolute_ns=float(absolute[-1]),
                      mean_rmsd={'QField-Dyn': float(errors[0].mean()), 'NeuralMD': float(errors[1].mean())},
                      display_rotation_from_observation=rotation.tolist(),fixed_camera=True,
                      coordinates='QField-Dyn and truth read from delivered XTC; no per-frame ligand alignment',
                      source_task=task,file=filename.name,passed=True)
        (a.output / (task['key'] + '.json')).write_text(json.dumps(review, indent=2))
        reviews.append(review); print(json.dumps(dict(completed=task['key'],frames=n)), flush=True)
    (a.output / 'review.json').write_text(json.dumps(dict(completed=True,selection=selection,rows=reviews), indent=2))


if __name__ == '__main__':
    main()
