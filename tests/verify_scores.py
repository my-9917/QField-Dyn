"""Independent metric identities and native generated-geometry checks."""
import json
from pathlib import Path
import numpy as np
import torch
from semiflexible_scores import probability_scores, score_paths, physics_scores
from ligand_geometry import geometry_terms

torch.set_num_threads(2)
source = Path('results/interface_v1')
record = torch.load(source/'1B6H_T1.pt', weights_only=False, map_location='cpu')
truth = np.asarray(record['X_future'])
paths = np.stack([truth, truth+np.array([.15, -.2, .1])])
metrics, detail = score_paths(paths, record, np.ones(3))
reference = np.array([[.3, .4, .1], [.5, .7, .2]])
samples = np.stack([reference, reference+.2, reference-.1])
scales = np.array([1., 2., 3.])
result = probability_scores(samples, reference, scales)
z, y = (samples/scales).reshape(3, -1), (reference/scales).ravel()
es = (np.linalg.norm(z-y, axis=1).mean()-sum(np.linalg.norm(z[a]-z[b]) for a in range(3) for b in range(a))/6)/np.sqrt(y.size)
assert abs(result['feature_energy_score']-es) < 1e-12
zero, _ = score_paths(np.stack([truth, truth]), record, np.ones(3))
assert zero['Geo']['coordinate_energy_score_angstrom'] == 0
assert zero['Probability']['feature_energy_score'] == 0
assert all(row['crps'] == 0 and row['coverage_90'] == 1 for row in zero['Probability']['features'].values())
assert zero['Geo']['mean_rmsd_angstrom'] == 0
torch_terms = geometry_terms(torch.from_numpy(paths), record['geometry'])
physics, _ = physics_scores(paths, record['geometry'])
assert abs(physics['severe_overlap_mean_squared']-float(torch_terms['severe_overlap'].mean())) < 1e-12
assert abs(physics['bond_mean_squared_excess']-float(torch_terms['bond_strain'].mean())) < 1e-12
assert abs(physics['angle_mean_squared_excess']-float(torch_terms['angle_strain'].mean())) < 1e-12
out = Path('results/score_interface_v1'); out.mkdir()
(out/'review.json').write_text(json.dumps(dict(passed=True, feature_es_absolute_error=abs(result['feature_energy_score']-es),
    exact_truth_identities_passed=True, loss_evaluation_geometry_agreement=True, metrics=metrics), indent=2))
print(json.dumps(dict(passed=True, feature_es_absolute_error=abs(result['feature_energy_score']-es))))
