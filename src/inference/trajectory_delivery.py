"""Truth-free medoid selection and physical-unit XTC readback checks."""
import numpy as np
from MDAnalysis.lib.formats.libmdaxdr import XTCFile
from scipy.spatial.distance import cdist


XTC_PRECISION = 5


def write_xtc(path, coordinates, times, steps, box):
    """One full-atom writer for output selection, feedback and delivered files."""
    import MDAnalysis as mda
    xyz = np.asarray(coordinates)
    universe = mda.Universe.empty(xyz.shape[1], trajectory=True)
    universe.trajectory.ts.triclinic_dimensions = np.asarray(box, dtype=np.float32)
    with mda.Writer(str(path), n_atoms=xyz.shape[1], precision=XTC_PRECISION) as writer:
        for frame, timestamp, step in zip(xyz, times, steps):
            universe.atoms.positions = frame
            universe.trajectory.ts.time = float(timestamp)
            universe.trajectory.ts.frame = int(step)
            writer.write(universe.atoms)


def coordinate_encoding_error_bound(maximum_absolute_coordinate_angstrom):
    """Per-axis half-grid error plus float32 writer/reader conversion roundoff."""
    return .5*10.**(1-XTC_PRECISION)+4*np.finfo(np.float32).eps*maximum_absolute_coordinate_angstrom


def ensemble_medoid(features, scales):
    points = (np.asarray(features)/np.asarray(scales)).reshape(len(features), -1)
    return int(np.argmin(cdist(points, points).sum(1)))


def read_xtc(path):
    """Read physical coordinates and the actual stored frame metadata."""
    with XTCFile(str(path)) as reader:
        frames = [reader.read() for _ in range(len(reader))]
    return dict(coordinates_angstrom=np.asarray([frame.x for frame in frames], dtype=float)*10.,
        times_ps=np.asarray([frame.time for frame in frames]),
        boxes_angstrom=np.asarray([frame.box for frame in frames], dtype=float)*10.,
        steps=np.asarray([frame.step for frame in frames]), precisions=np.asarray([frame.prec for frame in frames]))


def review_xtc(path, coordinates, meta, expected_box):
    """XTC stores nm; model and MDAnalysis coordinates use Angstrom."""
    xyz = np.asarray(coordinates)
    stored = read_xtc(path)
    restored, times, boxes = (stored[key] for key in ['coordinates_angstrom', 'times_ps', 'boxes_angstrom'])
    assert restored.shape == xyz.shape == (meta['n_pred'], meta['n_atoms'], 3)
    assert np.isfinite(restored).all() and np.isfinite(boxes).all()
    expected_times = np.arange(meta['n_obs'], meta['n_obs']+meta['n_pred'])*meta['dt_ps']
    assert np.array_equal(stored['steps'], np.arange(meta['n_obs'], meta['n_obs']+meta['n_pred']))
    assert np.all(stored['precisions'] == 10.**XTC_PRECISION)
    error = float(np.max(np.abs(restored-xyz)))
    time_error = float(np.max(np.abs(times-expected_times)))
    box_error = float(np.max(np.abs(boxes-expected_box)))
    encoding_bound = float(coordinate_encoding_error_bound(np.abs(xyz).max()))
    assert error <= encoding_bound and time_error < .01 and box_error < .001
    return {'passed': True, 'frames': len(restored), 'atoms': xyz.shape[1],
            'coordinate_roundtrip_max_angstrom': error, 'timestamp_max_error_ps': time_error,
            'box_max_error_angstrom': box_error, 'first_time_ps': float(times[0]), 'last_time_ps': float(times[-1]),
            'xtc_precision': XTC_PRECISION, 'coordinate_error_bound_angstrom': encoding_bound}
