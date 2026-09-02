import numpy as np


SEGMENTS = ((0, 10), (10, 20), (20, 40), (40, 80))


def build_bases(previous, current, prediction, heavy):
    previous_center = previous[heavy].mean(axis=0)
    current_center = current[heavy].mean(axis=0)
    prediction_center = prediction[:, heavy].mean(axis=1)
    centroid_velocity = current_center - previous_center
    translation = prediction_center - current_center
    internal_deformation = (
        prediction
        - prediction_center[:, None]
        - (current - current_center)[None]
    )
    internal_velocity = current - previous - centroid_velocity
    return np.stack(
        (
            np.broadcast_to(translation[:, None], prediction.shape),
            np.broadcast_to(centroid_velocity, prediction.shape),
            internal_deformation,
            np.broadcast_to(internal_velocity, prediction.shape),
        ),
        axis=-1,
    )


def apply_patch_residual(previous, current, prediction, heavy, coefficients):
    system_bases = build_bases(previous, current, prediction, heavy)
    corrected = prediction.copy()
    for segment_index, (start, stop) in enumerate(SEGMENTS):
        corrected[start:stop] += np.sum(
            system_bases[start:stop]
            * coefficients[segment_index][None, None, None],
            axis=-1,
        )
    return corrected

