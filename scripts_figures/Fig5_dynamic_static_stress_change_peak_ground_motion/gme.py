import numpy as np


def rotate_horizontal(series_x, series_y, angle):
    """
    from https://github.com/GEMScienceTools/gmpe-smtk/blob/master/smtk/intensity_measures.py
    Rotates two time-series according to a specified angle
    :param nunmpy.ndarray series_x:
        Time series of x-component
    :param nunmpy.ndarray series_y:
        Time series of y-component
    :param float angle:
        Angle of rotation (decimal degrees)
    """
    angle = angle * (np.pi / 180.0)
    rot_hist_x = (np.cos(angle) * series_x) + (np.sin(angle) * series_y)
    rot_hist_y = (-np.sin(angle) * series_x) + (np.cos(angle) * series_y)
    return rot_hist_x, rot_hist_y


def compute_gmrotdpp_PGA(x_a, y_a):
    angles = np.arange(0.0, 90.0, 1.0)
    max_a_theta = np.zeros([len(angles), 1], dtype=float)

    for iloc, theta in enumerate(angles):
        if iloc == 0:
            max_a_theta[iloc, :] = np.sqrt(
                np.max(np.fabs(x_a), axis=0) * np.max(np.fabs(y_a), axis=0)
            )
        else:
            rot_x, rot_y = rotate_horizontal(x_a, y_a, theta)
            max_a_theta[iloc, :] = np.sqrt(
                np.max(np.fabs(rot_x), axis=0) * np.max(np.fabs(rot_y), axis=0)
            )

    return np.percentile(max_a_theta, 50.0, axis=0)[0]
