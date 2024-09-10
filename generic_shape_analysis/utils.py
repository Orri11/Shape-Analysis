import geomstats.backend as gs
import numpy as np

def apply_func_to_list(input_list, func, output_index = 0):
    """Apply the input function func to the input list input_list.

    This function goes through the list and applies the function to every element in the list.

    Parameters
    ----------
    input_ds : list
        Input list.
    func : callable
        Function to be applied to the values of the list, i.e. the shapes.
    output_index = 0) : int 
        Index of the output elements to be used. Default is 0, as functions should return the transformed shape array as first element of the output.

    Returns
    -------
    output_list : list
    """
    output_list = []
    for i , shape in enumerate(input_list):
        output_list.append(func(shape)[output_index])
        output_list[i] = np.array(output_list[i])
    return output_list


def interpolate(curve, nb_points, tol=1e-10):
    """Interpolate a discrete curve with nb_points from a discrete curve.
       Then, process curve to ensure that there are no consecutive duplicate points 
    Parameters
    ----------
    curve : array_like
        Shape to interpolate.
    nb_points : int
        Desired number of points of the interpolated shape.
    tol : float, optional
        Tolerance for the distance between two consecutive points. If the distance is smaller than tol, the two points are considered to be the same.

    Returns
    -------
    interpolation : array_like
        Shape with nb_points points
    """
    old_length = curve.shape[0]
    interpolation = gs.zeros((nb_points, 2))
    incr = old_length / nb_points
    pos = 0
    for i in range(nb_points):
        index = int(gs.floor(pos))
        interpolation[i] = curve[index] + (pos - index) * (
            curve[(index + 1) % old_length] - curve[index]
        )
        pos += incr

    
    dist = interpolation[1:] - interpolation[:-1]
    dist_norm = np.sqrt(np.sum(np.square(dist), axis=1))

    if np.any(dist_norm < tol):
        for i in range(len(interpolation) - 1):
            if np.sqrt(np.sum(np.square(interpolation[i + 1] - interpolation[i]), axis=0)) < tol:
                interpolation[i + 1] = (interpolation[i] + interpolation[i + 2]) / 2

    return (interpolation,)