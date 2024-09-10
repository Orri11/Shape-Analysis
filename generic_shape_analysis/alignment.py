import numpy as np
import scipy 
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import normalize

import utils

### Note : All functions return a tuple, even if only one value is returned. This is to make all functions compatible with the apply_func_to_list function.


def apply_correspondence(shape_list, correspondence_list):
    """Create correspondence among all shapes in a shape list based on a pre-made list of corresponding points. 
    The function will re-order the shapes so every shape's respective corresponding points will be in the same index.
    Parameters
    ----------
    shape_list : list of array_like elements
        List of shapes to be aligned. Shapes must have the same number of points.
    correspondence_points : array_like
        Correspondence points to be applied to the shapes. The points are respresented as a list of indices, where the i-th element of the list 
        is the index of the corresponding point. The list must be of the same order as the the shapes in the shape_list and contain the same number of correspondence points for each element.

    Returns
    -------
    aligned_shapes : list of array_like elements
        List of shapes with the correspondence points applied.
        The number of points in each shape will be the same, but may differ from the number of points in the original shapes.
    """
    
    shape_length = len(shape_list[0])
    # Find the maximum distance between each pair of corresponding points across all shapes
    num_points = len(correspondence_list[0])
    max_dists = np.zeros(num_points).astype(int)
    for shape_points in (correspondence_list):
        for j , point in enumerate(shape_points):
            if j+1 == num_points:
                dist = ( shape_length + shape_points[0] ) - shape_points[j] if shape_points[j] > shape_points[0] else shape_points[0] - shape_points[j]
                if dist > max_dists[j]:
                    max_dists[j] = dist
            else:
                dist = shape_points[j+1] - shape_points[j] if shape_points[j+1] > shape_points[j] else ( shape_length + shape_points[j+1] ) - shape_points[j]
                if dist > max_dists[j]:
                    max_dists[j] = dist

    # Re-order the shapes so that the first correspondence point is at index 0 
    aligned_shapes = []
    for i , shape_points in enumerate(correspondence_list):
        shape_to_align = shape_list[i].copy()
        for j , point in enumerate(shape_points):
            if j == 0:
                dist = shape_points[j+1] - shape_points[j] if shape_points[j+1] > shape_points[j] else ( shape_length + shape_points[j+1] ) - shape_points[j]
                sub_array = [shape_list[i][(point + k) % shape_length] for k in range(dist)]
                shape_to_align[0:dist] = sub_array
                
                # Interpolate the number of points between the first and second correspondence points so they are equal 
                # to the maximum distance between the two points across all shapes.
                temp = shape_to_align[0:dist]
                section = utils.interpolate(temp, max_dists[j])[0]
                interp_shape = section
            elif j+1 < num_points:
                new_index = shape_points[j] - shape_points[0] if shape_points[j] > shape_points[0] else ( shape_length + shape_points[j] ) - shape_points[0]
                dist = shape_points[j+1] - shape_points[j] if shape_points[j+1] > shape_points[j] else ( shape_length + shape_points[j+1] ) - shape_points[j]
                sub_array = [shape_list[i][(point + k) % shape_length] for k in range(dist)]
                shape_to_align[new_index:new_index+dist] = sub_array 
                
                # Interpolate the number of points between the current correspondence point and the next one so they are equal 
                # to the maximum distance between the two points across all shapes.
                temp = shape_to_align[new_index:new_index+dist]
                section = utils.interpolate(temp, max_dists[j])[0]
                interp_shape = np.vstack((interp_shape,section))
            else:
                new_index = shape_points[j] - shape_points[0] if shape_points[j] > shape_points[0] else ( shape_length + shape_points[j] ) - shape_points[0]
                dist = shape_points[0] - shape_points[j] if shape_points[0] > shape_points[j] else ( shape_length + shape_points[0] ) - shape_points[j]
                sub_array = [shape_list[i][(point + k) % shape_length] for k in range(dist)]
                shape_to_align[new_index:new_index+dist] = sub_array

                # Interpolate the number of points between the last correspondence point and the first one so they are equal 
                # to the maximum distance between the two points across all shapes.
                temp = shape_to_align[new_index:new_index+dist]
                section = utils.interpolate(temp, max_dists[j])[0]
                interp_shape = np.vstack((interp_shape,section))
                interp_shape = np.vstack((interp_shape,interp_shape[0]))

        aligned_shapes.append(interp_shape)
    return (aligned_shapes,max_dists)


def procrustes(shape, ref_shape):
    """
    Perform Procrustes analysis on two shapes. The function will standardize and center around origin the the shapes.
    Then, it will find the best rotation and scaling to fit the second shape to the first one.
    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1`.
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1.

    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The roation and scale of `data2` that best fits `data1`. Centered, but not necessarily
        standardized. 


    """
    mtx1 = np.array(ref_shape, dtype=np.float64, copy=True)
    mtx2 = np.array(shape, dtype=np.float64, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return (mtx2, mtx1)


def exhaustive_search(shape, ref_shape):
    """Perform an exhaustive search of all points in a shape to find the best Procrustes alignment solution to the reference shape.

    Returns
    -------
    aligned_shape : array_like
        The shape scaled, rotated and point-reordered, according to the parameters which minimize L2 distances from the reference shape.
    """
    num_points = len(shape)
    distances = np.zeros(num_points)
    for shift in range(num_points):
        reparametrized = [shape[(i + shift) % num_points] for i in range(num_points)]
        aligned, std_ref_shape = procrustes(reparametrized, ref_shape)
        distances[shift] =  np.sqrt(np.sum(np.square(aligned - std_ref_shape)))
    shift_min = np.argmin(distances)
    reparametrized_min = [
        shape[(i + shift_min) % num_points] for i in range(num_points)
    ]
    aligned_shape,_ = procrustes(reparametrized_min, ref_shape)

    return (aligned_shape,)

def generalized_procrustes(shape_list , ex_search = True , tol = 1e-3):
    """Perform the generalized Procrustes analysis on a list of shapes.
     The function uses the first shape in the dataset as the mean shape, aligns all other shapes to it. Then it re-calculates the mean shape based on all aligned shapes and aligns the new mean
     to the old mean shape. If the new mean shape does not converge to the old mean shape, the process is repeated until convergence.
     ** Convergence is defined as the L2 distance between the new mean shape and the old mean shape being smaller than a given tolerance.
    Parameters
    ----------
    shape_list : list of array_like elements
        List of shapes to be aligned.
    ex_search : bool, optional
        If True, the function will perform an exhaustive search of all points in a shape to find the best Procrustes alignment to the mean shape.
    tol : float, optional
        The tolerance for convergence. The function will stop the iterative process when the L2 distance between the new mean shape and the old mean shape is smaller than this value.

    Returns
    -------
    aligned_shapes : list of array_like elements
        List of shapes aligned to the mean shape.
    """   
    mean_shape = procrustes(shape_list[0],shape_list[0])[1]
    aligned_shapes = []
    for shape in shape_list:
        if ex_search:
            aligned_shape = exhaustive_search(shape, mean_shape)[0]
        else:
            aligned_shape = procrustes(shape, mean_shape)[0]
        aligned_shapes.append(aligned_shape)
    mean_shape_new = np.mean(aligned_shapes, axis=0)
    if ex_search:
        mean_shape_new = exhaustive_search(mean_shape_new, mean_shape)[0]
    else:
        mean_shape_new = procrustes(mean_shape_new, mean_shape)[0] 
    while np.sqrt(np.sum(np.square(mean_shape_new - mean_shape))) > tol:
        mean_shape = mean_shape_new
        aligned_shapes = []
        for shape in shape_list:
            if ex_search:
                aligned_shape = exhaustive_search(shape, mean_shape)[0]
            else:
                aligned_shape = procrustes(shape, mean_shape)[0]
            aligned_shapes.append(aligned_shape)
        mean_shape_new = np.mean(aligned_shapes, axis=0)
        if ex_search:
            mean_shape_new = exhaustive_search(mean_shape_new, mean_shape)[0]
        else:
            mean_shape_new = procrustes(mean_shape_new, mean_shape)[0]
    return (aligned_shapes, mean_shape_new)