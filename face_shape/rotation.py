import math
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

def get_indexes(face_landmarks):
    """
    Extracts face parts from a list of detected landmarks. The face parts indexes are taken from Mediapipe
    Input: List of landmarks detected by Mediapipe face landmark task (Numpy array)

    Output: Every face part available from Mediapipe's task (eyes,mouth,nose etc.)

    """



    FACEMESH_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
    FACEMESH_LEFT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362,263, 466, 388, 387, 386, 385, 384, 398, 362)
    FACEMESH_LEFT_EYEBROW = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
    FACEMESH_RIGHT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133 ,33, 246, 161, 160, 159, 158, 157, 173, 133)
    FACEMESH_RIGHT_EYEBROW = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
    FACEMESH_FACE_OVAL = (10, 297, 332, 284 ,251, 389, 356,454 ,323, 361,288, 397,365, 379, 378, 400,377, 152, 148, 176, 149,150, 136, 172, 58, 132, 93, 234 ,127, 162, 21, 54, 103, 67, 109, 10)
    FACEMESH_NOSE = (168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64, 98)
    
    indexes = {}
    indexes['lips'] = face_landmarks[FACEMESH_LIPS, :]
    indexes['left_eye'] = face_landmarks[FACEMESH_LEFT_EYE, :]
    indexes['left_eyebrow'] = face_landmarks[FACEMESH_LEFT_EYEBROW, :]
    indexes['right_eye'] = face_landmarks[FACEMESH_RIGHT_EYE, :]
    indexes['right_eyebrow'] = face_landmarks[FACEMESH_RIGHT_EYEBROW, :]
    indexes['face_oval'] = face_landmarks[FACEMESH_FACE_OVAL, :]
    indexes['nose'] = face_landmarks[FACEMESH_NOSE, :]
    # Return results
    return indexes


def get_plotting_indexes():
    PLOT_LIPS = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
                 (267, 269), (269, 270), (270, 409), (409, 291),(78, 95), (95, 88), (88, 178), (178, 87), (87, 14),(14, 317), (317, 402), (402, 318), (318, 324),(324, 308), (78, 191), (191, 80), (80, 81), 
                 (81, 82),(82, 13), (13, 312), (312, 311), (311, 310),(310, 415), (415, 308)]

    PLOT_LEFT_EYE = [(263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),(386, 385), (385, 384), (384, 398), (398, 362)]

    PLOT_LEFT_EYEBROW = [(276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)]

    PLOT_RIGHT_EYE = [(33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),(159, 158), (158, 157), (157, 173), (173, 133)]

    PLOT_RIGHT_EYEBROW = [(46, 53), (53, 52), (52, 65), (65, 55),(70, 63), (63, 105), (105, 66), (66, 107)]

    PLOT_FACE_OVAL = [(10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400),
                     (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54),
                     (54, 103), (103, 67), (67, 109), (109, 10)]
    PLOT_NOSE = [(168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1), (1, 19), (19, 94), (94, 2), (98, 97), (97, 2), (2, 326), (326, 327), (327, 294), (294, 278), (278, 344), (344, 440), (440, 275),
                 (275, 4), (4, 45), (45, 220), (220, 115), (115, 48), (48, 64), (64, 98)]

    PLOT_FULL_CONTOUR = []
    PLOT_FULL_CONTOUR.extend(PLOT_LIPS)
    PLOT_FULL_CONTOUR.extend(PLOT_LEFT_EYE)
    PLOT_FULL_CONTOUR.extend(PLOT_LEFT_EYEBROW)
    PLOT_FULL_CONTOUR.extend(PLOT_RIGHT_EYE)
    PLOT_FULL_CONTOUR.extend(PLOT_RIGHT_EYEBROW)
    PLOT_FULL_CONTOUR.extend(PLOT_FACE_OVAL)
    PLOT_FULL_CONTOUR.extend(PLOT_NOSE)

    plot_indexes = {}
    plot_indexes['lips'] = PLOT_LIPS
    plot_indexes['left_eye'] = PLOT_LEFT_EYE
    plot_indexes['left_eyebrow'] = PLOT_LEFT_EYEBROW
    plot_indexes['right_eye'] = PLOT_RIGHT_EYE
    plot_indexes['right_eyebrow'] = PLOT_RIGHT_EYEBROW
    plot_indexes['face_oval'] = PLOT_FACE_OVAL
    plot_indexes['nose'] = PLOT_NOSE
    plot_indexes['full_contour'] = PLOT_FULL_CONTOUR
    # Return results
    return plot_indexes

def rotate_point(origin: ArrayLike, point: ArrayLike, angle: float):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = np.asarray(origin)
    px, py = np.asarray(point)
    angle = float(angle)

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def rotate_shape(landmarks: ArrayLike):
    """
    Rotate a shape to best align its (vertical) center line with the y-axis.
    Input: array of shape {N x 2}, where
        N = number of landmarks 
        2 = dimension of data, must be 2.
        
    Output: rotated shape
    """

    # Get face landmarks
    landmarks = np.asarray(landmarks)
    face_parts = get_indexes(landmarks)

    # Calculate the points for the creation of the vertical line that best represents the center of the face
    # ...center of the two eyes
    left_eye_mean = np.mean(face_parts['left_eye'], axis=0)
    right_eye_mean = np.mean(face_parts['right_eye'], axis=0)
    eye_mean = np.mean([left_eye_mean, right_eye_mean], axis=0)
    # ...center of the two eyebrows
    left_eyebrow_mean = np.mean(face_parts['left_eyebrow'], axis=0)
    right_eyebrow_mean = np.mean(face_parts['right_eyebrow'], axis=0)
    eyebrow_mean = np.mean([left_eyebrow_mean, right_eyebrow_mean], axis=0)
    # ...center of the mouth
    mouth_right_tip = face_parts['lips'][np.argmax(face_parts['lips'][:, 0])]
    mouth_left_tip = face_parts['lips'][np.argmin(face_parts['lips'][:, 0])]
    mouth_mean = np.mean([mouth_left_tip, mouth_right_tip], axis=0)
    # ...top and bottom of the nose
    # TODO : Check if current extraction works with Cagdas's landmarks
    nose_top = face_parts['nose'][np.argmin(face_parts['nose'][:, 1])]
    nose_bottom = face_parts['nose'][np.argmax(face_parts['nose'][:, 1])]
    # ...top of the face oval
    indices = np.where(face_parts['face_oval'][:, 1] > face_parts['nose'][np.argmax(face_parts['nose'][:, 1])][1])[0]
    subset = face_parts['face_oval'][indices]
    dists = np.zeros(subset.shape[0])
    for i in range(subset.shape[0]):
        temp = np.abs(subset[i][0] - face_parts['nose'][np.argmax(face_parts['nose'][:, 1])][0])
        dists[i] = temp
    closet_point = np.argmin(dists)
    closet_point = indices[closet_point]
    top_tip = face_parts['face_oval'][closet_point]
    # ...bottom of the face oval
    indices = np.where(face_parts['face_oval'][:, 1] < face_parts['nose'][np.argmin(face_parts['nose'][:, 1])][1])[0]
    subset = face_parts['face_oval'][indices]
    dists = np.zeros(subset.shape[0])
    for i in range(subset.shape[0]):
        temp = np.abs(subset[i][0] - face_parts['nose'][np.argmin(face_parts['nose'][:, 1])][0])
        dists[i] = temp
    closet_point = np.argmin(dists)
    closet_point = indices[closet_point]
    bottom_tip = face_parts['face_oval'][closet_point]

    # Fit a regression line
    reg_points = np.array([eye_mean, eyebrow_mean, mouth_mean, nose_top])
    X, y = reg_points[:, 0].reshape(-1, 1), reg_points[:, 1].reshape(-1, 1)
    reg = LinearRegression().fit(y, X)
    line_y = np.linspace(mouth_mean[1]  , eyebrow_mean[1] , 20).reshape(-1, 1)
    line_x = reg.predict(line_y)
    line = np.hstack((line_x, line_y))

    # Calculate angle between the line and y-axis in radians
    min_point = np.argmin(line[:, 1])
    max_point = np.argmax(line[:, 1])
    deltaY , deltaX = line[max_point][1] - line[min_point][1], line[max_point][0] - line[min_point][0] 
    angle = math.atan2(deltaX, deltaY)
    # Rotate face points
    rotated_shape = np.zeros(landmarks.shape)
    for point in range(landmarks.shape[0]):
        rx, ry = rotate_point((0, 0), landmarks[point, :], angle)
        rotated_shape[point] = rx, ry

    # Return results
    return rotated_shape 



### Auxillary function for plotting faces
def plot_landmarks(shapes_to_plot,
                   connections,
                   color,
                   linewidth,
                   titles):
  """Plot the landmarks and the connections in matplotlib 2d.

  Args:
    landmark_list: A normalized landmark list proto message to be plotted.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected.
    landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
      drawing settings such as color and line thickness.
    connection_drawing_spec: A DrawingSpec object that specifies the
      connections' drawing settings such as color and line thickness.
    elevation: The elevation from which to view the plot.
    azimuth: the azimuth angle to rotate the plot.

  Raises:
    ValueError: If any connection contains an invalid landmark index.
  """
  n_shapes = shapes_to_plot.shape[0]
  if n_shapes != len(titles):
    raise ValueError('Number of titles should be equal to the number of shapes to plot')
  

  if n_shapes == 1:
    fig,ax = plt.subplots(ncols = 1 , nrows = 1 , figsize = (8,8))
  elif n_shapes ==2:
    fig,ax = plt.subplots(ncols = 2 , nrows = 1 , figsize = (12,6))
  else:
     fig,ax = plt.subplots(ncols = n_shapes , nrows = 1 , figsize = (18,6))

  if n_shapes == 1:
    for i,shape in enumerate(shapes_to_plot):
      plotted_landmarks = {}
      for idx, landmark in enumerate(shape):
        plotted_landmarks[idx] = (landmark[0], landmark[1])
      if connections:
        num_landmarks = len(shape)
      # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
          start_idx = connection[0]
          end_idx = connection[1]
          if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
            raise ValueError(f'Landmark index is out of range. Invalid connection '
                          f'from landmark #{start_idx} to landmark #{end_idx}.')
          if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
            landmark_pair = [
                plotted_landmarks[start_idx], plotted_landmarks[end_idx]
            ]
            ax.plot(
              [landmark_pair[0][0], landmark_pair[1][0]],
              [landmark_pair[0][1], landmark_pair[1][1]],
              color = color,
              linewidth=linewidth)
          
      ax.set_title(titles[i])
      ax.invert_yaxis()
  else:
    for i,shape in enumerate(shapes_to_plot):
      plotted_landmarks = {}
      for idx, landmark in enumerate(shape):
        plotted_landmarks[idx] = (landmark[0], landmark[1])
      if connections:
        num_landmarks = len(shape)
      # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
          start_idx = connection[0]
          end_idx = connection[1]
          if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
            raise ValueError(f'Landmark index is out of range. Invalid connection '
                          f'from landmark #{start_idx} to landmark #{end_idx}.')
          if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
            landmark_pair = [
                plotted_landmarks[start_idx], plotted_landmarks[end_idx]
            ]
            ax[i].plot(
              [landmark_pair[0][0], landmark_pair[1][0]],
              [landmark_pair[0][1], landmark_pair[1][1]],
              color = color,
              linewidth=linewidth)
          
      ax[i].set_title(titles[i])
      ax[i].invert_yaxis()
  
  plt.show()