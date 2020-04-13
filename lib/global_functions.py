# Python Libraries #
import datetime as dt
import numpy as np
import math as mth

# OpenCV #
import cv2 as cv

# Written Libraries #
from lib.image import *


def message_print(msg: str):
    """
    Print a console message with current datetime
    :param msg: Message for printing
    :return: nothing
    """
    print(str(dt.datetime.now()) + ":" + msg)


def findMax_2x2(mtrx):
    """
    Find the max value of x and y values on a 2x2 matrix
    :param mtrx: A 2x2 matrix [[x1, y1], [x2, y2]]
    :return: (x_max, y_max) or (None, None)
    """
    if len(mtrx) > 0:
        x_max = mtrx[0][0]  # Set x_max = first X value of mtrx
        y_max = mtrx[0][1]  # Set y_max = first Y value of mtrx

        for m in mtrx:  # For all pair values in mtrx
            if x_max < m[0]:  # If x_max less than X_current value
                x_max = m[0]  # Set new x_max = X_current
            if y_max < m[1]:  # If y_max less than Y_current value
                y_max = m[1]  # Set new y_max = Y_current
        return x_max, y_max  # return x_max and y_max values
    return None, None  # If mtrx has no elements then return None, None


def find_color_list(img: Image(), pts_inlier: []):
    """
    Find the pixel colors of feature points of an image. If image is single band, then use the same pixel value for
    all bands.
    :param img: An Image() object
    :param pts_inlier: The pixel points list (i,j)
    :return: Nothing
    """
    colors = []  # list to store the colors
    img_open = cv.imread(img.src)  # open the image

    img_size = img_open.shape  # Take the shape of the image

    if len(img_size) == 3:  # If shape is 3 then the image has more than 1 bands (we assume the first 3 represent BGR)
        blue = img_open[:, :, 0]  # take blue channel
        green = img_open[:, :, 1]  # take green channel
        red = img_open[:, :, 2]  # take red channel
    else:  # If shape is not 3, then the image is grayscale and set the same image as (blue, green, red)
        blue = img_open  # take blue channel
        green = img_open  # take green channel
        red = img_open  # take red channel

    # ------------------------------------------- #
    # Uncomment the next lines for debugging
    # ------------------------------------------- #
    # cv.imwrite("./blue.jpg", blue)
    # cv.imwrite("./green.jpg", green)
    # cv.imwrite("./red.jpg", red)
    # cv.imwrite("./img.jpg", img_L_open)
    # x, y = findMax_2x2(pts_L_fund)
    # print(x, y)
    # ------------------------------------------- #

    for index in pts_inlier:  # for each index in pts_inlier
        i_L = index[1]  # take the i coord
        j_L = index[0]  # take the j coord
        # print(i_L)
        # print(j_L)
        col_r = red[i_L][j_L]  # find the red pixel color
        col_g = green[i_L][j_L]  # find the green pixel color
        col_b = blue[i_L][j_L]  # find the blue pixel color
        col = [col_r, col_g, col_b]  # store them to list named col
        colors.append(col)  # append the col list to color list
    return colors  # return color list


def export_as_ply(vertices, colors, filename):
    """
    Export a list of vertices as ply
    :param vertices: A list of vertices
    :param colors: The corresponding color for the vertices
    :param filename: The path to the file.ply
    :return: Nothing
    """
    colors = np.array(colors)
    vertices = np.array(vertices)
    colors = colors.reshape(-1, 3)  # Reshape color list
    vertices = vertices.reshape(-1, 3)  # Reshape vertices list
    vertices = np.hstack([vertices, colors])  # Merge vertices and colors

    # Set ply_header
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    with open(filename, 'w') as f:  # While f has something to write
        f.write(ply_header % dict(vert_num=len(vertices)))  # write the file
        np.savetxt(f, vertices, '%f %f %f %d %d %d')  # save the file


def find_scale_parameter(pnt_cloud_1: [], pnt_cloud_2: []):
    points_src = np.array(pnt_cloud_1)
    points_dst = np.array(pnt_cloud_2)
    scale_list = []
    #scale = 0
    #scale_count = 0
    point_list_size = len(points_src)
    if point_list_size > 1000:
        point_list_size = 1000

    for p_id_1 in range(0, point_list_size - 1):
        x1_1 = points_src[p_id_1][0]
        y1_1 = points_src[p_id_1][1]
        z1_1 = points_src[p_id_1][2]

        x2_1 = points_dst[p_id_1][0]
        y2_1 = points_dst[p_id_1][1]
        z2_1 = points_dst[p_id_1][2]

        for p_id_2 in range(p_id_1 + 1, point_list_size):
            x1_2 = points_src[p_id_2][0]
            y1_2 = points_src[p_id_2][1]
            z1_2 = points_src[p_id_2][2]

            x2_2 = points_dst[p_id_2][0]
            y2_2 = points_dst[p_id_2][1]
            z2_2 = points_dst[p_id_2][2]

            dx_1 = float(x1_1 - x1_2)
            dy_1 = float(y1_1 - y1_2)
            dz_1 = float(z1_1 - z1_2)

            dx_2 = float(x2_1 - x2_2)
            dy_2 = float(y2_1 - y2_2)
            dz_2 = float(z2_1 - z2_2)

            dist1: float = mth.sqrt(dx_1 * dx_1 + dy_1 * dy_1 + dz_1 * dz_1)
            dist2: float = mth.sqrt(dx_2 * dx_2 + dy_2 * dy_2 + dz_2 * dz_2)

            if dist2 != 0:
                scale_val: float = dist1 / dist2
                scale_list.append(scale_val)
                #scale += scale_val
                #scale_count += 1

    scale: float = np.mean(scale_list)
    scale_error: float = np.std(scale_list)
    # print(scale)
    # print(scale_error)

    return float(scale), float(scale_error)
