import cv2 as cv
import datetime as dt


def message_print(msg: str):
    print(str(dt.datetime.now()) + ":" + msg)


def findMax_2x2(mtrx):
    """
    Find the max value of x and y values on a 2x2 matrix
    :param mtrx: A 2x2 matrix [[x1, y1], [x2, y2]]
    :return: x_max, y_max
    """
    x_max = 0
    y_max = 0

    for m in mtrx:
        if x_max < m[0]:
            x_max = m[0]
        if y_max < m[1]:
            y_max = m[1]
    return x_max, y_max


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

    img_size = img_open.shape

    if len(img_size) == 3:
        blue = img_open[:, :, 0]  # take blue channel
        green = img_open[:, :, 1]  # take green channel
        red = img_open[:, :, 2]  # take red channel
    else:
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
    return colors


def export_as_ply(vertices, colors, filename):
    """
    Export a list of vertices as ply
    :param vertices: A list of vertices
    :param colors:  The corresponding color for the vertices
    :param filename: The path to the file.ply
    :return: Nothing
    """
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

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
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')
