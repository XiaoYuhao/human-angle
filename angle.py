import math
import numpy as np

def new_axis(point0, point1, point2):
    x0 = point0[0]
    y0 = point0[1]
    z0 = point0[2]

    x1 = point1[0]
    y1 = point1[1]
    z1 = point1[2]

    x2 = point2[0]
    y2 = point2[1]
    z2 = point2[2]

    A = np.array([[x1-x0, y1-y0, z1-z0],
                  [y1-y0, x0-x1, 0],
                  [z1-z0, 0, x0-x1]])
    b = np.transpose(np.array([x2*(x1-x0)+y2*(y1-y0)+z2*(z1-z0), x0*y1-x1*y0, x0*z1-x1*z0]))

    x = np.linalg.solve(A, b)

    origin_x = x[0]
    origin_y = x[1]
    origin_z = x[2]

    vec_x = np.array([x0-origin_x, y0-origin_y, z0-origin_z])
    vec_z = np.array([x2-origin_x, y2-origin_y, z2-origin_z])
    vec_y = np.cross(vec_z, vec_x)      #注意，根据右手法则，叉积时z轴向量在前

    norm_x = np.linalg.norm(vec_x)
    norm_y = np.linalg.norm(vec_y)
    norm_z = np.linalg.norm(vec_z)
    vec_x = vec_x / norm_x              #需要将坐标轴向量单位化
    vec_y = vec_y / norm_y
    vec_z = vec_z / norm_z

    axis_matrix = np.array([[vec_x[0], vec_y[0], vec_z[0], origin_x],
                            [vec_x[1], vec_y[1], vec_z[1], origin_y],
                            [vec_x[2], vec_y[2], vec_z[2], origin_z],
                            [0, 0, 0, 1]])

    print(axis_matrix)
    return axis_matrix                  #返回由原坐标系转换到新坐标系的变换矩阵



if __name__ == '__main__':
    point0 = [1, 0, 0]
    point1 = [0, 1, 0]
    point2 = [0, 0, 1]
    new_axis(point0, point1, point2)
