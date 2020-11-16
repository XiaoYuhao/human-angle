import numpy as np
import json, random, math

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

    return axis_matrix                  #返回由原坐标系转换到新坐标系的变换矩阵

def angle(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    len1 = np.sqrt(vector1.dot(vector1))
    len2 = np.sqrt(vector2.dot(vector2))

    dot = vector1.dot(vector2)

    cos = dot/(len1*len2)
    return math.acos(cos)


def change_axis(axis_matrix, points):
    axis_matrix_tran = np.linalg.inv(axis_matrix)       #矩阵求逆
    points = np.array(points).transpose()
    ones = np.array([1 for i in range(points.shape[1])])
    points = np.row_stack((points, ones))               #添加齐次项

    new_points = np.matmul(axis_matrix_tran, points)    #坐标乘上坐标系的逆矩阵，将原坐标转换成新坐标系的坐标
    new_points = new_points.transpose()
    new_points = np.delete(new_points, 3, axis=1)       #去掉齐次项

    points = new_points.tolist()
    return points

import xlrd

def read_excel(file_path):
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    '''
    pattern = {
        "head-flexion": table.cell(3,1).value if isinstance(table.cell(3,1).value,float) else 0.0,
        "neck-flexion": table.cell(3,2).value if isinstance(table.cell(3,2).value,float) else 0.0,
        "head-backward": table.cell(3,3).value if isinstance(table.cell(3,3).value,float) else 0.0,
        "neck-backward": table.cell(3,4).value if isinstance(table.cell(3,4).value,float) else 0.0,
        "head-left-shift": table.cell(3,5).value if isinstance(table.cell(3,5).value,float) else 0.0,
        "neck-left-shift": table.cell(3,6).value if isinstance(table.cell(3,6).value,float) else 0.0,
        "head-left-lean": table.cell(3,7).value if isinstance(table.cell(3,7).value,float) else 0.0,
        "neck-left-lean": table.cell(3,8).value if isinstance(table.cell(3,8).value,float) else 0.0,
        "left-shoulder-raise": table.cell(3,9).value if isinstance(table.cell(3,9).value,float) else 0.0,
        "left-tremble": table.cell(3,10).value if isinstance(table.cell(3,10).value,float) else 0.0,
        "head-right-shift": table.cell(4,5).value if isinstance(table.cell(4,5).value,float) else 0.0,
        "neck-right-shift": table.cell(4,6).value if isinstance(table.cell(4,6).value,float) else 0.0,
        "head-right-lean": table.cell(4,7).value if isinstance(table.cell(4,7).value,float) else 0.0,
        "neck-right-lean": table.cell(4,8).value if isinstance(table.cell(4,8).value,float) else 0.0,
        "right-shoulder-raise": table.cell(4,9).value if isinstance(table.cell(4,9).value,float) else 0.0,
        "right-tremble": table.cell(4,10).value if isinstance(table.cell(4,10).value,float) else 0.0
    }
    '''
    pattern = {
        "head-flexion": 1 if isinstance(table.cell(3,1).value,float) else 0,
        "neck-flexion": 1 if isinstance(table.cell(3,2).value,float) else 0,
        "head-backward": 1 if isinstance(table.cell(3,3).value,float) else 0,
        "neck-backward": 1 if isinstance(table.cell(3,4).value,float) else 0,
        "head-left-shift": 1 if isinstance(table.cell(3,5).value,float) else 0,
        "neck-left-shift": 1 if isinstance(table.cell(3,6).value,float) else 0,
        "head-left-lean": 1 if isinstance(table.cell(3,7).value,float) else 0,
        "neck-left-lean": 1 if isinstance(table.cell(3,8).value,float) else 0,
        "left-shoulder-raise": 1 if isinstance(table.cell(3,9).value,float) else 0,
        "left-tremble": 1 if isinstance(table.cell(3,10).value,float) else 0,
        "head-right-shift": 1 if isinstance(table.cell(4,5).value,float) else 0,
        "neck-right-shift": 1 if isinstance(table.cell(4,6).value,float) else 0,
        "head-right-lean": 1 if isinstance(table.cell(4,7).value,float) else 0,
        "neck-right-lean": 1 if isinstance(table.cell(4,8).value,float) else 0,
        "right-shoulder-raise": 1 if isinstance(table.cell(4,9).value,float) else 0,
        "right-tremble": 1 if isinstance(table.cell(4,10).value,float) else 0
    }
    #print(pattern)
    return pattern

import os

def work(path):
    xlsx_path = 'label/' + path
    data_path = 'data3d/hr_pose_' + path.split('.')[0] + '.npy'
    save_path = "dataset/" + path.split('.')[0] + '.json'
    if os.path.exists(save_path):
        print(save_path)
        return None
    #print(xlsx_path)
    #print(data_path) 

    data = np.load(data_path)
    keypoints = []

    for keypoint in data:
        #print(keypoint.shape)
        axis = new_axis(keypoint[12], keypoint[15], keypoint[8])
        points = [keypoint[8], keypoint[9], keypoint[10], keypoint[11], keypoint[14]]
        points = change_axis(axis, points)
        frame_data = {
            "thorax": points[0],
            "neck": points[1],
            "head": points[2],
            "lshoulder": points[3],
            "rshoulder": points[4],
            "pattern": read_excel(xlsx_path)
        }
        keypoints.append(frame_data)

    with open(save_path, 'w') as f:
        json.dump(keypoints, f)
    print(save_path)

def work2(path):
    xlsx_path = 'label/' + path
    data_path = 'data3d/hr_pose_' + path.split('.')[0] + '.npy'
    
    data = np.load(data_path)
    angles = []

    vector_x = [1, 0, 0]
    vector_y = [0, 1, 0]
    vector_z = [0, 0, 1]

    for keypoint in data:
        axis = new_axis(keypoint[12], keypoint[15], keypoint[8])
        points = [keypoint[8], keypoint[9], keypoint[10], keypoint[11], keypoint[14]]
        points = change_axis(axis, points)

        vector_8_9 = [points[1][0]-points[0][0], points[1][1]-points[0][1], points[1][2]-points[0][2]]
        vector_9_10 = [points[2][0]-points[1][0], points[2][1]-points[1][1], points[2][2]-points[1][2]]
        vector_8_14 = [points[3][0]-points[0][0], points[3][1]-points[0][1], points[3][2]-points[0][2]]
        vector_8_11 = [points[4][0]-points[0][0], points[4][1]-points[0][1], points[4][2]-points[0][2]]

        frame_data = {
            "thorax_neck_head_angle": angle(vector_8_9, vector_9_10),
            "neck_head_x_angle": angle(vector_9_10, vector_x),
            "neck_head_y_angle": angle(vector_9_10, vector_y),
            "neck_head_z_angle": angle(vector_9_10, vector_z),
            "thorax_neck_x_angle": angle(vector_8_9, vector_x),
            "thorax_neck_y_angle": angle(vector_8_9, vector_y),
            "thorax_neck_z_angle": angle(vector_8_9, vector_z),
            "thorax_lshoulder_x_angle": angle(vector_8_14, vector_x),
            "thorax_rshoulder_x_angle": angle(vector_8_11, vector_x),
            "pattern": read_excel(xlsx_path)
        } 
        angles.append(frame_data)
    print(data_path)
    return angles

def make_dataset():
    paths = list((os.listdir("dataset/data")))
    test_paths = ['查M芳20171221.json', '陈G飞20190711.json', '黄X蓉20181101.json', '丁X平20190606.json', '丁X蕾20190624.json', '金G珍20180913.json']
    
    train_objs = []
    for path in paths:
        if path in test_paths:
            continue
        with open(os.path.join("dataset/data", path), "r") as j:
            objs = json.load(j)
        train_objs.extend(objs)
        print(path)
    
    with open("dataset/train.json", "w") as j:
        json.dump(train_objs, j)
    
    test_objs = []
    for path in test_paths:
        with open(os.path.join("dataset/data", path), "r") as j:
            objs = json.load(j)
        test_objs.extend(objs)
        print(path)
    
    with open("dataset/test.json", "w") as j:
        json.dump(test_objs, j)

def show_data():
    with open("dataset/train.json", "r") as j:
        objs = json.load(j)
    head_flexion = 0
    neck_flexion = 0
    head_backward = 0
    neck_backward = 0
    head_left_shift = 0
    neck_left_shift = 0
    head_left_lean = 0
    neck_left_lean = 0
    left_shoulder_rais = 0
    left_tremble = 0
    head_right_shift = 0
    neck_right_shift = 0
    head_right_lean = 0
    neck_right_lean = 0
    right_shoulder_rais = 0
    right_tremble = 0
    for obj in objs:
        head_flexion += obj['pattern']["head-flexion"]
        neck_flexion += obj['pattern']["neck-flexion"]
        head_backward += obj['pattern']["head-backward"]
        neck_backward += obj['pattern']["neck-backward"]
        head_left_shift += obj['pattern']["head-left-shift"]
        neck_left_shift += obj['pattern']["neck-left-shift"]
        head_left_lean += obj['pattern']["head-left-lean"]
        neck_left_lean += obj['pattern']["neck-left-lean"]
        left_shoulder_rais += obj['pattern']["left-shoulder-raise"]
        left_tremble += obj['pattern']["left-tremble"]
        head_right_shift += obj['pattern']["head-right-shift"]
        neck_right_shift += obj['pattern']["neck-right-shift"]
        head_right_lean += obj['pattern']["head-right-lean"]
        neck_right_lean += obj['pattern']["neck-right-lean"]
        right_shoulder_rais += obj['pattern']["right-shoulder-raise"]
        right_tremble += obj['pattern']["right-tremble"]
    
    counts = {
        "head-flexion": head_flexion,
        "neck-flexion": neck_flexion,
        "head-backward": head_backward,
        "neck-backward": neck_backward,
        "head-left-shift": head_left_shift,
        "neck-left-shift": neck_left_shift,
        "head-left-lean": head_left_lean,
        "neck-left-lean": neck_left_lean,
        "left-shoulder-raise": left_shoulder_rais,
        "left-tremble": left_tremble,
        "head-right-shift": head_right_shift,
        "neck-right-shift": neck_right_shift,
        "head-right-lean": head_right_lean,
        "neck-right-lean": neck_right_lean,
        "right-shoulder-raise": right_shoulder_rais,
        "right-tremble": right_tremble
    }
    print(counts)


from sklearn.cluster import KMeans 
from sklearn import metrics

if __name__ == '__main__':
    #make_dataset()
    #show_data()
    #test_paths = ['查M芳20171221.xlsx', '陈G飞20190711.xlsx', '黄X蓉20181101.xlsx', '丁X平20190606.xlsx', '丁X蕾20190624.xlsx', '金G珍20180913.xlsx']
    paths = list((os.listdir("label")))
    #work(paths[0])
    datas = []
    for path in paths:
        data = work2(path)
        datas.extend(data)
    random.shuffle(datas)
    train_len = int(len(datas)*0.9)
    train_datas = datas[:train_len]
    test_datas = datas[train_len:]
    with open("dataset/train3.json", "w") as j:
        json.dump(train_datas, j)

    with open("dataset/test3.json", "w") as j:
        json.dump(test_datas, j)

    '''
    data_path = 'outputs/hr_pose_丁X蕾20190624.npy'
    data = np.load(data_path)
    keypoints = []
    X = []
    for keypoint in data:
        #print(keypoint.shape)
        axis = new_axis(keypoint[12], keypoint[15], keypoint[8])
        points = [keypoint[9], keypoint[10], keypoint[11], keypoint[14]]
        points = change_axis(axis, points)
        X.append(points)
        frame_data = {
            "neck":points[0],
            "head":points[1],
            "lshoulder":points[2],
            "rshoulder":points[3]
        }
        keypoints.append(frame_data)
    
    X = np.array(X)
    X = np.reshape(X, (len(data),-1))
   
    y_pred = KMeans(n_clusters=5, random_state=9).fit_predict(X)
    print(len(y_pred))
    print(y_pred)
    score = metrics.calinski_harabasz_score(X, y_pred)
    print(score)
    '''
                             



    '''
    point0 = [1, 0, 0]
    point1 = [0, 1, 0]
    point2 = [0, 0, 1]
    axis = new_axis(point0, point1, point2)
    point = [0.5, 0.5, 0]
    points = [[0.5, 0.5, 0],[1, 1, 0],[2, 2, 0]] 
    points = change_axis(axis, points)
    print(points)
    '''


    '''
    "thorax_neck_head_cos": 
    "neck_head_x_cos": 
    "neck_head_y_cos": 
    "neck_head_z_cos": 
    "thorax_neck_x_cos": 
    "thorax_neck_y_cos": 
    "thorax_neck_z_cos": 
    "thorax_lshoulder_x":
    "thorax_rshoulder_x":
    "pattern":
    =======================
    "head-flexion":         thorax_neck_head_angle < thresh
    "neck-flexion":         thorax_neck_y_angle < thresh
    "head-backward":        thorax_neck_head_angle > thresh   
    "neck-backward":        thorax_neck_y_angle > thresh
    "head-left-shift":  
    "neck-left-shift": 
    "head-left-lean":       neck_head_x_angle < thresh
    "neck-left-lean":       thorax_neck_x_angle < thresh
    "left-shoulder-raise":  thorax_lshoulder_x_angle > thresh
    "left-tremble": 
    "head-right-shift": 
    "neck-right-shift": 
    "head-right-lean":      neck_head_x_angle > thresh
    "neck-right-lean":      thorax_neck_x_angle > thresh
    "right-shoulder-raise": thorax_rshoulder_x_angle < thresh
    "right-tremble": 
    
    '''