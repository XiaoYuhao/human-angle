import numpy as np
import json, random, math, os
from sklearn.cluster import KMeans 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

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
    #return (math.acos(cos) * (180 / 3.14))
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
    
    '''
    pattern = {
        "head-flexion": 1 if isinstance(table.cell(3,1).value,float) else 0,
        "neck-flexion": 1 if isinstance(table.cell(3,2).value,float) else 0,
        "head-backward": 1 if isinstance(table.cell(3,3).value,float) else 0,
        "neck-backward": 1 if isinstance(table.cell(3,4).value,float) else 0,
        "head-shift": 1 if isinstance(table.cell(3,5).value,float) or isinstance(table.cell(4,5).value,float) else 0,
        "neck-shift": 1 if isinstance(table.cell(3,6).value,float) or isinstance(table.cell(4,6).value,float) else 0,
        "head-lean": 1 if isinstance(table.cell(3,7).value,float) or isinstance(table.cell(4,7).value,float) else 0,
        "neck-lean": 1 if isinstance(table.cell(3,8).value,float) or isinstance(table.cell(4,8).value,float) else 0,
        "shoulder-raise": 1 if isinstance(table.cell(3,9).value,float) or isinstance(table.cell(4,9).value,float) else 0,
        "tremble": 1 if isinstance(table.cell(3,10).value,float) or isinstance(table.cell(4,10).value,float) else 0,
    }
    '''
    #print(pattern)
    return pattern

def select(pred, sample_num):
    mmap = dict()
    print(len(pred))
    #print(pred)
    thresh = int(len(pred) / 20)
    print(thresh)
    for val in pred:
        if val not in mmap.keys():
            mmap[val] = 1
        else:
            mmap[val] += 1
    mmap_sort = sorted(mmap.items(), key = lambda e:e[1], reverse=True)
    select_indexs = []
    #index = [i for i,x in enumerate(pred) if x == mmap_sort[0][0]]
    #select_index = random.sample(index, 1)
    #select_indexs.extend(select_index)
    #sample_num = 1
    
    for item in mmap_sort:
        if (item[1] > thresh or item[1] > 5) and item[1] >= sample_num:
            #print(item[0])   
            index = [i for i,x in enumerate(pred) if x == item[0]]
            select_index = random.sample(index, sample_num)
            #print(select_index)
            select_indexs.extend(select_index)
    
    print(select_indexs)
    return select_indexs

def work(path):
    xlsx_path = 'label/' + path
    data_path = 'data3d/hr_pose_' + path.split('.')[0] + '.npy'
    
    data = np.load(data_path)
    angles = []

    vector_x = [1, 0, 0]
    vector_y = [0, 1, 0]
    vector_z = [0, 0, 1]

    X = []

    pattern = read_excel(xlsx_path)
    for i, keypoint in enumerate(data):
        axis = new_axis(keypoint[12], keypoint[15], keypoint[8])
        points = [keypoint[8], keypoint[9], keypoint[10], keypoint[11], keypoint[14]]
        points = change_axis(axis, points)
        
        vector_8_9 = [points[1][0]-points[0][0], points[1][1]-points[0][1], points[1][2]-points[0][2]]
        vector_9_10 = [points[2][0]-points[1][0], points[2][1]-points[1][1], points[2][2]-points[1][2]]
        vector_8_14 = [points[3][0]-points[0][0], points[3][1]-points[0][1], points[3][2]-points[0][2]]
        vector_8_11 = [points[4][0]-points[0][0], points[4][1]-points[0][1], points[4][2]-points[0][2]]

        frame_angels = [angle(vector_8_9, vector_9_10), angle(vector_9_10, vector_x), angle(vector_9_10, vector_y), angle(vector_9_10, vector_z),
                 angle(vector_8_9, vector_x), angle(vector_8_9, vector_y), angle(vector_8_9, vector_z), angle(vector_8_14, vector_x), angle(vector_8_11, vector_x)]
        X.append(frame_angels)
        frame_data = {
            "thorax_neck_head_angle": frame_angels[0],
            "neck_head_x_angle": frame_angels[1],
            "neck_head_y_angle": frame_angels[2],
            "neck_head_z_angle": frame_angels[3],
            "thorax_neck_x_angle": frame_angels[4],
            "thorax_neck_y_angle": frame_angels[5],
            "thorax_neck_z_angle": frame_angels[6],
            "thorax_lshoulder_x_angle": frame_angels[7],
            "thorax_rshoulder_x_angle": frame_angels[8],
            "pattern": pattern,
            "index": path.split('.')[0] + '[' + str(i) + ']'
        } 
        angles.append(frame_data)
    
    X = np.array(X)
    X = np.reshape(X, (len(data), -1))

    y_pred = KMeans(n_clusters=30, random_state=9).fit_predict(X)
    sample_num = 1 
    select_indexs = select(y_pred, sample_num)
    
    print(data_path)
    return [angles[i] for i in select_indexs]
    #return angles

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
    datasets = ["dataset/trainset11.json", "dataset/testset11.json"]
    for dataset in datasets:
        print(dataset)
        with open(dataset, "r") as j:
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
        
        '''
        head_flexion = 0
        neck_flexion = 0
        head_backward = 0
        neck_backward = 0
        head_shift = 0
        neck_shift = 0
        head_lean = 0
        neck_lean = 0
        shoulder_raise = 0
        tremble = 0
        for obj in objs:
            head_flexion += obj['pattern']["head-flexion"]
            neck_flexion += obj['pattern']["neck-flexion"]
            head_backward += obj['pattern']["head-backward"]
            neck_backward += obj['pattern']["neck-backward"]
            head_shift += obj['pattern']["head-shift"]
            neck_shift += obj['pattern']["neck-shift"]
            head_lean += obj['pattern']["head-lean"]
            neck_lean += obj['pattern']["neck-lean"]
            shoulder_raise += obj['pattern']["shoulder-raise"]
            tremble += obj['pattern']["tremble"]

        counts = {
            "head-flexion": head_flexion,
            "neck-flexion": neck_flexion,
            "head-backward": head_backward,
            "neck-backward": neck_backward,
            "head-shift": head_shift,
            "neck-shift": neck_shift,
            "head-lean": head_lean,
            "neck-lean": neck_lean,
            "shoulder-raise": shoulder_raise,
            "tremble": tremble,
        }
        '''
        print(counts)

if __name__ == '__main__':
    #show_data()
    #paths = list((os.listdir("label")))
    #data = work(paths[2])
    '''
    paths = list((os.listdir("label")))
    datas = []
    for path in paths:
        data = work(path)
        datas.extend(data)
    random.shuffle(datas)
    train_len = int(len(datas)*0.9)
    train_datas = datas[:train_len]
    test_datas = datas[train_len:]
    with open("dataset/trainset2.json", "w") as j:
        json.dump(train_datas, j)

    with open("dataset/testset2.json", "w") as j:
        json.dump(test_datas, j)
    
    print(len(train_datas))
    print(len(test_datas))
    '''
    
    test_paths = ['陈X炜20191205.xlsx', '陈G飞20190711.xlsx', '黄X蓉20181101.xlsx', '丁X蕾20190624.xlsx', '洪W英20181018.xlsx', 
                  '丁X蕾20170907.xlsx', '查M芳20190613.xlsx', '贾D鹏20191031.xlsx', '金G珍20170907.xlsx', '朱X燕20151008.xlsx',
                  '葛B莉20170607.xlsx', '韩X雪20190827.xlsx', '黄X蓉20171026.xlsx', '陈G云20181220.xlsx', '班G君20181108.xlsx' ]
    paths = list((os.listdir("label")))
    train_datas = []
    test_datas = []
    for path in paths:
        data = work(path)
        if path not in test_paths:
            train_datas.extend(data)
        else:
            test_datas.extend(data)
    print(len(train_datas))
    print(len(test_datas))



    with open("dataset/trainset11.json", "w") as j:
        json.dump(train_datas, j)

    with open("dataset/testset11.json", "w") as j:
        json.dump(test_datas, j)
    
    show_data()
    
