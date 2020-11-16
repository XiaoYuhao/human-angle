import json
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def evaluate(Y_pred, Y_test):
    normal_label = 0
    normal_true = 0
    patient_label = 0
    patient_true = 0
    for label, pred in zip(Y_test, Y_pred):
        if label == 0:
            normal_label += 1
            if pred == 0:
                normal_true += 1
        elif label == 1:
            patient_label += 1
            if pred == 1:
                patient_true += 1
    
    if patient_label == 0:
        patient = -1
    else:
        patient = patient_true / patient_label
    if normal_label == 0:
        normal = -1
    else:
        normal = normal_true / normal_label
    '''
    print("patient : %f" %(patient))
    print("normal: %f" %(normal))
    print("accuracy : " + str(metrics.accuracy_score(Y_test, Y_pred)))
    print("precision : " + str(metrics.precision_score(Y_test, Y_pred)))
    print("f1 score : " + str(metrics.f1_score(Y_test, Y_pred)))
    '''
    data = {
        "patient" : patient,
        "normal" : normal,
        "accuracy" : metrics.accuracy_score(Y_test, Y_pred),
        "precision" : metrics.precision_score(Y_test, Y_pred),
        "recall": metrics.recall_score(Y_test, Y_pred),
        "f1_score" : metrics.f1_score(Y_test, Y_pred)
    }
    return data


def train1(X_train, Y_train, X_test, Y_test):
    '''
    print("X_train : " + str(len(X_train)))
    print("Y_train : " + str(len(Y_train)))
    print("X_test : " + str(len(X_test)))
    print("Y_test : " + str(len(Y_test)))
    '''
    
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    
    
    data = dict()
    ###
    
    from sklearn.linear_model import LogisticRegression
    clss = LogisticRegression()
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)
    
    print("LogisticRegression :")
    data["LogisticRegression"] = evaluate(Y_pred, Y_test)
    
    ###
    from sklearn.tree import DecisionTreeClassifier
    clss = DecisionTreeClassifier()
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)

    print("DecisionTreeClassifier :")
    data["DecisionTreeClassifier"] = evaluate(Y_pred, Y_test)

    from sklearn.tree import ExtraTreeClassifier
    clss = ExtraTreeClassifier()
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)

    print("ExtraTreeClassifier :")
    data["ExtraTreeClassifier"] = evaluate(Y_pred, Y_test)
    
    # KNeighborsClassifier
    from sklearn.neighbors import KNeighborsClassifier
    clss = KNeighborsClassifier()
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)

    print("KNeighborsClassifier :")
    data["KNeighborsClassifier"] = evaluate(Y_pred, Y_test)

    # RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    clss = RandomForestClassifier()
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)
    
    print("RandomForestClassifier :")
    data["RandomForestClassifier"] = evaluate(Y_pred, Y_test)

    ###
    '''
    # Naive_BayesClassifier
    from sklearn.naive_bayes import MultinomialNB
    clss = MultinomialNB(alpha=0.01)
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)

    print("Naive_BayesClassifier :")
    data["Naive_BayesClassifier"] = evaluate(Y_pred, Y_test)
    '''
    '''
    # SVM Classifier
    from sklearn.svm import SVC
    clss = SVC(kernel='rbf', probability=True)
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)

    print("SVM Classifier :")
    data["SVM Classifier"] = evaluate(Y_pred, Y_test)
    '''
    ###

    # QuadraticDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    clss = QuadraticDiscriminantAnalysis()
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)

    print("QuadraticDiscriminantAnalysis :")
    data["QuadraticDiscriminantAnalysis"] = evaluate(Y_pred, Y_test)

    ####
    '''
    # Linear Discriminant Analysis
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clss = LinearDiscriminantAnalysis()
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)

    print("LinearDiscriminantAnalysis :")
    data["LinearDiscriminantAnalysis"] = evaluate(Y_pred, Y_test)
    
    
    # GaussianNB
    from sklearn.naive_bayes import GaussianNB
    clss = GaussianNB()
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)

    print("GaussianNB :")
    data["GaussianNB"] = evaluate(Y_pred, Y_test)
    '''
    ####

    # AdaBoostClassifier
    from sklearn.ensemble import AdaBoostClassifier
    clss = AdaBoostClassifier()
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)

    print("AdaBoostClassifier :")
    data["AdaBoostClassifier"] = evaluate(Y_pred, Y_test)

    # GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    clss = GradientBoostingClassifier(n_estimators=200)
    clss.fit(X_train, Y_train)

    Y_pred = clss.predict(X_test)

    print("GradientBoostingClassifier :")
    data["GradientBoostingClassifier"] = evaluate(Y_pred, Y_test)

    return data

import xgboost as xgb
import numpy as np
def xgboost_train(X_train, Y_train, X_test, Y_test):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test  = np.array(X_test)
    Y_test  = np.array(Y_test)
    model = xgb.XGBClassifier(max_depth=10,learning_rate=0.05,n_estimators=160,silent=True)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(evaluate(Y_pred, Y_test))

import numpy as np
def corr(pattern):
    with open("dataset/trainset10.json", "r") as j:
        train_data = json.load(j)
    x_names = ["thorax_neck_head_angle", "neck_head_x_angle", "neck_head_y_angle", "neck_head_z_angle", "thorax_neck_x_angle",
               "thorax_neck_y_angle", "thorax_neck_z_angle", "thorax_lshoulder_x_angle", "thorax_rshoulder_x_angle"]
    x_corr = {}
    for x_name in x_names:
        x_angles = []
        y_labels = []
        for data in train_data:
            x_angles.append(data[x_name])
            y_labels.append(data['pattern'][pattern])
        x_corr[x_name] = np.corrcoef(x_angles, y_labels)[1][0]
    print(pattern)
    print(x_corr)

def read_data1():
    X_train = []
    Y_train = []
    X_test  = []
    Y_test  = []

    with open("dataset/train3.json", "r") as j:
        train_data = json.load(j)

    for data in train_data:
        angles = [data["thorax_neck_head_cos"], data["neck_head_x_cos"], data["neck_head_y_cos"], 
                data["neck_head_z_cos"], data["thorax_neck_x_cos"], data["thorax_neck_y_cos"],
                data["thorax_neck_z_cos"], data["thorax_lshoulder_x"], data["thorax_rshoulder_x"]]
        X_train.append(angles)
        patterns = [v for v in data['pattern'].values()]
        Y_train.append(patterns)


    with open("dataset/test3.json", "r") as j:
        test_data = json.load(j)


    for data in test_data:
        angles = [data["thorax_neck_head_cos"], data["neck_head_x_cos"], data["neck_head_y_cos"], 
                data["neck_head_z_cos"], data["thorax_neck_x_cos"], data["thorax_neck_y_cos"],
                data["thorax_neck_z_cos"], data["thorax_lshoulder_x"], data["thorax_rshoulder_x"]]
        X_test.append(angles)
        patterns = [v for v in data['pattern'].values()]
        Y_test.append(patterns)
    
    return X_train, Y_train, X_test, Y_test

def read_data2():
    X_train = []
    Y_train = []
    X_test  = []
    Y_test  = []

    with open("dataset/train7.json", "r") as j:
        train_data = json.load(j)

    for data in train_data:
        angles = [data["thorax_neck_head_angle"], data["neck_head_x_angle"], data["neck_head_y_angle"], 
                data["neck_head_z_angle"], data["thorax_neck_x_angle"], data["thorax_neck_y_angle"],
                data["thorax_neck_z_angle"], data["thorax_lshoulder_x_angle"], data["thorax_rshoulder_x_angle"]]
        X_train.append(angles)
        patterns = [v for v in data['pattern'].values()]
        Y_train.append(patterns)


    with open("dataset/test7.json", "r") as j:
        test_data = json.load(j)


    for data in test_data:
        angles = [data["thorax_neck_head_angle"], data["neck_head_x_angle"], data["neck_head_y_angle"], 
                data["neck_head_z_angle"], data["thorax_neck_x_angle"], data["thorax_neck_y_angle"],
                data["thorax_neck_z_angle"], data["thorax_lshoulder_x_angle"], data["thorax_rshoulder_x_angle"]]
        X_test.append(angles)
        patterns = [v for v in data['pattern'].values()]
        Y_test.append(patterns)

    return X_train, Y_train, X_test, Y_test

def read_data3(pattern):
    X_train = []
    Y_train = []
    X_test  = []
    Y_test  = []

    with open("dataset/trainset10.json", "r") as j:
        train_data = json.load(j)

    for data in train_data:
        angles = [data["thorax_neck_head_angle"], data["neck_head_x_angle"], data["neck_head_y_angle"], 
                data["neck_head_z_angle"], data["thorax_neck_x_angle"], data["thorax_neck_y_angle"],
                data["thorax_neck_z_angle"], data["thorax_lshoulder_x_angle"], data["thorax_rshoulder_x_angle"]]
        X_train.append(angles)
        #patterns = [v for v in data['pattern'].values()]
        patterns = data['pattern'][pattern]
        Y_train.append(patterns)

    with open("dataset/testset10.json", "r") as j:
        test_data = json.load(j)


    for data in test_data:
        angles = [data["thorax_neck_head_angle"], data["neck_head_x_angle"], data["neck_head_y_angle"], 
                data["neck_head_z_angle"], data["thorax_neck_x_angle"], data["thorax_neck_y_angle"],
                data["thorax_neck_z_angle"], data["thorax_lshoulder_x_angle"], data["thorax_rshoulder_x_angle"]]
        X_test.append(angles)
        #patterns = [v for v in data['pattern'].values()]
        patterns = data['pattern'][pattern]
        Y_test.append(patterns)

    return X_train, Y_train, X_test, Y_test

def write_data():
    with open("evaluate.json", "r") as j:
        data = json.load(j)

    with open("evaluate.md", "a") as f:
        keys = list(data.keys())
        vals = list(data.values())
        f.write("\n")
        string = "| pattern"
        string2 = "| ---"
        for clss in vals[0].keys():
            print(clss)
            string += " | " + str(clss)
            string2 += " | ---"
        print(string)
        print(string2)
        f.write(string + "\n")
        f.write(string2 + "\n")
        for i, item in enumerate(vals):
            print(keys[i])
            string = "| " + str(keys[i])
            for score in item.values():
                print(score)
                string += " | " + str("{:.3f}".format(score["accuracy"]))
            f.write(string + "\n")


if __name__ == '__main__':
    '''
    #patterns  = ['head-left-shift', 'head-right-shift', 'head-left-lean', 'head-right-lean', 'left-shoulder-raise', 'right-shoulder-raise']
    patterns  = ["head-shift", "neck-shift", "head-lean", "neck-lean", "shoulder-raise"]
    #patterns  = ["head-flexion", "neck-flexion", "head-backward", "neck-backward", "head-left-shift", "neck-left-shift", "head-left-lean",
    #"neck-left-lean", "left-shoulder-raise", "left-tremble", "head-right-shift", "neck-right-shift", "head-right-lean", "neck-right-lean", "right-shoulder-raise", "right-tremble"]
    data = dict()
    for pattern in patterns:
        X_train, Y_train, X_test, Y_test = read_data3(pattern)
        #print(X_test, Y_test)
        #data[pattern] = train1(X_train, Y_train, X_test, Y_test)
        #print(pattern)
        #xgboost_train(X_train, Y_train, X_test, Y_test)
        #corr(X_train, Y_train)
    
    with open("evaluate.json", "w") as j:
        json.dump(data, j)
    
    write_data()
    '''
    
    
    patterns  = ["head-shift", "neck-shift", "head-lean", "neck-lean", "shoulder-raise"]
    for pattern in patterns:
        corr(pattern)
    