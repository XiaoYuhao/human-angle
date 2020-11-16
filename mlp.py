import torch
import torch.nn as nn
from torch.utils.data import Dataset
'''
class RLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(RLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.hidden = nn.linear(output_size, output_size)
        self._state = 
    
    def forward(self, x, hidden):
        output = self.linear(x)

class RNN(nn.Module):
    def __inin__(self):
        super(RNN, self).__init__()
'''
import logger
log = logger.getLogger('model_hl2_hs256_bs2000_bn')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    
    def forward(self, predicted_scores, labels):
        #print(predicted_scores)
        #print(labels)
        #ones = torch.ones_like(labels)
        #zeros = torch.zeros_like(labels)
        zero_scores = torch.where(labels == 0, predicted_scores, labels)
        one_scores = torch.where(labels >= 1, predicted_scores, labels)
        #print(zero_scores)
        #print(one_scores)
        #ret = (one_scores - labels) ** 2 + torch.abs(zero_scores - labels)
        ret = (one_scores - labels) ** 2 + torch.abs(zero_scores - labels)
        ret = torch.sum(ret)
        return ret
        #loss_sum = torch.zero_
        #for i in range(predicted_scores.shape[0]):
             


class MultiLayerNet(nn.Module):
    def __init__(self, input_size=15, hidden_size=2048, output_size=16, layers=1):
        super(MultiLayerNet, self).__init__()
        hidden_layers = []
        for i in range(layers-1):
            hidden_layers += [nn.Linear(hidden_size, hidden_size),
                              nn.Dropout(p=0.5),
                              nn.ReLU(),
                              nn.BatchNorm1d(hidden_size)]

        self.multilayernet = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            *hidden_layers,
            nn.Linear(hidden_size, output_size),
        )
    

    def forward(self, x):
        y_pred = self.multilayernet(x)
        return y_pred

import os, json, random 

class Dataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, "r") as j:
            self.objects = json.load(j)
        
    def __getitem__(self, i):
        #points = self.objects[i]['thorax'] + self.objects[i]['neck'] + self.objects[i]['head'] + self.objects[i]['lshoulder'] + self.objects[i]['rshoulder']
        points = [self.objects[i]["thorax_neck_head_cos"] , self.objects[i]["neck_head_x_cos"] ,
                 self.objects[i]["neck_head_y_cos"] , self.objects[i]["neck_head_z_cos"] , 
                 self.objects[i]["thorax_neck_x_cos"] , self.objects[i]["thorax_neck_y_cos"] , 
                 self.objects[i]["thorax_neck_z_cos"] , self.objects[i]["thorax_lshoulder_x"] ,
                 self.objects[i]["thorax_rshoulder_x"]]
        points = torch.FloatTensor(points)
        #labels = [v for v in self.objects[i]['pattern'].values()]
        labels = [self.objects[i]['pattern']["head-left-shift"]]
        labels = torch.FloatTensor(labels)

        return points, labels

    def __len__(self):
        return len(self.objects)
    
    def collate_fn(self, batch):
        points = []
        labels = []

        for b in batch:
            points.append(b[0])
            labels.append(b[1])
        
        #print(points)
        #print(labels)
        points = torch.stack(points, dim=0)
        labels = torch.stack(labels, dim=0)

        return (points, labels)

def train(train_loader, model, criterion, optimizer):
    model.train()
    sum_loss = 0
    for i, (points, labels) in enumerate(train_loader):
        points = points.to(device)
        labels = labels.to(device)
            
        predicted_scores = model(points)     
        loss = criterion(predicted_scores, labels)
        sum_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return sum_loss.item()

def validate(val_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        sum_loss = 0
        for i, (points, labels) in enumerate(val_loader):
            points = points.to(device)
            labels = labels.to(device)
            
            predicted_scores = model(points)     
            loss = criterion(predicted_scores, labels)
            sum_loss += loss

    return sum_loss.item()




def main(hidden_layers, hidden_size, batch_size, learning_rate):
    log.debug("hidden_layers=%d, hidden_size=%d, batch_size=%d, learning_rate=%f" %(hidden_layers, hidden_size, batch_size, learning_rate))
    model = MultiLayerNet(input_size=9, hidden_size=hidden_size, output_size=1, layers=hidden_layers)
    #loss_fn = nn.MSELoss(reduction='sum')
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    #learning_rate = 1e-4

    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.0005)
    epoch = 1000
    #batch_size = 1000
    train_dataset = Dataset('dataset/train2.json')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=4, pin_memory=True)
    val_dataset = Dataset('dataset/test2.json')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=val_dataset.collate_fn, num_workers=4, pin_memory=True)

    model = model.to(device)
    loss_data = []
    for t in range(epoch):
        if t == 300:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=0.0005)
        elif t == 600:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5, weight_decay=0.0005)
        sum_loss = train(train_loader, model, criterion, optimizer)
        mean_loss = sum_loss / train_dataset.__len__()
        loss_data.append(mean_loss)
        if t % 20 == 0:
            print("Epoch[%d] loss:%f" %(t, mean_loss))
            log.debug("Epoch[%d] loss:%f" %(t, mean_loss))
            sum_loss = validate(val_loader, model, criterion)
            mean_loss = sum_loss / val_dataset.__len__()
            print("Validate loss:%f" %(mean_loss))
            log.debug("Vaildate loss:%f" %(mean_loss))



    log.debug("training finish.")
    log.debug(" \n \n \n")
    save_model = "weights/model_hl%d_hs%d_bs%d_bn.pth" %(hidden_layers, hidden_size, batch_size)
    torch.save(model.state_dict(), save_model)
    data = {
        'hidden_layers':hidden_layers,
        'hidden_size':hidden_size,
        'batch_size':batch_size,
        'learning_rate':learning_rate,
        'loss':loss_data
    }
    return data

def evaluate():
    model = MultiLayerNet(hidden_size=256, layers=2)
    model.load_state_dict(torch.load('weights/model_hl2_hs256_bs2000_bn.pth'))
    model = model.to(device)
    val_dataset = Dataset('dataset/test2.json')
    points, labels = val_dataset.__getitem__(2)
    points = torch.unsqueeze(points, dim=0).to(device)

    model.eval()
    predicted_scores = model(points)     
    sigmoid = nn.Sigmoid()
    predicted_scores = sigmoid(predicted_scores)
    ones = torch.ones_like(predicted_scores)
    zeros = torch.zeros_like(predicted_scores)
    predicted_lables = torch.where(predicted_scores >= 0.3, ones, zeros)
    print(predicted_scores)
    print(predicted_lables)
    print(labels)


if __name__ == '__main__':
    '''
    datas = []
    for hidden_layers in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for hidden_size in [128, 256, 512, 1048, 2048, 4096]:
            for batch_size in [100, 200, 500, 1000, 2000]:
                for learning_rate in [1e-4, 5e-4, 5e-5]:
                    data = train(hidden_layers=hidden_layers, hidden_size=hidden_size, batch_size=batch_size, learning_rate=learning_rate)
                    datas.append(data)
    
    with open('grid_search_result.json', 'w') as f:
        json.dump(datas, f)
    
    '''
    '''
    datas = []
    for hidden_layers in [5]:
        for hidden_size in [1024]:
            for batch_size in [5, 10, 20, 50, 100, 200, 500, 1000]:
                for learning_rate in [5e-4]:
                    data = train(hidden_layers=hidden_layers, hidden_size=hidden_size, batch_size=batch_size, learning_rate=learning_rate)
                    datas.append(data)
    
    with open('log/grid_search_result_bn_epoch500.json', 'w') as f:
        json.dump(datas, f)
    '''

    main(hidden_layers=1, hidden_size=16, batch_size=2000, learning_rate=1e-4)
    #evaluate()