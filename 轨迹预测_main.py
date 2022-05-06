#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# https://blog.csdn.net/Muzi_Water/article/details/103921115


# In[2]:


class Config:
    def __init__(self):
        # Unnamed: 0.1	
#         self.index = 'Unnamed: 0	信息类型	信息来源	目标时间	目标编号	纬度	经度	船舶类型	纬度2	经度2	目标时间2	间隔时间	间隔距离	临时速度	State	SegmentID'.split('	')
        self.index = ['','纬度','经度','目标时间']
        self.root_data_path = 'new_data2/'
        self.batch_size = 16
        self.epoch_num = 200  #总共的epoch数量
        self.path_root = ''
        self.num_val  = 2 
        self.line_number = 7
config = Config()


# In[3]:


from math import radians, cos, sin, asin, sqrt
 
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371.393 # 地球平均半径，单位为公里
    return c * r * 1000.0 # 单位为m


# In[4]:


def form_data(df,line_number):
    '''
    df：DataFrame:  '经度' '纬度' '目标时间'
    line_number:每个序列的长度
    '''
    df = df.to_numpy()
    new_df_list=[]
    for i in range(line_number,len(df)):
        new_df_list.append(df[i-line_number:i])
    return new_df_list


# In[5]:


data_list = []
no_1 = config.index.index('纬度')  # '纬度'的第几位的序号
no_2 = config.index.index('经度')  # '经度'的第几位的序号
tot = 0
for path in os.listdir(config.root_data_path):
    s_data = pd.read_csv(os.path.join(config.root_data_path,path))
    tot += len(s_data)
    s_data_list = form_data(s_data,config.line_number)
    for s_data in s_data_list:
        data_list.extend(s_data[:,no_1:no_2+1])
data_list[0]


# In[6]:


tot - len(os.listdir(config.root_data_path))*config.line_number


# In[7]:


Scaler = preprocessing.MinMaxScaler()
data_list = Scaler.fit_transform(data_list)


# In[8]:


data_list[0]


# In[9]:


x_list = []
y_list = []
temp_list = []
for data in data_list:
    temp_list.append(data)
    if len(temp_list) ==config.line_number:
        x_list.append(temp_list[0:-1])
        y_list.append(temp_list[-1])
        temp_list = []


# In[10]:


x_dim = len(x_list[0][0])
y_dim = len(y_list[0])
print(x_dim,y_dim)


# In[11]:


np.array(x_list).shape,np.array(y_list).shape


# In[12]:


X_train, X_val, y_train, y_val = train_test_split(x_list,y_list, test_size = 0.2,random_state = 10)


# In[13]:


torch.tensor(X_train[0]).cuda()


# In[14]:


class LineDataSet(Dataset):
    def __init__(self,data,label):
#         data_path = 'new_data3/'
        self.data = data
        self.label = label
    def __getitem__(self, index):
        return torch.tensor(self.data[index]).to(torch.float32),torch.tensor(self.label[index]).to(torch.float32)
    def __len__(self):
        return len(self.data)
train_dataset = LineDataSet(X_train,y_train)
train_loader = DataLoader(train_dataset,batch_size=config.batch_size)
val_dataset = LineDataSet(X_val,y_val)
val_loader = DataLoader(val_dataset,batch_size=config.batch_size)


# In[15]:


input_size = x_dim
hidden_size = 10
output_size = 2


# In[16]:


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first =True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
#         print(x.shape)
        out, h_state = self.rnn(x)
#         print('out1',out.shape)
        out = self.linear(out) #=> out[batch_size*seq,hidden_size] --> [batch_size*seq,output_size]
#         print('out3',out.shape)
        return out[:,-1]


# In[17]:


from tqdm import tqdm
def train(model,loader,epoch_num,epoch_no):
    print('start training')
    model.train()
    total_loss = 0.0
    t = tqdm(loader)
    for i,(data,label) in enumerate(t):
#         print('data',data)
#         print('label',label)
        if torch.cuda.is_available():
            label = label.cuda()
            data = data.cuda()
        model.zero_grad()
        out = model(data)
#         print('out',out.shape)
#         print('label',label.shape)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data.item()
    
    train_avg_loss = total_loss/len(loader.dataset)
    print('Epoch{}/{}: train loss: {} train score{}'.format(epoch_no,epoch_num+start_epoch,train_avg_loss,1/train_avg_loss))


# In[18]:


def validation(model, loader, epoch):
    print('start validation')
    model.eval()
    total_loss = 0.0
    global best_score
    with torch.no_grad():
        t = tqdm(loader)
        for index, (data, label) in enumerate(t):
            if torch.cuda.is_available():
                label = label.cuda()
                data = data.cuda()
            out = model(data)
            loss = criterion(out, label)
            # print('out shape',out.shape)
            # print('out',out)
            total_loss += loss.data.item()
    
    val_avg_loss = total_loss/len(loader.dataset)
    score = 1/val_avg_loss
    if score > best_score:
        print("Saving.....")
        state = {
            'model':model.state_dict(),
            'score': score,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state,'./checkpoint/ckpt.pth')
        best_score = score
    print("Test score {:.6f}".format(score))


# In[ ]:


model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
start_epoch = 0
best_score = 0.0 #用loss的倒数来表示score
resume = True
if resume:
    if not os.path.isdir(config.path_root+'checkpoint'):
        print('Error: no checkpoint directory found!')
        start_epoch = 0
        best_score = 0.0 #用loss的倒数来表示score
    else:
        # load model
        print("==> resume form checkpoint......")

        checkpoint = torch.load(config.path_root + './checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        best_score = checkpoint['score']
        start_epoch = checkpoint['epoch']
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), 1e-2)
print('start epoch', start_epoch)
for i in range(start_epoch, start_epoch + config.epoch_num):
    train(model=model, loader=train_loader,epoch_no=i,epoch_num=config.epoch_num)
    if i % config.num_val == 0:
        validation(model=model, loader=val_loader,epoch=i)


# In[ ]:


def predict(model, loader):
    t = tqdm(loader)
    distance = 0
    for index, (data, label) in enumerate(t):
        pred = model(data.cuda())
        
        pred_lon1,pred_lat1 = Scaler.inverse_transform(pred.detach().cpu().numpy())[:,0],                            Scaler.inverse_transform(pred.detach().cpu().numpy())[:,1]
        
        label_lon2,label_lat2 = Scaler.inverse_transform(label.detach().cpu().numpy())[:,0]                                ,Scaler.inverse_transform(label.detach().cpu().numpy())[:,1]
        dis = 0
        for i in range(len(pred_lon1)):
            dis+=haversine(pred_lon1[i], pred_lat1[i], label_lon2[i], label_lat2[i])
        distance += dis
    return distance/len(loader.dataset)


# In[ ]:


predict(model,val_loader)


# In[ ]:




