#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import radians, cos, sin, asin, sqrt
torch.cuda.is_available()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[2]:


# import zipfile
# import os
# def zip_ya(start_dir):
#     start_dir = start_dir  # 要压缩的文件夹路径
#     file_news = start_dir + '.zip'  # 压缩后文件夹的名字

#     z = zipfile.ZipFile(file_news, 'w', zipfile.ZIP_DEFLATED)
#     for dir_path, dir_names, file_names in os.walk(start_dir):
#         f_path = dir_path.replace(start_dir, '')  # 这一句很重要，不replace的话，就从根目录开始复制
#         f_path = f_path and f_path + os.sep or ''  # 实现当前文件夹以及包含的所有文件的压缩
#         for filename in file_names:
#             z.write(os.path.join(dir_path, filename), f_path + filename)
#     z.close()
#     return file_news
# zip_ya('new_data')


# In[3]:


class Config:
    def __init__(self):
        # Unnamed: 0.1	
#         self.index = 'Unnamed: 0	信息类型	信息来源	目标时间	目标编号	纬度	经度	船舶类型	纬度2	经度2	目标时间2	间隔时间	间隔距离	临时速度	State	SegmentID'.split('	')
        self.index = ['经度','纬度','目标时间']
        self.root_data_path = 'new_data3/'
        self.sequence_length = 6
        self.batch_size = 16
        self.epoch_num = 200  #总共的epoch数量
        self.path_root = ''
        self.num_val  = 2 
        self.line_number = 7  # 相同时间间隔的 点的个数，我们需要用line_number个点预测下一个点
config = Config()
len(config.index)


# In[4]:


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


# In[5]:


def cluster_points(traj,dis_threshold=10,time_threshold=120,invalid_limits=0):
    """
    在规定时间及以下，某物体连续移动的距离没有超过聚类点的距离阈值，
    期间允许出现某几次阈值距离外的畸变点，那么这样的一些点统一可以聚类为一个点
    """
    new_trajectory = pd.DataFrame()
    x = [traj['经度'].values[0]]
    y = [traj['纬度'].values[0]]
#     t = [traj['目标时间'].values[0]]
    
    last_x = x[0]
    last_y = y[0]
#     last_t = t[0]
    
    invalid_times = 0
    for i in range(1, len(traj)):
        cur_x = traj['经度'].values[i]
        cur_y = traj['纬度'].values[i]
#         cur_t = traj['目标时间'].values[i]
        
        if haversine(cur_x, cur_y, last_x, last_y) >= dis_threshold :#or cur_t - last_t >= time_threshold
            invalid_times += 1
            if invalid_times >= invalid_limits:
                last_x = cur_x
                last_y = cur_y
#                 last_t = cur_t
                invalid_times = 0
        x.append(last_x)
        y.append(last_y)
#         t.append(last_t)
    
    new_trajectory['纬度'] = y
    new_trajectory['经度'] = x
#     new_trajectory['目标时间'] = t
    new_trajectory['目标时间'] = traj['目标时间'].values
    return new_trajectory


# In[6]:


def data_proc(s_data):
    s_data = s_data[s_data['经度']<s_data['经度'].mean() + 1]
    s_data = s_data[s_data['纬度']<s_data['纬度'].mean() + 1]

    s_data = s_data[s_data['临时速度'] < 20]
    s_data = cluster_points(s_data)
    return s_data


# In[7]:


def select_good_data(path_,df):
    '''
    for a csv
    :存间隔为110s - 130s的轨迹
    '''
    df_np = df.to_numpy()
    no_ = config.index.index('目标时间') # '目标时间'的第几位的序号
    for i,line in enumerate(df_np):
        time = line[no_]
        if i==0:
            time_list = [time]
            line_list = [line]
            continue
        delta_t =  time - time_list[-1]
        if delta_t > 115:
            if delta_t < 125:
                # 这个点加入 line_list
                time_list.append(time)
                line_list.append(line)
                print("index {} ,len(line_list){}, delta_t {}".format(i+2,len(line_list),delta_t))
            else:  # delta_t > 125
                # 看line_list 有几个数据，需要达到 line_number 个
                # 多于 line_number 个的
                if len(line_list) >= config.line_number:
                    df = pd.DataFrame(line_list,columns=config.index,dtype=float)
                    if not os.path.isdir('new_data2'):
                        os.mkdir('new_data2')
                    df.iloc[:,:].to_csv('new_data2/'+path_.split('.')[0]+'_'+str(i)+'.csv')
                
                # 寻找下一个line_list —— 时间间隔较为均匀的点序列
                time_list = [time]
                line_list = [line]


# In[8]:


def form_data(df,line_number):
    '''
    df：DataFrame:  '纬度' '经度' '目标时间'
    line_number:每个序列的长度
    '''
    df = df.to_numpy()
    new_df_list=[]
    for i in range(line_number,len(df)):
        new_df_list.append(df[i-line_number:i])
    return new_df_list


# In[9]:


x_train_path = 'data'
path = '412043150.csv'
s_data = pd.read_csv(os.path.join(x_train_path,path))
s_data = data_proc(s_data)
s_data


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
x_train_path = 'data/'
tot = 0  # 记录
for i,path in enumerate(os.listdir(x_train_path)):
    print(path)
    s_data = pd.read_csv(os.path.join(x_train_path,path))
    s_data = data_proc(s_data)
    select_good_data(path,s_data)
#     plt.plot(s_data['纬度'],s_data['经度'])
#     plt.show()


# In[11]:


a = '123456'
a.index('1')

