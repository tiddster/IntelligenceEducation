import torch
import random
import numpy as np
import pandas as pd
from torch.utils import data

batch_size = 512

train_path = "P:\Dataset\DKT\\train.csv"
train_split_path = "P:\Dataset\DKT\\train_split.csv"
test_path = "P:\Dataset\DKT\\test.csv"

def split_data(f, radio):
    data_file = pd.read_csv(f)
    data_len = len(data_file)

    # 特征与标签
    stu_id = torch.Tensor(data_file['student'].values).reshape(data_len,1).long()
    question_id = torch.Tensor(data_file['question'].values).reshape(data_len,1).long()
    skill_id = torch.Tensor(data_file['skill'].values).reshape(data_len,1).long()
    labels = torch.Tensor(data_file['correctness'].values).reshape(data_len,1).long()
    data = torch.cat((stu_id,question_id,skill_id,labels),1)
    data = data.numpy()
    indices = list(range(data_len))
    random.shuffle(indices)
    train_indices = indices[0:int(data_len*(1-radio))]
    test_indices = indices[int(data_len*(1-radio))+1:-1]
    np.savetxt('train_split.csv', data[train_indices], header="student,question,skill,correctness",delimiter=',',fmt="%d",comments='')
    np.savetxt('test_split.csv', data[test_indices], header="student,question,skill,correctness",delimiter=',', fmt="%d",comments='')


split_data(train_path,0.3)

# 加载训练集特征和标签，并组合成数据集
def load_data(f, batch_size, is_train=True):
    data_file = pd.read_csv(f)
    data_len = len(data_file)

    # 特征与标签
    stu_id = torch.Tensor(data_file['student'].values).reshape(data_len, 1).long()
    question_id = torch.Tensor(data_file['question'].values).reshape(data_len, 1).long()
    skill_id = torch.Tensor(data_file['skill'].values).reshape(data_len, 1).long()
    features = torch.cat((stu_id, question_id, skill_id), 1)
    labels = torch.Tensor(data_file['correctness'].values).long()

    dataset = data.TensorDataset(features, labels)
    return data.DataLoader(dataset, batch_size, shuffle=is_train), features, labels


data_iter, features, labels = load_data('train_split.csv', batch_size)


# 目标函数:做对的概率,abi:能力矩阵,dif:难度矩阵,dis:区分度矩阵,gue:猜度矩阵
def probability(abi, dif, dis, gue, X):
    D = 1.702
    stu_id = X[:, 0]
    que_id = X[:, 1]
    ski_id = X[:, 2]
    return gue[0, que_id] + (1 - gue[0, que_id]) / \
           (1 + torch.exp(-D * dis[0, que_id] * (abi[stu_id, ski_id] - dif[que_id, ski_id])))


# 初始化四个权重矩阵，能力、难度、区分度、猜度
ability = torch.rand(1080, 32, requires_grad=True)
dif = torch.zeros(609, 32, requires_grad=False)
dis = torch.ones((1, 609), requires_grad=True)
gue = torch.full(size=(1, 609), fill_value=0.25, requires_grad=True)

# 对于难度矩阵进行约束
for i in range(len(features)):
    dif[features[i, 1], features[i, 2]] = 1
dif.requires_grad_(True)


# 损失函数，交叉熵损失函数
def cross_entropy(y_hat, y):
    return -(torch.log(y_hat) * y + torch.log((1 - y_hat)) * (1 - y))


# 计算在训练集上的精度
def acc(y, y_hat):
    y_hat = torch.where(y_hat > 0.5, 1, 0)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) / len(y)


num_epochs = 60
net = probability
loss = cross_entropy
trainer = torch.optim.Adam(params=(ability, dif, dis, gue))  # 优化函数

# 训练
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(ability, dif, dis, gue, X), y)
        trainer.zero_grad()
        l.sum().backward()
        trainer.step()
    with torch.no_grad():
        train_l = loss(net(ability, dif, dis, gue, features), labels)
        train_acc = acc(labels, net(ability, dif, dis, gue, features))
        print(f'epoch:{epoch + 1},loss{float(train_l.mean()):f},acc{train_acc}')


# 加载测试集特征
def load_data2(f):
    data_file = pd.read_csv(f)
    data_len = len(data_file)

    stu_id = torch.Tensor(data_file['student'].values).reshape(data_len, 1).long()
    question_id = torch.Tensor(data_file['question'].values).reshape(data_len, 1).long()
    skill_id = torch.Tensor(data_file['skill'].values).reshape(data_len, 1).long()
    features = torch.cat((stu_id, question_id, skill_id), 1)

    return features


# 测试集特征
features_test = load_data2("test_split.csv")

# 预测
p_all = probability(ability, dif, dis, gue, features_test)
p_all = p_all.detach().numpy()
for i in range(len(p_all)):
    if p_all[i] < 0:
        p_all[i] = random.uniform(0, 0.5)
p_all = np.round(p_all, 4)
np.savetxt('p.csv', p_all, delimiter=',', fmt='%.04f')
