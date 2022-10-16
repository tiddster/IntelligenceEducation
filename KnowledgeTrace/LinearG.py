import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

train_path = "D:\新建文件夹\Dataset\DKT\\train.csv"
final_path = "D:\新建文件夹\Dataset\DKT\\test.csv"

skill_set = list(range(32))

train_data = pd.read_csv(train_path)[:38077]
final_data = pd.read_csv(final_path)

final_student_list = np.asarray(list(final_data["student"])).reshape((-1, 1))
final_questionList =  np.asarray(list(final_data["question"])).reshape((-1, 1))
final_skill_list = list(final_data["skill"])

stuList = list(train_data["student"])
studentList = np.asarray(stuList).reshape((-1, 1))
skillList = list(train_data["skill"])
questionList = np.asarray(list(train_data["question"])).reshape((-1, 1))
correctList = list(train_data["correctness"])

stu_TF = np.zeros((len(skillList), 2))
final_TF = np.zeros((len(list(final_data["student"])), 2))

for i, (s, c) in enumerate(zip(stuList, correctList)):
    if i != 0 and stuList[i-1] == stuList[i]:
        stu_TF[i] = stu_TF[i - 1]
    if c == 0:
        stu_TF[i][1] += 1
    else:
        stu_TF[i][0] += 1
    final_TF[s] = stu_TF[i]

# train_ss_dict = {s:[] for s in set(studentList)[:max_id+1]}
# train_sc_dict = {s:[[0], [0]] for s in set(list(train_data["student"]))}
# for stu, skill in zip(studentList, skillList):
#     train_ss_dict[stu].append(skill)
#
# for stu, corr in zip(studentList, correctList):
#     train_sc_dict[stu].append(corr)

# question2skill = {}
# for q, s in zip(questionList, skillList):
#     question2skill.update({q: s})

import torch.nn.functional as F
from sklearn.decomposition import PCA

train_skill_onehot = F.one_hot(torch.tensor(skillList), len(set(skill_set)))
final_skill_onehot = F.one_hot(torch.tensor(final_skill_list), len(set(skill_set)))

pca = PCA(n_components=5)
scaler = MinMaxScaler(feature_range=[-0.8, 0.8])
studentList = scaler.fit_transform(studentList)
stu_TF = scaler.fit_transform(stu_TF)
final_student_list = scaler.fit_transform(final_student_list)
final_questionList = scaler.fit_transform(final_questionList)
questionList = scaler.fit_transform(questionList)

train_skill_onehot = pca.fit_transform(train_skill_onehot.numpy())
final_skill_onehot = pca.fit_transform(final_skill_onehot.numpy())

studentList = torch.from_numpy(studentList)
stu_TF = torch.from_numpy(stu_TF)
train_skill_onehot = torch.from_numpy(train_skill_onehot)
questionList = torch.from_numpy(questionList)

final_student_list = torch.from_numpy(final_student_list)
final_TF = torch.from_numpy(final_TF)
final_skill_onehot = torch.from_numpy(final_skill_onehot)
final_questionList = torch.from_numpy(final_questionList)

# student和skill构成的矩阵
stu_skill = torch.cat((studentList, stu_TF, questionList, train_skill_onehot), dim=1)
final_stu_skill = torch.cat((final_student_list, final_TF, final_questionList, final_skill_onehot), dim=1)

# 划分数据集
train_onehot, test_onehot, train_corr, test_corr = train_test_split(stu_skill.numpy(), correctList, test_size=0.1,
                                                                    random_state=1)

from torch.utils.data import Dataset, DataLoader


class DKTDataset(Dataset):
    def __init__(self, onehot, correctList):
        self.onehot = torch.tensor(onehot, dtype=torch.float)
        self.correctList = torch.tensor(correctList)

    def __getitem__(self, index):
        return self.onehot[index], self.correctList[index]

    def __len__(self):
        return len(self.correctList)


train_dataset = DKTDataset(train_onehot, train_corr)
test_dataset = DKTDataset(test_onehot, test_corr)
train_iter = DataLoader(dataset=train_dataset, shuffle=True, batch_size=128, num_workers=2)
test_iter = DataLoader(dataset=test_dataset, shuffle=False, batch_size=128, num_workers=2)

import torch.nn as nn
import torch.optim as optim


class LinearG(nn.Module):
    def __init__(self):
        super(LinearG, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.BatchNorm1d(9),
            nn.Linear(9, 2)
        )


    def forward(self, X):
        """
        :param X:[batch_size, seq_len]
        :return:
        """
        outputs = self.fc(X)
        return F.sigmoid(outputs)


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
net = LinearG().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


def train():
    for epoch in range(200):
        train_total, train_correct = 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += y.shape[0]
            train_correct += (outputs.argmax(dim=1) == y).sum().item()

        if (epoch + 1) % 2 == 0:
            test_total, test_correct = 0, 0
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                outputs = net(X)

                test_total += y.shape[0]
                test_correct += (outputs.argmax(dim=1) == y).sum().item()
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss),
                  'train_accuracy =', '{:.2f}%'.format(train_correct / train_total * 100),
                  'test_accuracy =', '{:.2f}%'.format(test_correct / test_total * 100))


if __name__ == "__main__":
    train()

    labels = [0 for _ in range(len(final_stu_skill))]
    final_dataset = DKTDataset(final_stu_skill, labels)
    final_iter = DataLoader(final_dataset, batch_size=128, shuffle=False)

    predict = []
    for X, _ in final_iter:
        X = X.to(device)
        preds = net(X)
        preds = preds[:, 1]

        for p in preds:
            predict.append(p.item())

    with open("res4.txt", 'w') as f:
        for p in predict:
            p = str(p) + "\n"
            f.write(p)
