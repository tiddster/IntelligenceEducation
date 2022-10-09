import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

root_path = "D:\新建文件夹\Dataset\DKT\\train.csv"
test_path = "D:\新建文件夹\Dataset\DKT\\test.csv"

train_data = pd.read_csv(root_path)
test_data = pd.read_csv(test_path)

test_student_list = np.asarray(list(test_data["student"])).reshape((-1, 1))
# test_questionList = list(test_data["question"])
test_skill_list = list(test_data["skill"])

studentList = np.asarray(list(train_data["student"])).reshape((-1, 1))
skillList = list(train_data["skill"])
# questionList = list(data["question"])
correctList = list(train_data["correctness"])

# train_ss_dict = {s:[] for s in set(studentList)[:max_id+1]}
# train_sc_dict = {s:[] for s in set(studentList)[:max_id+1]}
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

train_skill_onehot = F.one_hot(torch.tensor(skillList), len(set(skillList)))
test_skill_onehot = F.one_hot(torch.tensor(test_skill_list), len(set(skillList)))


pca = PCA(n_components=5)
scaler = MinMaxScaler(feature_range=[-2,2])
studentList = scaler.fit_transform(studentList)
test_student_list = scaler.fit_transform(test_student_list)
train_skill_onehot = pca.fit_transform(train_skill_onehot.numpy())
test_skill_onehot = pca.fit_transform(test_skill_onehot.numpy())

train_skill_onehot = torch.from_numpy(train_skill_onehot)
test_skill_onehot = torch.from_numpy(test_skill_onehot)
studentList = torch.from_numpy(studentList)
test_student_list = torch.from_numpy(test_student_list)

stu_skill = torch.cat((studentList, train_skill_onehot), dim=1)
test_stu_skill = torch.cat((test_student_list, test_skill_onehot), dim=1)

train_onehot, test_onehot, train_corr, test_corr = train_test_split(stu_skill.numpy(), correctList, test_size=0.1, random_state=1)

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

class LSTM_Attention(nn.Module):
    def __init__(self):
        super(LSTM_Attention, self).__init__()
        self.lstm = nn.LSTM(1, 100, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(100*2, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def Attenion(self, lstm_output, final_state):
        # lstm_output : [batch_size, seq_len, num_hidden * num_directions(=2)], F matrix
        # final_state : [num_directions(=2), batch_size, num_hidden]
        batch_size = len(lstm_output)
        # hidden=[batch_size, num_hidden*num_directions(=2), 1]
        hidden = final_state.view((batch_size, -1, 1))

        # torch.bmm为多维矩阵的乘法：a=[b, h, w], c=[b,w,m]  bmm(a,b)=[b,h,m], 也就是对每一个batch都做矩阵乘法
        # squeeze(2), 判断第三维上维度是否为1，若为1则去掉
        # attn_weights:
        # = [batch_size, seq_len, num_hidden * num_directions(=2)] @  [batch_size, num_hidden*num_directions(=2), 1]
        # = [batch_size, seq_len, 1]
        attn_weights = lstm_output @ hidden

        soft_attn_weights = F.softmax(attn_weights, 1)

        # context
        # = [batch_size, num_hidden * num_directions(=2), seq_len] @  [batch_size, seq_len, 1]
        # = [batch_size, num_hidden * num_directions]
        context = (lstm_output.transpose(1, 2) @ soft_attn_weights).squeeze(2)
        return context

    def forward(self, X):
        """
        :param X:[batch_size, seq_len]
        :return:
        """
        # inputs: [batch_size, seq_len, embedding_dim]
        inputs = X.unsqueeze(-1)
        # inputs: [seq_len, batch_size, embedding_dim]
        inputs = inputs.transpose(0, 1)
        outputs, (final_hidden_state, final_cell_state) = self.lstm(inputs)
        # output : [batch_size, seq_len, n_hidden]
        # final_hidden_state : [1, batch_size, num_hidden]
        outputs = outputs.transpose(0, 1)
        attn_output = self.Attenion(outputs, final_hidden_state)

        # attn_output : [batch_size, num_classes], attention : [batch_size, seq_len, 1]
        outputs = self.fc(attn_output)
        return F.sigmoid(outputs)



device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
net = LSTM_Attention().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)


def train():
    for epoch in range(50):
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

    labels = [0 for _ in range(len(test_stu_skill))]
    final_dataset = DKTDataset(test_stu_skill, labels)
    final_iter = DataLoader(final_dataset, batch_size=128, shuffle=False)

    predict = []
    for X, _ in final_iter:
        X = X.to(device)
        preds = net(X)
        preds = preds[:, 1]

        for p in preds:
            predict.append(p.item())


    with open("res.txt", 'w') as f:
        for p in predict:
            p = str(p) + "\n"
            f.write(p)