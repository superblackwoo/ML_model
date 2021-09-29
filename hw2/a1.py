import numpy as np

print('Loading data ...')

data_root = 'ml2021spring-hw2/timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))

import torch
from torch.utils.data import Dataset


class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


VAL_RATIO = 0.2  # val_ratto 默认是0.2， 如果为0的话是拿所有数据进行训练。

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))

# 这个地方我把训练数据顺序打乱
import random

indices = [i for i in range(percent)]
random.shuffle(indices)
train_x, train_y = train_x[indices], train_y[indices]

BATCH_SIZE = 256
num_workers = 4  # 这里大家根据自己cpu资源改写，cpu core比较填写叫小的数字，一般是2的幂次方
from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=num_workers)  # only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

import gc

del train, train_label, train_x, train_y, val_x, val_y
gc.collect()

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.bn0 = nn.BatchNorm1d(429)
        self.layer1 = nn.Linear(429, 4096)  # 429是样本特征个数， 1024第一层神经元个数
        self.bn1 = nn.BatchNorm1d(4096)
        self.layer2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.layer3 = nn.Linear(2048, 1024)  # 第二层神经元个数512
        self.bn3 = nn.BatchNorm1d(1024)
        self.layer4 = nn.Linear(1024, 512)  # 第三层128
        self.bn4 = nn.BatchNorm1d(512)
        self.layer5 = nn.Linear(512, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 39)  # 最后一层 39
        self.dropout = nn.Dropout(p=0.5)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()  # 激活函数

    def forward(self, x):
        # 第一层前向传递
        x = self.bn0(x)
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        # 第二层前向传递
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        # x = self.dropout(x)
        # 第三层前向传递
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.sigmoid(x)
        # x = self.dropout(x)
        # layer4
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.sigmoid(x)
        x = self.dropout(x)

        # layer5
        x = self.layer5(x)
        x = self.bn5(x)
        x = self.sigmoid(x)
        # x = self.dropout(x)
        # 最后一层输出
        x = self.out(x)
        return x


# check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# fix random seed for reproducibility
same_seeds(0)

# get device
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 30  # number of training epoch
learning_rate = 0.0005  # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# start training

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        batch_loss.backward()
        optimizer.step()

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)

                val_acc += (
                        val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                val_acc / len(val_set), val_loss / len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
        ))
        # 这段代码是如果拿所有数据来训练模型，保存模型中指定epoch结果
#         if epoch == 3:
#             torch.save(model.state_dict(), 'model_epch_3.ckpt')
#         elif epoch == 6:
#             torch.save(model.state_dict(), 'model_epch_6.ckpt')
#         elif epoch == 9:
#             torch.save(model.state_dict(), 'model_epch_9.ckpt')
#         elif epoch == 12:
#             torch.save(model.state_dict(), 'model_epch_12.ckpt')
#         elif epoch == 14:
#             torch.save(model.state_dict(), 'model_epch_14.ckpt')
# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

predict = []
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
