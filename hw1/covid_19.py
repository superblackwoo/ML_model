import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 数据处理
import numpy as np
import csv
import os

# 画图
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# plt.switch_backend('sgg')

# 下面三个包是新增的
from sklearn.model_selection import train_test_split
import pandas as pd
import pprint as pp

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

tr_path = 'covid.train.csv'
tt_path = 'covid.test.csv'
data_tr = pd.read_csv(tr_path)
data_tt = pd.read_csv(tt_path)

print(data_tr.head(3))
print(data_tt.head(3))
print(data_tr.columns)

data_tr.drop(['id'], axis=1, inplace=True)
data_tt.drop(['id'], axis=1, inplace=True)  # 由于id列用不到，删除id列

cols = list(data_tr.columns)  # 拿到特征列名称
pp.pprint(data_tr.columns)

pp.pprint(data_tr.info())  # 看每列数据类型和大小

WI_index = cols.index('WI')  # WI列是states one-hot编码最后一列，取值为0或1，后面特征分析时需要把states特征删掉
WI_index  # wi列索引

pp.pprint(data_tr.iloc[:, 40:].describe())  # 从上面可以看出wi 列后面是cli, 所以列索引从40开始， 并查看这些数据分布
print(data_tt.iloc[:, 40:].describe())  # 查看测试集数据分布，并和训练集数据分布对比，两者特征之间数据分布差异不是很大

plt.figure(1)
plt.scatter(data_tr.loc[:, 'cli'], data_tr.loc[:, 'tested_positive.2'])  # 肉眼分析cli特征与目标之间相关性
plt.title("cli -- tested_positive.2")

plt.figure(2)
plt.scatter(data_tr.loc[:, 'ili'], data_tr.loc[:, 'tested_positive.2'])
plt.title("ili -- tested_positive.2")

plt.figure(3)
plt.scatter(data_tr.loc[:, 'cli'], data_tr.loc[:, 'ili'])  # cli 和ili两者差不多，所以这两个特征用一个就行
plt.title("cli -- ili")

plt.figure(4)
plt.scatter(data_tr.loc[:, 'tested_positive'], data_tr.loc[:, 'tested_positive.2'])  # day1 目标值与day3目标值相关性，线性相关的
plt.title("tested_positive -- tested_positive.2")

plt.figure(5)
plt.scatter(data_tr.loc[:, 'tested_positive.1'], data_tr.loc[:, 'tested_positive.2'])  # day2 目标值与day3目标值相关性，线性相关的
plt.title("tested_positive.1 -- tested_positive.2")
# plt.show()

print("上面手动分析太累，还是利用corr方法自动分析")
print(data_tr.iloc[:, 40:].corr())  # 上面手动分析太累，还是利用corr方法自动分析
# 锁定上面相关性矩阵最后一列，也就是目标值列，每行是与其相关性大小
data_corr = data_tr.iloc[:, 40:].corr()
target_col = data_corr['tested_positive.2']
print(target_col)

feature = target_col[target_col > 0.8]  # 在最后一列相关性数据中选择大于0.8的行，这个0.8是自己设的超参，大家可以根据实际情况调节
print(feature)

feature_cols = feature.index.tolist()  # 将选择特征名称拿出来
feature_cols.pop()  # 去掉test_positive标签
pp.pprint(feature_cols)  # 得到每个需要特征名称列表

feats_selected = [cols.index(col) for col in feature_cols]  # 获取该特征对应列索引编号，后续就可以用feats + feats_selected作为特征值
print(feats_selected)


def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


""" 
###################################################################################################################
"""


class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''

    def __init__(self,
                 path,
                 mu,  # mu,std是我自己加，baseline代码归一化有问题，我重写归一化部分
                 std,
                 mode='train',
                 target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:  # target_only 默认是false, 所以用的是全量特征，如果要用自己选择特征，则实例化这个类的时候，设置成True
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            # TODO: Using 40 states & 4 tested_positive features (indices = 57 & 75)

            feats = list(range(40)) + feats_selected  # feats_selected是我们选择特征, 40代表是states特征

            # 如果用只用两个特征，可以忽略前面数据分析过程,直接这样写
            # feats = list(range(40)) + [57, 75]

        if self.mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]

            # Splitting training data into train & dev sets
            #             if mode == 'train':
            #                 indices = [i for i in range(len(data)) if i % 10 != 0]
            #             elif mode == 'dev':
            #                 indices = [i for i in range(len(data)) if i % 10 == 0]

            # baseline上面这段代码划分训练集和测试集按照顺序选择数据，可能造成数据分布问题，我改成随机选择
            indices_tr, indices_dev = train_test_split([i for i in range(data.shape[0])], test_size=0.3, random_state=0)
            if self.mode == 'train':
                indices = indices_tr
            elif self.mode == 'dev':
                indices = indices_dev
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        # self.data[:, 40:] = \
        # (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
        # / self.data[:, 40:].std(dim=0, keepdim=True)
        # self.data = (self.data - self.data.mean(dim = 0, keepdim = True)) / self.data.std(dim=0, keepdim=True)

        # baseline这段代码数据归一化用的是当前数据归一化，事实上验证集上和测试集上归一化一般只能用过去数据即训练集上均值和方差进行归一化

        if self.mode == "train":  # 如果是训练集，均值和方差用自己数据
            self.mu = self.data[:, 40:].mean(dim=0, keepdim=True)
            self.std = self.data[:, 40:].std(dim=0, keepdim=True)
        else:  # 测试集和开发集，传进来的均值和方差是来自训练集保存，如何保存均值和方差，看数据dataload部分
            self.mu = mu
            self.std = std

        self.data[:, 40:] = (self.data[:, 40:] - self.mu) / self.std  # 归一化
        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False, mu=None,
                    std=None):  # 训练集不需要传mu,std, 所以默认值设置为None
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mu, std, mode=mode, target_only=target_only)  # Construct dataset
    if mode == 'train':  # 如果是训练集，把训练集上均值和方差保存下来
        mu = dataset.mu
        std = dataset.std
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)  # Construct dataloader
    return dataloader, mu, std


class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''

    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 68),  # 70是我调得最好的， 而且加层很容易过拟和
            nn.ReLU(),
            nn.Linear(68, 1)
        )
        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L2 regularization here
        eps = 1e-6
        l2_reg = 0
        alpha = 0.0001
        # 这段代码是l2正则，但是实际操作l2正则效果不好，大家也也可以调，把下面这段代码取消注释就行
        #         for name, w in self.net.named_parameters():
        #             if 'weight'  in name:
        #                 l2_reg += alpha * torch.norm(w, p = 2).to(device)
        return torch.sqrt(self.criterion(pred, target) + eps) + l2_reg
        # lr_reg=0, 后面那段代码用的是均方根误差，均方根误差和kaggle评测指标一致，而且训练模型也更平稳


def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}  # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()  # set model to training mode
        for x, y in tr_set:  # iterate through the dataloader
            optimizer.zero_grad()  # set gradient to zero
            x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
            pred = model(x)  # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()  # compute gradient (backpropagation)
            optimizer.step()  # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()  # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:  # iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)  # compute averaged loss

    return total_loss


def test(tt_set, model, device):
    model.eval()  # set model to evalutation mode
    preds = []
    for x in tt_set:  # iterate through the dataloader
        x = x.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            preds.append(pred.detach().cpu())  # collect prediction
    preds = torch.cat(preds, dim=0).numpy()  # concatenate all predictions and convert to a numpy array
    return preds


device = get_device()  # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/

# target_only = False  ## TODO: Using 40 states & 2 tested_positive features
target_only = True  # 使用自己的特征，如果设置成False，用的是全量特征
# TODO: How to tune these hyper-parameters to improve your model's performance? 这里超参数没怎么调，已经最优的了
config = {
    'n_epochs': 3000,  # maximum number of epochs
    'batch_size': 270,  # mini-batch size for dataloader
    'optimizer': 'SGD',  # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.005,  # learning rate of SGD
        'momentum': 0.5  # momentum for SGD
    },
    'early_stop': 200,  # early stopping epochs (the number epochs since your model's last improvement)
    # 'save_path': 'models/model.pth'  # your model will be saved here
    'save_path': 'models/model_select.path'
}
tr_set, tr_mu, tr_std = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set, mu_none, std_none = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only, mu=tr_mu,
                                            std=tr_std)
tt_set, mu_none, std_none = prep_dataloader(tr_path, 'test', config['batch_size'], target_only=target_only, mu=tr_mu,
                                            std=tr_std)

model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

plot_learning_curve(model_loss_record, title='deep model')
dev(dv_set, model, device)  # 验证集损失

del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, 'pred.csv')  # save prediction file to pred.csv

print("done")
