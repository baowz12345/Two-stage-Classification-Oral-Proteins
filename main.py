import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch, \
    KMeans
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import model_selection as cv, tree
from sklearn import metrics,preprocessing
# Import some data to play with
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef, f1_score, accuracy_score, precision_score
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm, global_mean_pool, global_max_pool, global_add_pool

import torch.nn.functional as F
from torch.nn import Parameter

import warnings
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch, \
    KMeans
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier
from deepforest import CascadeForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import model_selection as cv, tree
from sklearn import metrics,preprocessing
# Import some data to play with
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef, f1_score, accuracy_score, precision_score
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm, global_mean_pool, global_max_pool, global_add_pool

import torch.nn.functional as F
from torch.nn import Parameter

import warnings
import pydotplus
import graphviz
warnings.filterwarnings('ignore')
from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
import psutil
import os
import torch
import warnings
def dataread(fname):
    s = np.loadtxt(fname, dtype=np.float32, delimiter=' ')
    qian1 = [row for row in s if row[-1] == 1]
    X1 = [sublist[:-1] for sublist in qian1]
    # 使用列表推导式筛选出最后一列为0的子列表
    hou0 = [row for row in s if row[-1] == 0]
    X0 = [sublist[:-1] for sublist in hou0]
    X1 = np.array(X1)
    X0 = np.array(X0)
    # s = np.vstack((filtered_data, other_data))
    # print(s)
    # end = s.shape[1] - 1
    # X = s[:, :end]
    # y = s[:, -1]
    # X1=filtered_data[:, :-1]
    # X0=other_data[:, :-1]
    # y1=filtered_data[:, -1]
    # y0=other_data[:, -1]
    return X1,X0

#1.2.2
def Linear(X_train, X_test, y_train, y_test):
    clf = RidgeClassifier().fit(X_train, y_train)
    print( y_test)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    print(hun)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***Linear***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn)+' '+str(sp)+'\n')
    f.close()
    y_score = clf.decision_function(X_test)
    print(type(y_score))
    print(type(y_test))
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.4.1
def SVC_T(X_train, X_test, y_train, y_test):
    from sklearn import svm, datasets
    random_state = np.random.RandomState(0)
    # n_samples, n_features = X.shape
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
    y_pred = svm.fit(X_train, y_train).predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***SVM***")

    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
    y_score = svm.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.5.1
def SGD(X_train, X_test, y_train, y_test):
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***SGD***")

    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.decision_function(X_test)
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.6.2
def Neighbors(X_train, X_test, y_train, y_test):
    neigh = RadiusNeighborsClassifier(radius=1.0)
    neigh = neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)

    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***Neighbors***")

    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = neigh.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.7.2
def GPC(X_train, X_test, y_train, y_test):
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X_train, y_train)
    y_pred = gpc.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0,0]/(hun[0,0]+hun[0,1])
    sp = hun[1,1]/(hun[1,1]+hun[1,0])
    print("***GPC***")
    print("sn = ",sn)
    print("sp = ",sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = gpc.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.9.1
def Gaussian_NB(X_train, X_test, y_train, y_test):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***Gaussian_NB***")
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    print("sn = ", sn)
    print("sp = ", sp)

    y_score = gnb.fit(X_train, y_train).predict_proba(X_test)
    y_score = y_score[:,1]# y_score是正类的概率估计值
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.9.4
def Bernoulli_NB(X_train, X_test, y_train, y_test):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    bnb = BernoulliNB()
    y_pred = bnb.fit(X_train, y_train).predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***Bernoulli_NB***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = bnb.fit(X_train, y_train).predict_proba(X_test)
    y_score = y_score[:, 1]  # y_score是正类的概率估计值
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.10.1
def DT(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #tree.plot_tree(clf)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***DT***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr,roc_auc

#1.11.1
def Bagging(X_train, X_test, y_train, y_test):
    clf = BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***Bagging***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.11.2
def RandomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***RandomForest***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.11.3
def AdaBoost(X_train, X_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***AdaBoost***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.11.4
def GradientBoosting(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***GradientBoosting***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.11.5
def HistGradientBoosting(X_train, X_test, y_train, y_test):
    clf = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***HistGradientBoosting***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#1.17.2
def  MLPClassifier_1 (X_train, X_test, y_train, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])

    print("***MLPClassifier***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#定义class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=2).double()
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(1, 100, num_layers=1, dropout=0.5,
                            bidirectional=True).double()
        self.liner1 = nn.Linear(200, 1).double()
        self.liner2 = nn.Linear(10, 2).double()

    def forward(self, x):
        output = self.conv1(x)
        output = self.max_pool1(output)
        w = output.shape[2]
        self.lstm = nn.LSTM(w, 100, num_layers=1, dropout=0.5,
                            bidirectional=True).double()
        hidden_cell = (torch.zeros([2, 10, 100], dtype=torch.double), torch.zeros([2, 10, 100], dtype=torch.double))
        # x.view(-1,40 * 14)
        lstm_out, (h_n, h_c) = self.lstm(output, hidden_cell)
        output = self.liner1(lstm_out)
        output = output.permute(0, 2, 1)
        output = self.liner2(output)
        output = output.squeeze(1)
        output = F.softmax(output)
        return output


def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    return TP, FP, TN, FN

# CNNBilstm
def CNNBilstm(X_train, X_test, Y_train, Y_test):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.3, random_state=0)
    train_data = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(Y_train)))
    valid_data = TensorDataset(torch.from_numpy(np.array(X_test)), torch.from_numpy(np.array(Y_test)))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
    test_loader = DataLoader(valid_data, shuffle=True, batch_size=8)

    # 建模三件套：loss，优化，epochs -
    model = Net()  # 模型
    loss_function = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    epochs = 5
    # 开始训练
    model.train()
    for i in range(epochs):
        acc1 = []
        precision_scores = []
        f1_scores = []
        recall_scores = []
        sp1 = []
        MCC1 = []
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq.unsqueeze(1).double())  # .squeeze()
            # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            acc1.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
            TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
            if ((TN + FP) != 0):
                Sp = TN / (TN + FP)
            else:
                Sp = 0
            sp1.append(Sp)
            MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
            MCC1.append(MCC)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
        #print("Train Step:", i," acc:{:.6f}, pre:{:.4f},f1score:{:.4f},Sn:{:.4f},Sp:{:.4f},MCC:{:.4f} ".format(np.array(acc1).mean(),np.array(precision_scores).mean(),np.array(f1_scores).mean(),
        # np.array(recall_scores).mean(), np.array(sp1).mean(), np.array(MCC1).mean()))
    # 开始验证
    model.eval()
    acc2 = []
    precision_scores = []
    f1_scores = []
    recall_scores = []
    sp1 = []
    MCC1 = []
    p1 = []
    l1 = []

    for i in range(epochs):
        for seq, labels in test_loader:  # 这里偷个懒，就用训练数据验证哈！
            y_pred = model(seq.unsqueeze(1).double())  # .squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            p = y_pred.detach().numpy().tolist()
            l = labels.numpy().tolist()
            for j in range(len(p)):

                p1.append(p[j])
                l1.append(int(l[j]))
            acc2.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
    TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
    if ((TN + FP) != 0):
        Sp = TN / (TN + FP)
    else:
        Sp = 0
    sp1.append(Sp)
    MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
    MCC1.append(MCC)
    print('***CNNBilstm***')
    print('SN = ', np.array(recall_scores).mean(), 'SP = ', np.array(sp1).mean())

    fpr, tpr, threshold = roc_curve(l1, [y[1] for y in p1])  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

def CNNBilstm_Attention(X_train, X_test, Y_train, Y_test):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.3, random_state=0)
    train_data = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(Y_train)))
    valid_data = TensorDataset(torch.from_numpy(np.array(X_test)), torch.from_numpy(np.array(Y_test)))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
    test_loader = DataLoader(valid_data, shuffle=True, batch_size=8)

    # 建模三件套：loss，优化，epochs -
    model = Net()  # 模型
    loss_function = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    epochs = 5
    # 开始训练
    model.train()
    for i in range(epochs):
        acc1=[]
        precision_scores = []
        f1_scores=[]
        recall_scores=[]
        sp1=[]
        MCC1=[]
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq.unsqueeze(1).double())#.squeeze()
            # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            acc1.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
            TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
            if((TN+FP)!=0):
              Sp = TN / (TN + FP)
            else:
              Sp=0
            sp1.append(Sp)
            MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
            MCC1.append(MCC)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
    # 开始验证
    model.eval()
    acc2 = []
    precision_scores = []
    f1_scores = []
    recall_scores = []
    sp1 = []
    MCC1 = []
    p1 = []
    l1 = []
    y_score =[]
    for i in range(epochs):
        for seq, labels in test_loader:  # 这里偷个懒，就用训练数据验证哈！
            y_pred = model(seq.unsqueeze(1).double())#.squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            p = torch.max(y_pred, dim=1)[1].numpy().tolist()
            l = labels.numpy().tolist()
            for j in range(len(p)):
                p1.append(p[j])
                l1.append(int(l[j]))
            acc2.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
    TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
    if ((TN + FP) != 0):
        Sp = TN / (TN + FP)
    else:
        Sp = 0
    sp1.append(Sp)
    MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
    MCC1.append(MCC)
    print('***CNNBilstm_Attention***')
    print('SN = ', np.array(recall_scores).mean(), 'SP = ', np.array(sp1).mean())

    fpr, tpr, threshold = roc_curve(l1,p1)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#TextCNN
def TextCNN(X_train, X_test, Y_train, Y_test):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.3, random_state=0)
    train_data = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(Y_train)))
    valid_data = TensorDataset(torch.from_numpy(np.array(X_test)), torch.from_numpy(np.array(Y_test)))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
    test_loader = DataLoader(valid_data, shuffle=True, batch_size=8)
    # 建模三件套：loss，优化，epochs -
    model = Net()  # 模型
    loss_function = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    epochs = 5
    # 开始训练
    model.train()
    for i in range(epochs):
        acc1=[]
        precision_scores = []
        f1_scores=[]
        recall_scores=[]
        sp1=[]
        MCC1=[]
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq.unsqueeze(1).double())#.squeeze()
            # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            acc1.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
            TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
            if((TN+FP)!=0):
              Sp = TN / (TN + FP)
            else:
              Sp=0
            sp1.append(Sp)
            MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
            MCC1.append(MCC)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
    # 开始验证
    model.eval()
    acc2 = []
    precision_scores = []
    f1_scores = []
    recall_scores = []
    sp1 = []
    MCC1 = []
    p1 = []
    l1 = []
    for i in range(epochs):
        for seq, labels in test_loader:  # 这里偷个懒，就用训练数据验证哈！
            y_pred = model(seq.unsqueeze(1).double())#.squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            p = torch.max(y_pred, dim=1)[1].numpy().tolist()
            l = labels.numpy().tolist()
            for j in range(len(p)):
                p1.append(p[j])
                l1.append(int(l[j]))
            acc2.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
    TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
    if ((TN + FP) != 0):
        Sp = TN / (TN + FP)
    else:
        Sp = 0
    sp1.append(Sp)
    MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
    MCC1.append(MCC)
    print('***TextCNN***')

    print('SN = ', np.array(recall_scores).mean(), 'SP = ', np.array(sp1).mean())

    fpr, tpr, threshold = roc_curve(l1, p1)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#Graph_Code
class GAT_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=1):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=heads)
        self.gat2 = GATConv(hidden * heads, classes)

    def forward(self, data):
        x, edge_index = data['x'],data['edge_index']
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GraphSAGE_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(GraphSAGE_Net, self).__init__()
        self.sage1 = SAGEConv(features, hidden)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data['x'],data['edge_index']
        x = self.sage1(x, edge_index).float()
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index).float()
        return F.log_softmax(x, dim=1)
class GCN_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(features, hidden)
        self.conv2 = GCNConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data['x'],data['edge_index']
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class Graph_Class(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Graph_Class, self).__init__()
    self.conv1 = GCNConv(input_size, hidden_size)
    self.conv2 = GCNConv(hidden_size, hidden_size)
    self.conv3 = GCNConv(hidden_size, hidden_size)
    self.gat1 = GATConv(input_size, hidden_size, heads=4, dropout=0.6)
    self.gat2 = GATConv(hidden_size, hidden_size, heads=4, dropout=0.6)
    self.norm1 = BatchNorm(hidden_size*4)
    self.norm2 = BatchNorm(hidden_size)
    self.line1 =nn.Linear(64, output_size)
  def forward(self, data):
    # 1. Obtain node embeddings
    x = self.conv1(data['x'], data['edge_index'])
    x = self.norm2(x)
    x = torch.relu(x)
    x = self.gat2(x, data['edge_index'])
    x = self.norm1(x)
    x = torch.relu(x)
    x = F.dropout(x, p=0.6, training=self.training)
    x = torch.sigmoid(self.line1(x))
    return x

def Graph_Code(fname):
    warnings.filterwarnings('ignore')

    idx_features_labels = np.genfromtxt(fname,
                                        dtype=np.dtype(str))
    data = {}
    data['x'] = Parameter(torch.from_numpy(np.float32(idx_features_labels[:, 0:-1])))
    data['label'] = torch.tensor(torch.from_numpy(np.float32(idx_features_labels[:, -1])))
    w = idx_features_labels.shape
    num_node_features = w[1] - 1    #这个参数是维度也就是列数
    num_classes = 2
    x1 = torch.arange(w[0], dtype=torch.int64)      #这边输入的是行数
    x2 = torch.arange(w[0], dtype=torch.int64)
    data['edge_index'] = torch.stack([x1, x2], 0)
    # print(data['edge_index'])
    # print(data['edge_index'].dtype)
    train_mask = range(0, w[0])             #这是训练集的范围就是 0行到多少行
    val_mask = range(w[1] - 1, w[0])
    test_mask = range(w[1] - 1, w[0])       #测试集就是  w[1] - 1 行 到 w[0]行


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Graph_Class(num_node_features, 16, num_classes).float()#.to(device)/
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # main loop
    dur = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    sp1=[]
    p2=[]
    l2=[]
    MCC1 = []
    # batch=torch.tensor(1)
    for epoch in range(100):
        if epoch >= 3:
            t0 = time.time()
        best_val_acc = 0
        logits = model(data)
        logp = F.log_softmax(logits, 1)
        pred = F.log_softmax(logits, 1).argmax(1)
        loss = F.nll_loss(logp[train_mask], data['label'][train_mask].long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss = F.cross_entropy(logits[train_mask], labels[train_mask].long())
        train_acc = (pred[train_mask] == data['label'][train_mask]).float().mean()
        val_acc = (pred[val_mask] == data['label'][test_mask]).float().mean()
        test_acc = (pred[test_mask] == data['label'][test_mask]).float().mean()
        MCC =0
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        logit = logits.detach().numpy().tolist()
        p = pred[test_mask].numpy().tolist()
        l = data['label'][test_mask].numpy().tolist()
        vali_f1 = f1_score(l, p, average="micro")
        recall=recall_score(l, p, labels=None, pos_label=1, average='binary', sample_weight = None)
        p1=logp[test_mask].detach().numpy().tolist()
        for i in range(len(p1)):
            p2.append(p1[i])
            l2.append(int(l[i]))
        if epoch >= 3:
            dur.append(time.time() - t0)
        TP, FP, TN, FN = perf_measure(l, p)
        # Sp = TN/(TN+FP)
        if ((TN + FP) != 0):
            Sp = TN / (TN + FP)
        else:
            Sp = 0
        sp1.append(Sp)
        MCC = matthews_corrcoef(l, p)
        accuracy_scores.append(accuracy_score(l, p))
        precision_scores.append(precision_score(l, p))
        recall_scores.append(recall_score(l, p))
        f1_scores.append(f1_score(l, p))
        #print("Epoch {:05d} | Time(s) {:.4f} | train acc: {:.6f}| test acc: {:.6f}| f1_score: {:.5f}| Sn: {:.5f}| MCC: {:.5f}".format(
            #epoch+1, np.mean(dur),train_acc,test_acc,vali_f1,recall,MCC))

    # TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
    # if ((TN + FP) != 0):
    #     Sp = TN / (TN + FP)
    # else:
    #     Sp = 0
    # sp1.append(Sp)
    # MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
    # MCC1.append(MCC)
    print('***GraphCode***')

    print('SN = ', np.array(recall_scores).mean(), 'SP = ', np.array(sp1).mean())
    if np.array(recall_scores).mean()+np.array(sp1).mean()!=0:
        fpr, tpr, threshold = roc_curve(l2, [y[1] for y in p2])  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        return fpr, tpr, roc_auc

def XGboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    hun = metrics.confusion_matrix(y_test, y_pred)
    print(hun)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***XGboosting***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn)+' '+str(sp)+'\n')
    f.close()
    y_pred_prob = model.predict_proba(X_test)[:, 1]#这边用了y_pred_prob代替了y_score因为不是sklearn库。
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    return fpr, tpr, roc_auc


def lightgbm(X_train, X_test, y_train, y_test):
    ## 定义 LightGBM 模型
    clf = LGBMClassifier()
    # 在训练集上训练LightGBM模型
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    ## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
    hun = metrics.confusion_matrix(y_test, y_pred)
    sn = hun[0, 0] / (hun[0, 0] + hun[0, 1])
    sp = hun[1, 1] / (hun[1, 1] + hun[1, 0])
    print("***Lightgbm***")

    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()

    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

#BLS-CroSS-validation
def show_accuracy(predictLabel, Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count / len(Label), 5))

class node_generator(object):
    def __init__(self, isenhance=False):
        self.Wlist = []
        self.blist = []
        self.function_num = 0
        self.isenhance = isenhance

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(x, 0)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def linear(self, x):
        return x

    def orth(self, W):
        """
        orth是正交基的意思，求正交基可能是为了使增强节点彼此无关
        目前看来，这个函数应该配合下一个generator函数是生成权重的
        此函数传入的weights与传出的weights的shape是一样的。
        """
        for i in range(0, W.shape[1]):
            w = np.mat(W[:, i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:, j].copy()).T
                w_sum += (w.T.dot(wj))[0, 0] * wj
            w -= w_sum
            w = w / np.sqrt(w.T.dot(w))
            W[:, i] = np.ravel(w)

        return W

    def generator(self, shape, times):
        for i in range(times):
            W = 2 * np.random.random(size=shape) - 1
            if self.isenhance == True:
                W = self.orth(W)  # 只在增强层使用
            b = 2 * np.random.random() - 1
            yield (W, b)

    def generator_nodes(self, data, times, batchsize, function_num):
        # 按照bls的理论，mapping layer是输入乘以不同的权重加上不同的偏差之后得到的
        # 若干组，所以，权重是一个列表，每一个元素可作为权重与输入相乘
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]

        self.function_num = {'linear': self.linear,
                             'sigmoid': self.sigmoid,
                             'tanh': self.tanh,
                             'relu': self.relu}[function_num]  # 激活函数供不同的层选择
        # 下面就是先得到一组mapping nodes，再不断叠加，得到len(Wlist)组mapping nodes
        nodes = self.function_num(data.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            nodes = np.column_stack((nodes, self.function_num(data.dot(self.Wlist[i]) + self.blist[i])))
        return nodes

    def transform(self, testdata):
        testnodes = self.function_num(testdata.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.function_num(testdata.dot(self.Wlist[i]) + self.blist[i])))
        return testnodes

class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0

    def fit_transform(self, traindata):
        self._mean = traindata.mean(axis=0)
        self._std = traindata.std(axis=0)
        return (traindata - self._mean) / (self._std + 0.001)

    def transform(self, testdata):
        return (testdata - self._mean) / (self._std + 0.001)

class broadNet(object):
    def __init__(self, map_num=10, enhance_num=10, map_function='linear', enhance_function='linear', batchsize='auto'):
        self.map_num = map_num
        self.enhance_num = enhance_num
        self.batchsize = batchsize
        self.map_function = map_function
        self.enhance_function = enhance_function

        self.W = 0
        self.pseudoinverse = 0
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.mapping_generator = node_generator()
        self.enhance_generator = node_generator(isenhance=True)

    def fit(self, data, label):
        if self.batchsize == 'auto':
            self.batchsize = data.shape[1]

        data = self.normalscaler.fit_transform(data)
        label = self.onehotencoder.fit_transform(np.mat(label).T)

        mappingdata = self.mapping_generator.generator_nodes(data, self.map_num, self.batchsize, self.map_function)
        enhancedata = self.enhance_generator.generator_nodes(mappingdata, self.enhance_num, self.batchsize,
                                                             self.enhance_function)
        #
        # print('number of mapping nodes {0}, number of enhence nodes {1}'.format(mappingdata.shape[1],
        #                                                                         enhancedata.shape[1]))
        # print('mapping nodes maxvalue {0} minvalue {1} '.format(round(np.max(mappingdata), 5),
        #                                                         round(np.min(mappingdata), 5)))
        # print('enhence nodes maxvalue {0} minvalue {1} '.format(round(np.max(enhancedata), 5),
        #                                                         round(np.min(enhancedata), 5)))

        inputdata = np.column_stack((mappingdata, enhancedata))
        # print('input shape ', inputdata.shape)
        pseudoinverse = np.linalg.pinv(inputdata)
        # 新的输入到输出的权重
        # print('pseudoinverse shape:', pseudoinverse.shape)
        self.W = pseudoinverse.dot(label)

    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def decode1(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis[1])
        return np.array(Y)

    def accuracy(self, predictlabel, label):
        label = np.ravel(label).tolist()
        predictlabel = predictlabel.tolist()
        count = 0
        for i in range(len(label)):
            if label[i] == predictlabel[i]:
                count += 1
        return (round(count / len(label), 5))

    def predict(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_mappingdata = self.mapping_generator.transform(testdata)
        test_enhancedata = self.enhance_generator.transform(test_mappingdata)

        test_inputdata = np.column_stack((test_mappingdata, test_enhancedata))
        return self.decode(test_inputdata.dot(self.W))
    def predict_score(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_mappingdata = self.mapping_generator.transform(testdata)
        test_enhancedata = self.enhance_generator.transform(test_mappingdata)

        test_inputdata = np.column_stack((test_mappingdata, test_enhancedata))
        return self.decode1(test_inputdata.dot(self.W))

def BLS_Cross(traindata, testdata, trainlabel, testlabel):

    label = trainlabel
    print(label)
    data = traindata
    print(data)


    bls = broadNet(map_num=10,
                   enhance_num=10,
                   map_function='relu',
                   enhance_function='relu',
                   batchsize=10)

    starttime = datetime.datetime.now()
    bls.fit(traindata, trainlabel)
    endtime = datetime.datetime.now()
    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))
    predictlabel = bls.predict(testdata)
    # print(show_accuracy(predictlabel, testlabel))
    Training_Division_prec = precision_score(testlabel, predictlabel, pos_label=1)
    Training_Division_recall = recall_score(testlabel, predictlabel, pos_label=1)
    Training_Division_f1 = f1_score(testlabel, predictlabel, pos_label=1)
    # test_auc = roc_auc_score(text_label, lr_score[:, 1])
    con_matrix = confusion_matrix(testlabel, predictlabel)
    print('--con_matrix',con_matrix)
    print('testlabel',testlabel)#17个数
    print('predictlabel',predictlabel)
    Training_Division_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    Training_Division_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (
            ((con_matrix[1][1] + con_matrix[0][1]) * (con_matrix[1][1] + con_matrix[1][0]) * (
                        con_matrix[0][0] + con_matrix[0][1]) * (con_matrix[0][0] + con_matrix[1][0])) ** 0.5)
    print("Training Set Division  :acc: ", show_accuracy(predictlabel, testlabel), " ; prec: ",  Training_Division_spec, " ; recall: ", Training_Division_recall ,
          " ; f1: ", Training_Division_f1, "  ; spec:", Training_Division_spec, " ; mcc: ", Training_Division_mcc)
    #十折交叉验证
    KF = KFold(n_splits=10, shuffle=True, random_state=100)
    Pre=[]
    Acc=[]
    Sp=[]
    Sn=[]
    F1=[]
    p=[]
    l=[]
    MCC=[]
    for train_index, test_index in KF.split(data):
        bls.fit(data[train_index], label[train_index])
        predictlabel = bls.predict(data[test_index])
        predictscore = bls.predict_score(data[test_index])
        k= predictscore.tolist()
        m=label[test_index].tolist()
        for i in range(len(m)):
            p.append(k[i])
            l.append(m[i])
        Acc.append(show_accuracy(predictlabel, label[test_index]))
        Training_Division_prec = precision_score(label[test_index], predictlabel, pos_label=1)
        Training_Division_recall = recall_score(label[test_index], predictlabel, pos_label=1)
        Training_Division_f1 = f1_score(label[test_index], predictlabel, pos_label=1)
        print('++label[test_index]',label[test_index])
        print('predictlabel', predictlabel)
        con_matrix = confusion_matrix(label[test_index], predictlabel)
        print('con_matrix',con_matrix)
        print('type',type(con_matrix))
        if len(con_matrix)==1:
            new_array = np.zeros((2, 2))
            new_array[0, 0] = con_matrix[0, 0]
            Training_Division_spec = new_array[0][0] / (new_array[0][0] + new_array[0][1])
            Training_Division_mcc = (new_array[0][0] * new_array[1][1] - new_array[0][1] * new_array[1][0]) / (
                    ((new_array[1][1] + new_array[0][1]) * (new_array[1][1] + new_array[1][0]) * (
                            new_array[0][0] + new_array[0][1]) * (new_array[0][0] + new_array[1][0])) ** 0.5)
            Sp.append(Training_Division_spec)
            Pre.append(Training_Division_prec)
            Sn.append(Training_Division_recall)
            F1.append(Training_Division_f1)
            MCC.append(Training_Division_mcc)
        else:
            Training_Division_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
            Training_Division_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (
                    ((con_matrix[1][1] + con_matrix[0][1]) * (con_matrix[1][1] + con_matrix[1][0]) * (
                            con_matrix[0][0] + con_matrix[0][1]) * (con_matrix[0][0] + con_matrix[1][0])) ** 0.5)
            Sp.append(Training_Division_spec)
            Pre.append(Training_Division_prec)
            Sn.append(Training_Division_recall)
            F1.append(Training_Division_f1)
            MCC.append(Training_Division_mcc)
    print("Training Set Cross-validation  :acc: ", np.mean(Acc), " ; prec: ", np.mean(Pre)," ; recall: ", np.mean(Sn), " ; f1: ",   np.mean(F1), "  ; spec:", np.mean(Sp), " ; mcc: ", np.mean(MCC))
    f = open(fnameresult, 'a')
    f.write(str(np.mean(Sn)) + ' ' + str(np.mean(Sp)) + '\n')
    f.close()
    fpr, tpr, threshold = roc_curve(l, p)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc

def Cascade_Forest1(traindata, testdata, trainlabel, testlabel):
    model = CascadeForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    Linear_hun = metrics.confusion_matrix(y_test, y_pred)
    sn = Linear_hun[0, 0] / (Linear_hun[0, 0] + Linear_hun[0, 1])
    sp = Linear_hun[1, 1] / (Linear_hun[1, 1] + Linear_hun[1, 0])
    print("***Cascade Forest***")
    print("sn = ", sn)
    print("sp = ", sp)
    f = open(fnameresult, 'a')
    f.write(str(sn) + ' ' + str(sp) + '\n')
    f.close()
    y_score = model.predict_proba(X_test)
    y_score = y_score[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return fpr, tpr, roc_auc


def Roc(name,fpr, tpr,roc_auc):
    lw = 2
    #宽度
    plt.plot(fpr, tpr,
         lw=lw, label=name+' ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return plt.figure




if __name__ == "__main__":
    # features = ['AC', 'ACC', 'CC', 'DP', 'DR', 'KMER', 'PC-PseAAC', 'PC-PseAAC-General',  'SC-PseAAC',
    #             'SC-PseAAC-General']
    # feature = ['AAC','AC','ACC','APAAC','ASDC','CC','CKSAAGP','CKSAAP','CTDC','CTDD','CTDT','CTRIAD','DDE','DistancePair','DPC',
    #                 'DistancePair','DPC','GAAC','GDPC','GEARY','GTPC','KSCTRIAD','MORAN','NMBROTO','PAAC','QSORDER','SOCNUMBER','TPC',
    #                 'PseKRAAC 1', 'PseKRAAC 2', 'PseKRAAC 3A', 'PseKRAAC 3B', 'PseKRAAC 5', 'PseKRAAC 6A',
    #                 'PseKRAAC 6B', 'PseKRAAC 6C', 'PseKRAAC 7', 'PseKRAAC 8', 'PseKRAAC 9', 'PseKRAAC 10',
    #                 'PseKRAAC 11', 'PseKRAAC 12', 'PseKRAAC 13', 'PseKRAAC 14', 'PseKRAAC 15', 'PseKRAAC 16' ]
    # feature = ['AACPCA5', 'CKSAAPPCA5', 'DDEPCA5', 'DPCPCA5', 'CKSAAPPCA80A', 'CKSAAPPCA160A', 'CKSAAPPCA320A',
    #            'DDEPCA20A', 'DDEPCA40A', 'DDEPCA80A', 'DPCPCA20A', 'DPCPCA40A', 'DPCPCA80A']
    # feature = ['AAC1CKSAAGP1', 'AAC1CKSAAP1', 'AAC1CTDC1', 'AAC1CTDD1', 'AAC1CTDT1', 'AAC1DDE1', 'AAC1DPC1', 'AAC1GAAC1', 'AAC1GDPC1', 'AAC1Geary1', 'AAC1GTPC1', 'AAC1Moran1', 'AAC1NMBroto1', 'AAC1TPC1', 'CKSAAGP1CKSAAP1', 'CKSAAGP1CTDC1', 'CKSAAGP1CTDD1', 'CKSAAGP1CTDT1', 'CKSAAGP1DDE1', 'CKSAAGP1DPC1', 'CKSAAGP1GAAC1', 'CKSAAGP1GDPC1', 'CKSAAGP1Geary1', 'CKSAAGP1GTPC1', 'CKSAAGP1Moran1', 'CKSAAGP1NMBroto1', 'CKSAAGP1TPC1', 'CKSAAP1CTDC1', 'CKSAAP1CTDD1', 'CKSAAP1CTDT1', 'CKSAAP1DDE1', 'CKSAAP1DPC1', 'CKSAAP1GAAC1', 'CKSAAP1GDPC1', 'CKSAAP1Geary1', 'CKSAAP1GTPC1', 'CKSAAP1Moran1', 'CKSAAP1NMBroto1', 'CKSAAP1TPC1', 'CTDC1CTDD1', 'CTDC1CTDT1', 'CTDC1DDE1', 'CTDC1DPC1', 'CTDC1GAAC1', 'CTDC1GDPC1', 'CTDC1Geary1', 'CTDC1GTPC1', 'CTDC1Moran1', 'CTDC1NMBroto1', 'CTDC1TPC1', 'CTDD1CTDT1', 'CTDD1DDE1', 'CTDD1DPC1', 'CTDD1GAAC1', 'CTDD1GDPC1', 'CTDD1Geary1', 'CTDD1GTPC1', 'CTDD1Moran1', 'CTDD1NMBroto1', 'CTDD1TPC1', 'CTDT1DDE1', 'CTDT1DPC1', 'CTDT1GAAC1', 'CTDT1GDPC1', 'CTDT1Geary1', 'CTDT1GTPC1', 'CTDT1Moran1', 'CTDT1NMBroto1', 'CTDT1TPC1', 'DDE1DPC1', 'DDE1GAAC1', 'DDE1GDPC1', 'DDE1Geary1', 'DDE1GTPC1', 'DDE1Moran1', 'DDE1NMBroto1',
    # feature = ['DDE1TPC1', 'DPC1GAAC1', 'DPC1GDPC1', 'DPC1Geary1', 'DPC1GTPC1', 'DPC1Moran1', 'DPC1NMBroto1', 'DPC1TPC1', 'GAAC1GDPC1', 'GAAC1Geary1', 'GAAC1GTPC1', 'GAAC1Moran1', 'GAAC1NMBroto1', 'GAAC1TPC1', 'GDPC1Geary1', 'GDPC1GTPC1', 'GDPC1Moran1', 'GDPC1NMBroto1', 'GDPC1TPC1', 'Geary1GTPC1', 'Geary1Moran1', 'Geary1NMBroto1', 'Geary1TPC1', 'GTPC1Moran1', 'GTPC1NMBroto1', 'GTPC1TPC1', 'Moran1NMBroto1', 'Moran1TPC1', 'NMBroto1TPC1', ]
    # feature = ['AAC1CKSAAGP1CKSAAP1', 'AAC1CKSAAGP1CTDC1', 'AAC1CKSAAGP1CTDD1', 'AAC1CKSAAGP1CTDT1', 'AAC1CKSAAGP1DDE1', 'AAC1CKSAAGP1DPC1', 'AAC1CKSAAGP1GAAC1', 'AAC1CKSAAGP1GDPC1', 'AAC1CKSAAGP1Geary1', 'AAC1CKSAAGP1GTPC1', 'AAC1CKSAAGP1Moran1', 'AAC1CKSAAGP1NMBroto1', 'AAC1CKSAAGP1TPC1', 'AAC1CKSAAP1CTDC1', 'AAC1CKSAAP1CTDD1', 'AAC1CKSAAP1CTDT1', 'AAC1CKSAAP1DDE1', 'AAC1CKSAAP1DPC1', 'AAC1CKSAAP1GAAC1', 'AAC1CKSAAP1GDPC1', 'AAC1CKSAAP1Geary1', 'AAC1CKSAAP1GTPC1', 'AAC1CKSAAP1Moran1', 'AAC1CKSAAP1NMBroto1', 'AAC1CKSAAP1TPC1', 'AAC1CTDC1CTDD1', 'AAC1CTDC1CTDT1', 'AAC1CTDC1DDE1', 'AAC1CTDC1DPC1', 'AAC1CTDC1GAAC1', 'AAC1CTDC1GDPC1', 'AAC1CTDC1Geary1', 'AAC1CTDC1GTPC1', 'AAC1CTDC1Moran1', 'AAC1CTDC1NMBroto1', 'AAC1CTDC1TPC1', 'AAC1CTDD1CTDT1', 'AAC1CTDD1DDE1', 'AAC1CTDD1DPC1', 'AAC1CTDD1GAAC1', 'AAC1CTDD1GDPC1', 'AAC1CTDD1Geary1', 'AAC1CTDD1GTPC1', 'AAC1CTDD1Moran1', 'AAC1CTDD1NMBroto1', 'AAC1CTDD1TPC1', 'AAC1CTDT1DDE1', 'AAC1CTDT1DPC1', 'AAC1CTDT1GAAC1', 'AAC1CTDT1GDPC1', 'AAC1CTDT1Geary1', 'AAC1CTDT1GTPC1', 'AAC1CTDT1Moran1', 'AAC1CTDT1NMBroto1', 'AAC1CTDT1TPC1', 'AAC1DDE1DPC1', 'AAC1DDE1GAAC1', 'AAC1DDE1GDPC1', 'AAC1DDE1Geary1', 'AAC1DDE1GTPC1', 'AAC1DDE1Moran1', 'AAC1DDE1NMBroto1', 'AAC1DDE1TPC1', 'AAC1DPC1GAAC1', 'AAC1DPC1GDPC1', 'AAC1DPC1Geary1', 'AAC1DPC1GTPC1', 'AAC1DPC1Moran1', 'AAC1DPC1NMBroto1', 'AAC1DPC1TPC1', 'AAC1GAAC1GDPC1', 'AAC1GAAC1Geary1', 'AAC1GAAC1GTPC1', 'AAC1GAAC1Moran1', 'AAC1GAAC1NMBroto1',
    # feature = ['AAC1GAAC1TPC1', 'AAC1GDPC1Geary1', 'AAC1GDPC1GTPC1', 'AAC1GDPC1Moran1', 'AAC1GDPC1NMBroto1', 'AAC1GDPC1TPC1', 'AAC1Geary1GTPC1', 'AAC1Geary1Moran1', 'AAC1Geary1NMBroto1', 'AAC1Geary1TPC1', 'AAC1GTPC1Moran1', 'AAC1GTPC1NMBroto1', 'AAC1GTPC1TPC1', 'AAC1Moran1NMBroto1', 'AAC1Moran1TPC1', 'AAC1NMBroto1TPC1', 'CKSAAGP1CKSAAP1CTDC1', 'CKSAAGP1CKSAAP1CTDD1', 'CKSAAGP1CKSAAP1CTDT1', 'CKSAAGP1CKSAAP1DDE1', 'CKSAAGP1CKSAAP1DPC1', 'CKSAAGP1CKSAAP1GAAC1', 'CKSAAGP1CKSAAP1GDPC1', 'CKSAAGP1CKSAAP1Geary1', 'CKSAAGP1CKSAAP1GTPC1', 'CKSAAGP1CKSAAP1Moran1', 'CKSAAGP1CKSAAP1NMBroto1', 'CKSAAGP1CKSAAP1TPC1', 'CKSAAGP1CTDC1CTDD1', 'CKSAAGP1CTDC1CTDT1', 'CKSAAGP1CTDC1DDE1', 'CKSAAGP1CTDC1DPC1', 'CKSAAGP1CTDC1GAAC1', 'CKSAAGP1CTDC1GDPC1', 'CKSAAGP1CTDC1Geary1', 'CKSAAGP1CTDC1GTPC1', 'CKSAAGP1CTDC1Moran1', 'CKSAAGP1CTDC1NMBroto1', 'CKSAAGP1CTDC1TPC1', 'CKSAAGP1CTDD1CTDT1', 'CKSAAGP1CTDD1DDE1', 'CKSAAGP1CTDD1DPC1', 'CKSAAGP1CTDD1GAAC1', 'CKSAAGP1CTDD1GDPC1', 'CKSAAGP1CTDD1Geary1', 'CKSAAGP1CTDD1GTPC1', 'CKSAAGP1CTDD1Moran1', 'CKSAAGP1CTDD1NMBroto1', 'CKSAAGP1CTDD1TPC1', 'CKSAAGP1CTDT1DDE1', 'CKSAAGP1CTDT1DPC1', 'CKSAAGP1CTDT1GAAC1', 'CKSAAGP1CTDT1GDPC1', 'CKSAAGP1CTDT1Geary1', 'CKSAAGP1CTDT1GTPC1', 'CKSAAGP1CTDT1Moran1', 'CKSAAGP1CTDT1NMBroto1', 'CKSAAGP1CTDT1TPC1', 'CKSAAGP1DDE1DPC1', 'CKSAAGP1DDE1GAAC1', 'CKSAAGP1DDE1GDPC1', 'CKSAAGP1DDE1Geary1', 'CKSAAGP1DDE1GTPC1', 'CKSAAGP1DDE1Moran1', 'CKSAAGP1DDE1NMBroto1', 'CKSAAGP1DDE1TPC1', 'CKSAAGP1DPC1GAAC1', 'CKSAAGP1DPC1GDPC1', 'CKSAAGP1DPC1Geary1', 'CKSAAGP1DPC1GTPC1', 'CKSAAGP1DPC1Moran1', 'CKSAAGP1DPC1NMBroto1',
    # feature = ['CKSAAGP1DPC1TPC1', 'CKSAAGP1GAAC1GDPC1', 'CKSAAGP1GAAC1Geary1', 'CKSAAGP1GAAC1GTPC1', 'CKSAAGP1GAAC1Moran1', 'CKSAAGP1GAAC1NMBroto1', 'CKSAAGP1GAAC1TPC1', 'CKSAAGP1GDPC1Geary1', 'CKSAAGP1GDPC1GTPC1', 'CKSAAGP1GDPC1Moran1', 'CKSAAGP1GDPC1NMBroto1', 'CKSAAGP1GDPC1TPC1', 'CKSAAGP1Geary1GTPC1', 'CKSAAGP1Geary1Moran1', 'CKSAAGP1Geary1NMBroto1', 'CKSAAGP1Geary1TPC1', 'CKSAAGP1GTPC1Moran1', 'CKSAAGP1GTPC1NMBroto1', 'CKSAAGP1GTPC1TPC1', 'CKSAAGP1Moran1NMBroto1', 'CKSAAGP1Moran1TPC1', 'CKSAAGP1NMBroto1TPC1', 'CKSAAP1CTDC1CTDD1', 'CKSAAP1CTDC1CTDT1', 'CKSAAP1CTDC1DDE1', 'CKSAAP1CTDC1DPC1', 'CKSAAP1CTDC1GAAC1', 'CKSAAP1CTDC1GDPC1', 'CKSAAP1CTDC1Geary1', 'CKSAAP1CTDC1GTPC1', 'CKSAAP1CTDC1Moran1', 'CKSAAP1CTDC1NMBroto1', 'CKSAAP1CTDC1TPC1', 'CKSAAP1CTDD1CTDT1', 'CKSAAP1CTDD1DDE1', 'CKSAAP1CTDD1DPC1', 'CKSAAP1CTDD1GAAC1', 'CKSAAP1CTDD1GDPC1', 'CKSAAP1CTDD1Geary1', 'CKSAAP1CTDD1GTPC1', 'CKSAAP1CTDD1Moran1', 'CKSAAP1CTDD1NMBroto1', 'CKSAAP1CTDD1TPC1', 'CKSAAP1CTDT1DDE1', 'CKSAAP1CTDT1DPC1', 'CKSAAP1CTDT1GAAC1', 'CKSAAP1CTDT1GDPC1', 'CKSAAP1CTDT1Geary1', 'CKSAAP1CTDT1GTPC1', 'CKSAAP1CTDT1Moran1', 'CKSAAP1CTDT1NMBroto1', 'CKSAAP1CTDT1TPC1', 'CKSAAP1DDE1DPC1', 'CKSAAP1DDE1GAAC1', 'CKSAAP1DDE1GDPC1', 'CKSAAP1DDE1Geary1', 'CKSAAP1DDE1GTPC1', 'CKSAAP1DDE1Moran1', 'CKSAAP1DDE1NMBroto1', 'CKSAAP1DDE1TPC1', 'CKSAAP1DPC1GAAC1', 'CKSAAP1DPC1GDPC1', 'CKSAAP1DPC1Geary1', 'CKSAAP1DPC1GTPC1', 'CKSAAP1DPC1Moran1', 'CKSAAP1DPC1NMBroto1',
    # feature=[ 'CKSAAP1DPC1TPC1', 'CKSAAP1GAAC1GDPC1', 'CKSAAP1GAAC1Geary1', 'CKSAAP1GAAC1GTPC1', 'CKSAAP1GAAC1Moran1', 'CKSAAP1GAAC1NMBroto1', 'CKSAAP1GAAC1TPC1', 'CKSAAP1GDPC1Geary1', 'CKSAAP1GDPC1GTPC1', 'CKSAAP1GDPC1Moran1', 'CKSAAP1GDPC1NMBroto1', 'CKSAAP1GDPC1TPC1', 'CKSAAP1Geary1GTPC1', 'CKSAAP1Geary1Moran1', 'CKSAAP1Geary1NMBroto1', 'CKSAAP1Geary1TPC1', 'CKSAAP1GTPC1Moran1', 'CKSAAP1GTPC1NMBroto1', 'CKSAAP1GTPC1TPC1', 'CKSAAP1Moran1NMBroto1', 'CKSAAP1Moran1TPC1', 'CKSAAP1NMBroto1TPC1', 'CTDC1CTDD1CTDT1', 'CTDC1CTDD1DDE1', 'CTDC1CTDD1DPC1', 'CTDC1CTDD1GAAC1', 'CTDC1CTDD1GDPC1', 'CTDC1CTDD1Geary1', 'CTDC1CTDD1GTPC1', 'CTDC1CTDD1Moran1', 'CTDC1CTDD1NMBroto1', 'CTDC1CTDD1TPC1', 'CTDC1CTDT1DDE1', 'CTDC1CTDT1DPC1', 'CTDC1CTDT1GAAC1', 'CTDC1CTDT1GDPC1', 'CTDC1CTDT1Geary1', 'CTDC1CTDT1GTPC1', 'CTDC1CTDT1Moran1', 'CTDC1CTDT1NMBroto1', 'CTDC1CTDT1TPC1', 'CTDC1DDE1DPC1', 'CTDC1DDE1GAAC1', 'CTDC1DDE1GDPC1', 'CTDC1DDE1Geary1', 'CTDC1DDE1GTPC1', 'CTDC1DDE1Moran1', 'CTDC1DDE1NMBroto1', 'CTDC1DDE1TPC1', 'CTDC1DPC1GAAC1', 'CTDC1DPC1GDPC1', 'CTDC1DPC1Geary1', 'CTDC1DPC1GTPC1', 'CTDC1DPC1Moran1', 'CTDC1DPC1NMBroto1', 'CTDC1DPC1TPC1', 'CTDC1GAAC1GDPC1', 'CTDC1GAAC1Geary1', 'CTDC1GAAC1GTPC1', 'CTDC1GAAC1Moran1', 'CTDC1GAAC1NMBroto1', 'CTDC1GAAC1TPC1', 'CTDC1GDPC1Geary1', 'CTDC1GDPC1GTPC1', 'CTDC1GDPC1Moran1', 'CTDC1GDPC1NMBroto1', 'CTDC1GDPC1TPC1', 'CTDC1Geary1GTPC1', 'CTDC1Geary1Moran1', 'CTDC1Geary1NMBroto1', 'CTDC1Geary1TPC1', 'CTDC1GTPC1Moran1', 'CTDC1GTPC1NMBroto1',
    #feature=[ 'CTDC1GTPC1TPC1', 'CTDC1Moran1NMBroto1', 'CTDC1Moran1TPC1', 'CTDC1NMBroto1TPC1', 'CTDD1CTDT1DDE1', 'CTDD1CTDT1DPC1', 'CTDD1CTDT1GAAC1', 'CTDD1CTDT1GDPC1', 'CTDD1CTDT1Geary1', 'CTDD1CTDT1GTPC1', 'CTDD1CTDT1Moran1', 'CTDD1CTDT1NMBroto1', 'CTDD1CTDT1TPC1', 'CTDD1DDE1DPC1', 'CTDD1DDE1GAAC1', 'CTDD1DDE1GDPC1', 'CTDD1DDE1Geary1', 'CTDD1DDE1GTPC1', 'CTDD1DDE1Moran1', 'CTDD1DDE1NMBroto1', 'CTDD1DDE1TPC1', 'CTDD1DPC1GAAC1', 'CTDD1DPC1GDPC1', 'CTDD1DPC1Geary1', 'CTDD1DPC1GTPC1', 'CTDD1DPC1Moran1', 'CTDD1DPC1NMBroto1', 'CTDD1DPC1TPC1', 'CTDD1GAAC1GDPC1', 'CTDD1GAAC1Geary1', 'CTDD1GAAC1GTPC1', 'CTDD1GAAC1Moran1', 'CTDD1GAAC1NMBroto1', 'CTDD1GAAC1TPC1', 'CTDD1GDPC1Geary1', 'CTDD1GDPC1GTPC1', 'CTDD1GDPC1Moran1', 'CTDD1GDPC1NMBroto1', 'CTDD1GDPC1TPC1', 'CTDD1Geary1GTPC1', 'CTDD1Geary1Moran1', 'CTDD1Geary1NMBroto1', 'CTDD1Geary1TPC1', 'CTDD1GTPC1Moran1', 'CTDD1GTPC1NMBroto1', 'CTDD1GTPC1TPC1', 'CTDD1Moran1NMBroto1', 'CTDD1Moran1TPC1', 'CTDD1NMBroto1TPC1', 'CTDT1DDE1DPC1', 'CTDT1DDE1GAAC1', 'CTDT1DDE1GDPC1', 'CTDT1DDE1Geary1', 'CTDT1DDE1GTPC1', 'CTDT1DDE1Moran1', 'CTDT1DDE1NMBroto1', 'CTDT1DDE1TPC1', 'CTDT1DPC1GAAC1', 'CTDT1DPC1GDPC1', 'CTDT1DPC1Geary1', 'CTDT1DPC1GTPC1', 'CTDT1DPC1Moran1', 'CTDT1DPC1NMBroto1', 'CTDT1DPC1TPC1', 'CTDT1GAAC1GDPC1', 'CTDT1GAAC1Geary1', 'CTDT1GAAC1GTPC1', 'CTDT1GAAC1Moran1', 'CTDT1GAAC1NMBroto1', 'CTDT1GAAC1TPC1', 'CTDT1GDPC1Geary1', 'CTDT1GDPC1GTPC1', 'CTDT1GDPC1Moran1', 'CTDT1GDPC1NMBroto1', 'CTDT1GDPC1TPC1', 'CTDT1Geary1GTPC1', 'CTDT1Geary1Moran1', 'CTDT1Geary1NMBroto1', 'CTDT1Geary1TPC1', 'CTDT1GTPC1Moran1', 'CTDT1GTPC1NMBroto1', 'CTDT1GTPC1TPC1', 'CTDT1Moran1NMBroto1', 'CTDT1Moran1TPC1', 'CTDT1NMBroto1TPC1', 'DDE1DPC1GAAC1', 'DDE1DPC1GDPC1', 'DDE1DPC1Geary1', 'DDE1DPC1GTPC1', 'DDE1DPC1Moran1', 'DDE1DPC1NMBroto1', 'DDE1DPC1TPC1', 'DDE1GAAC1GDPC1', 'DDE1GAAC1Geary1', 'DDE1GAAC1GTPC1', 'DDE1GAAC1Moran1', 'DDE1GAAC1NMBroto1', 'DDE1GAAC1TPC1', 'DDE1GDPC1Geary1', 'DDE1GDPC1GTPC1', 'DDE1GDPC1Moran1', 'DDE1GDPC1NMBroto1', 'DDE1GDPC1TPC1', 'DDE1Geary1GTPC1', 'DDE1Geary1Moran1', 'DDE1Geary1NMBroto1', 'DDE1Geary1TPC1', 'DDE1GTPC1Moran1', 'DDE1GTPC1NMBroto1', 'DDE1GTPC1TPC1', 'DDE1Moran1NMBroto1', 'DDE1Moran1TPC1', 'DDE1NMBroto1TPC1', 'DPC1GAAC1GDPC1', 'DPC1GAAC1Geary1', 'DPC1GAAC1GTPC1', 'DPC1GAAC1Moran1', 'DPC1GAAC1NMBroto1', 'DPC1GAAC1TPC1', 'DPC1GDPC1Geary1', 'DPC1GDPC1GTPC1', 'DPC1GDPC1Moran1', 'DPC1GDPC1NMBroto1', 'DPC1GDPC1TPC1', 'DPC1Geary1GTPC1', 'DPC1Geary1Moran1', 'DPC1Geary1NMBroto1', 'DPC1Geary1TPC1', 'DPC1GTPC1Moran1', 'DPC1GTPC1NMBroto1', 'DPC1GTPC1TPC1', 'DPC1Moran1NMBroto1', 'DPC1Moran1TPC1', 'DPC1NMBroto1TPC1', 'GAAC1GDPC1Geary1', 'GAAC1GDPC1GTPC1', 'GAAC1GDPC1Moran1', 'GAAC1GDPC1NMBroto1', 'GAAC1GDPC1TPC1', 'GAAC1Geary1GTPC1', 'GAAC1Geary1Moran1', 'GAAC1Geary1NMBroto1', 'GAAC1Geary1TPC1', 'GAAC1GTPC1Moran1', 'GAAC1GTPC1NMBroto1', 'GAAC1GTPC1TPC1', 'GAAC1Moran1NMBroto1', 'GAAC1Moran1TPC1', 'GAAC1NMBroto1TPC1', 'GDPC1Geary1GTPC1', 'GDPC1Geary1Moran1', 'GDPC1Geary1NMBroto1', 'GDPC1Geary1TPC1', 'GDPC1GTPC1Moran1', 'GDPC1GTPC1NMBroto1', 'GDPC1GTPC1TPC1', 'GDPC1Moran1NMBroto1', 'GDPC1Moran1TPC1', 'GDPC1NMBroto1TPC1', 'Geary1GTPC1Moran1', 'Geary1GTPC1NMBroto1', 'Geary1GTPC1TPC1', 'Geary1Moran1NMBroto1', 'Geary1Moran1TPC1', 'Geary1NMBroto1TPC1', 'GTPC1Moran1NMBroto1', 'GTPC1Moran1TPC1', 'GTPC1NMBroto1TPC1', 'Moran1NMBroto1TPC1', ]
    
    
    #feature = []
    #feature.append(sys.argv[1])
    # feature = ['asdc','cksnap', 'dac', 'dpcp', 'dpcp2', 'geary', 'kmer', 'mismatch', 'mmi', 'moran',
    # feature = [ 'kmer', 'mismatch','rckmer']
    # feature = [           'nac',
    # feature = ['nmbroto','pseDNC', 'pseEIIP', 'pseknc', 'rckmer', 'subsequence', 'tac', 'tpcp']
    # feature = ['kr', 'km', 'mr', 'kmr']
    # feature = ['2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    # feature = ['12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
    # feature = ['liuqi1']
    # features = [ 'KMER', 'PC-PseAAC', 'PC-PseAAC-General', 'SC-PseAAC','SC-PseAAC-General']
    # features = ['AC_ACC', 'AC_CC', 'AC_DP', 'AC_DR', 'AC_KMER', 'AC_PC_PseAAC', 'AC_PC_PseAAC_General',
    #             'AC_PDT', 'AC_SC_PseAAC', 'AC_SC_PseAAC_General', 'ACC_CC', 'ACC_DP', 'ACC_DR', 'ACC_KMER',
    #             'ACC_PC_PseAAC', 'ACC_PC_PseAAC_General', 'ACC_PDT', 'ACC_SC_PseAAC', 'ACC_SC_PseAAC_General',
    #             'CC_DP', 'CC_DR', 'CC_KMER', 'CC_PC_PseAAC', 'CC_PC_PseAAC_General', 'CC_PDT', 'CC_SC_PseAAC',
    #             'CC_SC_PseAAC_General', 'DP_DR', 'DP_KMER', 'DP_PC_PseAAC', 'DP_PC_PseAAC_General', 'DP_PDT',
    #             'DP_SC_PseAAC', 'DP_SC_PseAAC_General', 'DR_KMER', 'DR_PC_PseAAC', 'DR_PC_PseAAC_General',
    #             'DR_PDT', 'DR_SC_PseAAC', 'DR_SC_PseAAC_General', 'KMER_PC_PseAAC', 'KMER_PC_PseAAC_General',
    #             'KMER_PDT', 'KMER_SC_PseAAC', 'KMER_SC_PseAAC_General', 'PC_PseAAC_PC_PseAAC_General',
    #             'PC_PseAAC_PDT', 'PC_PseAAC_SC_PseAAC', 'PC_PseAAC_SC_PseAAC_General', 'PC_PseAAC_General_PDT',
    #             'PC_PseAAC_General_SC_PseAAC', 'PC_PseAAC_General_SC_PseAAC_General', 'PDT_SC_PseAAC',
    #             'PDT_SC_PseAAC_General', 'SC_PseAAC_SC_PseAAC_General']
    # feature = ['QSOrder','GDPC','GAAC','CKSAAGP','ACC']
    # dataset = 'Active_site_'
    # feature = ['ANF','ASDC', 'CKSNAP','DAC','DPCP',
    # feature = ['EIIP','ENAC','Kmer','mismatch','MMI','NAC','NCP','PseEIIP','PSTNPds','PSTNPss','RCKmer','Subsequence']
    # feature =['subsequence1','kmer1','RCKmer1','PseKNC1','NCP1','NAC1','ENAC1','CKSNAP1','ANF1' ]
    # feature = ['all2a','all3a','all4a']
    # feature = ['CKSAAGP_KPCA','CKSAAGP_tsne','CKSAAGP_POD','CKSAAP_KPCA','CKSAAP_tsne','CKSAAP_POD','DPC_KPCA','DPC_tsne','DPC_POD','GAAC_KPCA','GAAC_tsne','GAAC_POD']
    # feature = ['Moran_KPCA','Moran_tsne','Moran_POD','TPC_KPCA','TPC_tsne','TPC_POD']
    # feature = ['CKSAAGP_POD','CSKAAP_POD','DPC_POD','DPC_tsne','GAAC_POD','Geary_POD','Moran_POD','TCP_POD']
    # feature = ['PPTAPE_1POD','PPTAPE_8POD','PPTAPE_9POD','PPTAPE_10POD','PPTAPE_11POD','PPTAPE_15POD','PPTAPE_20POD','PPTAPE_25POD','PPTAPE_1tsne','PPTAPE_8tsne','PPTAPE_9tsne','PPTAPE_10tsne','PPTAPE_11tsne','PPTAPE_15tsne','PPTAPE_20tsne','PPTAPE_25tsne']
    # feature = ['TAPE2_1POD','TAPE2_8POD','TAPE2_9POD','TAPE2_10POD','TAPE2_11POD','TAPE2_12POD','TAPE2_15POD','TAPE2_20POD','TAPE2_25POD','TAPE2_1tsne','TAPE2_8tsne','TAPE2_9tsne','TAPE2_10tsne','TAPE2_11tsne','TAPE2_12tsne','TAPE2_15tsne','TAPE2_20tsne','TAPE2_25tsne','TAPE2_1tsne','TAPE2_8tsne','TAPE2_9tsne','TAPE2_10tsne','TAPE2_11tsne','TAPE2_12tsne','TAPE2_15tsne','TAPE2_20tsne','TAPE2_25tsne','TAPE2fscnca3_tsne','TAPE2fscnca4_tsne','TAPE2fscnca5_tsne','TAPE2fscnca6_tsne','TAPE2mrmr2_tsne','TAPE2mrmr3_tsne','TAPE2mrmr4_tsne','TAPE2mrmr5_tsne','TAPE2_tsne','TAPE8_tsne','TAPE9_tsne','TAPE10_tsne','TAPE11_tsne','TAPE12_tsne','TAPE15_tsne','TAPE20_tsne','TAPE25_tsne','TAPEfscnca3_tsne','TAPEfscnca4_tsne','TAPEfscnca5_tsne','TAPEfscnca6_tsne','TAPEmrmr2_tsne','TAPEmrmr3_tsne','TAPEmrmr4_tsne','TAPEmrmr5_tsne','TAPE2fscnca3_POD','TAPE2fscnca4_POD','TAPE2fscnca5_POD','TAPE2fscnca6_POD','TAPE2mrmr2_POD','TAPE2mrmr3_POD','TAPE2mrmr4_POD','TAPE2mrmr5_POD','TAPE2_POD','TAPE8_POD','TAPE9_POD','TAPE10_POD','TAPE11_POD','TAPE12_POD','TAPE15_POD','TAPE20_POD','TAPE25_POD','TAPEfscnca3_POD','TAPEfscnca4_POD','TAPEfscnca5_POD','TAPEfscnca6_POD','TAPEmrmr2_POD','TAPEmrmr3_POD','TAPEmrmr4_POD','TAPEmrmr5_POD']
    # feature=['TAPE2fscnca_3','TAPE2fscnca_4','TAPE2fscnca_5','TAPE2fscnca_6','TAPE2mrmr_2','TAPE2mrmr_3','TAPE2mrmr_4','TAPE2mrmr_5','PPTAPEfscnca_3','PPTAPEfscnca_4','PPTAPEfscnca_5','PPTAPEfscnca_6','PPTAPEmrmr_4','PPTAPEmrmr_5','TAPEfscnca_3','TAPEfscnca_4','TAPEfscnca_5','TAPEfscnca_6','TAPE2mrmr_2','TAPE2mrmr_3','TAPE2mrmr_4','TAPE2mrmr_5','PPTAPEmrmr_2','PPTAPEmrmr_3','PPTAPEmrmr_4','PPTAPEmrmr_5','TAPEmrmr_2','TAPEmrmr_3','TAPEmrmr_4','TAPEmrmr_5','PPTAPE_15','PPTAPE_20','PPTAPE_25','TAPE2_15left','TAPE2_1left','TAPE2_9','TAPE2_8','TAPE2_25','TAPE2_15','TAPE2_12','TAPE2','PPTAPE','PPTAPE_1',]
    # feature = ['PPTAPE_8','PPTAPE_9','PPTAPE_10','PPTAPE_11','PPTAPE_12','PPTAPE_15','PPTAPE_20','PPTAPE_25','PPTAPE_30','TAPEall','TAPE1left','TAPE8','TAPE9','TAPE10','TAPE11','TAPE12','TAPE15','TAPE15left','TAPE20','TAPE25','TAPE30']
    # feature=['AAC']
    # feature = ['all1']
    # feature=['jianfa','max','min','average','sum']
    # feature=['TAPE']
    # feature=['ASDC','ASDC','MMI','NAC','NCP','PCPseDNC','PS2','PseDNC','PseKNC','SCPseDNC','Subsequence']#'DAC','DACC','DCC','DPCP','ENAC','LPDF'
    # feature = ['Moran', 'Geary', 'GAAC', 'DPC', 'CKSAAP', 'CKSAAGP', 'TPC']
    # feature = ['SOCNumber2','QSOrder2','PAAC2','KSTraid2','GTPC2','GDPC2','GAAC2','EGAAC2','EAAC2','DPC2','DDE2','CTDT2','CTDD2','CTDC2','CKSAAP2','CKSAAGP2','BLOSUM622','APAAC2','AAC2']
    # feature = ['DPC','TPC','CKSAAGP','CKSAAP','GAAC','Moran','Geary']
    # feature = ['ASDC','DPC','BLOSUM62','CKSAAGP','EAAC']
    # feature = ['AAC','APAAC','ASDC','BLOSUM62','CKSAAGP','CKSAAP','EAAC','EGAAC','GAAC',
    # feature = ['PAAC','PseKRAAC1','PseKRAAC2','PseKRAAC3A','PseKRAAC3B','PseKRAAC4','PseKRAAC5','PseKRAAC6A','PseKRAAC6B','PseKRAAC6C','PseKRAAC7','PseKRAAC8','PseKRAAC9',
    # feature = ['PseKRAAC10','PseKRAAC11','PseKRAAC12','PseKRAAC13',]
    # feature = ['PseKRAAC14','PseKRAAC15','PseKRAAC16']
    # feature = ['exponential_firstorder', 'exponential_glcm', 'exponential_gldm', 'exponential_glrlm', 'exponential_glszm', 'exponential_ngtdm', 'gradient_firstorder', 'gradient_glcm', 'gradient_gldm', 'gradient_glrlm', 'gradient_glszm', 'gradient_ngtdm', 'lbp_3D_k_firstorder', 'lbp_3D_k_glcm', 'lbp_3D_k_gldm', 'lbp_3D_k_glrlm', 'lbp_3D_k_glszm', 'lbp_3D_k_ngtdm', 'lbp_3D_m1_firstorder', 'lbp_3D_m1_glcm', 'lbp_3D_m1_gldm', 'lbp_3D_m1_glrlm', 'lbp_3D_m1_glszm', 'lbp_3D_m1_ngtdm', 'lbp_3D_m2_firstorder', 'lbp_3D_m2_glcm', 'lbp_3D_m2_gldm', 'lbp_3D_m2_glrlm', 'lbp_3D_m2_glszm', 'lbp_3D_m2_ngtdm', 'log_sigma_1_0_mm_3D_firstorder', 'log_sigma_1_0_mm_3D_glcm', 'log_sigma_1_0_mm_3D_gldm', 'log_sigma_1_0_mm_3D_glrlm', 'log_sigma_1_0_mm_3D_glszm', 'log_sigma_1_0_mm_3D_ngtdm', 'log_sigma_2_0_mm_3D_firstorder', 'log_sigma_2_0_mm_3D_glcm', 'log_sigma_2_0_mm_3D_gldm', 'log_sigma_2_0_mm_3D_glrlm', 'log_sigma_2_0_mm_3D_glszm', 'log_sigma_2_0_mm_3D_ngtdm', 'log_sigma_3_0_mm_3D_firstorder', 'log_sigma_3_0_mm_3D_glcm', 'log_sigma_3_0_mm_3D_gldm', 'log_sigma_3_0_mm_3D_glrlm', 'log_sigma_3_0_mm_3D_glszm', 'log_sigma_3_0_mm_3D_ngtdm', 'logarithm_firstorder', 'logarithm_glcm', 'logarithm_gldm', 'logarithm_glrlm', 'logarithm_glszm', 'logarithm_ngtdm', 'original_firstorder', 'original_glcm', 'original_gldm', 'original_glrlm', 'original_glszm', 'original_ngtdm', 'original_shape', 'square_firstorder', 'square_glcm', 'square_gldm', 'square_glrlm', 'square_glszm', 'square_ngtdm', 'squareroot_firstorder', 'squareroot_glcm', 'squareroot_gldm', 'squareroot_glrlm', 'squareroot_glszm', 'squareroot_ngtdm', 'wavelet_HHH_firstorder', 'wavelet_HHH_glcm', 'wavelet_HHH_gldm', 'wavelet_HHH_glrlm', 'wavelet_HHH_glszm', 'wavelet_HHH_ngtdm', 'wavelet_HHL_firstorder', 'wavelet_HHL_glcm', 'wavelet_HHL_gldm', 'wavelet_HHL_glrlm', 'wavelet_HHL_glszm', 'wavelet_HHL_ngtdm', 'wavelet_HLH_firstorder', 'wavelet_HLH_glcm', 'wavelet_HLH_gldm', 'wavelet_HLH_glrlm', 'wavelet_HLH_glszm', 'wavelet_HLH_ngtdm', 'wavelet_HLL_firstorder', 'wavelet_HLL_glcm', 'wavelet_HLL_gldm', 'wavelet_HLL_glrlm', 'wavelet_HLL_glszm', 'wavelet_HLL_ngtdm', 'wavelet_LHH_firstorder', 'wavelet_LHH_glcm', 'wavelet_LHH_gldm', 'wavelet_LHH_glrlm', 'wavelet_LHH_glszm', 'wavelet_LHH_ngtdm', 'wavelet_LHL_firstorder', 'wavelet_LHL_glcm', 'wavelet_LHL_gldm', 'wavelet_LHL_glrlm', 'wavelet_LHL_glszm', 'wavelet_LHL_ngtdm', 'wavelet_LLH_firstorder', 'wavelet_LLH_glcm', 'wavelet_LLH_gldm', 'wavelet_LLH_glrlm', 'wavelet_LLH_glszm', 'wavelet_LLH_ngtdm', 'wavelet_LLL_firstorder', 'wavelet_LLL_glcm', 'wavelet_LLL_gldm', 'wavelet_LLL_glrlm', 'wavelet_LLL_glszm', 'wavelet_LLL_ngtdm']  # feature = ['ASDC1','BLOSUM621','CKSAAGP1','CKSAAP1','DPC1','EAAC1','EGAAC1','GAAC1','GDPC1','GTPC1','PseKRAACtype11','PseKRAACtype21','PseKRAACtypeA31']
    # feature = ['all','all1','all2','all3','all4','all5','all6','all7','all8','all9','all10','all11','allfirst','allglcm','allgldm','allglrlm','allglszm','allgtdm']
    feature = ['NBNP_tsne','NBNP_tsne1']
    # feature = ['CKSAAGP','CKSAAP','DPC','GAAC','Geary','Moran','TCP']
    for feature in feature:

        fname = 'D:/' + feature + '.txt'

        # predict_label_result = 'D:/Code/图表征学习/GCNFrame/output/yepao/' + feature
        # fname = 'D:/细胞器蛋白质/图卷积/IPVP Graph/'+feature +'.txt'
        # fnameresult = 'D:/细胞器蛋白质/图卷积/IPVP Graph/' + feature + '_results.txt'
        # fnamefig = 'D:/细胞器蛋白质/图卷积/IPVP Graph/' + feature + 'w.png'
        # fname = 'D:/细胞器蛋白质/图卷积/IPVP Graph/' + feature + '.txt'
        # fname = 'D:/细胞器蛋白特征/降维/PVP/' + feature + '.txt'
        # fnameresult = 'D:/细胞器蛋白特征/降维/PVP/' + feature + '_resultsA.txt'
        # fnamefig = 'D:/细胞器蛋白特征/降维/PVP/' + feature + 'A.png'
        # fname = 'D:/细胞器蛋白特征/降维/PVP/' + feature + '.txt'


        # fname_train = 'D:/Code/图表征学习/GCNFrame/output/golgi/'+feature+'.txt'
        # fname_test = 'D:/Code/图表征学习/GCNFrame/output/mei/'+feature+'.txt'
        # fnameresult = 'D:/Code/图表征学习/GCNFrame/output/mei/'+feature+'_transformer_results.txt'
        # fnamefig = 'D:/Code/图表征学习/GCNFrame/output/mei/'+feature+'_transformer.png'
        # X_train, y_train = dataread(fname_train)
        # X_test, y_test = dataread(fname_test)

        # X, y = dataread(fname)

    # fname = 'D:/iLearnPlus-main/data/白芍/active site/AAC1CKSAAGP1.txt'
    # fnameresult = 'D:/iLearnPlus-main/data/白芍/active site/2阶/AAC1CKSAAGP1results.txt'
    # fnamefig = 'D:/iLearnPlus-main/data/白芍/active site/2阶/AAC1CKSAAGP1.png'
    # X,y = dataread(fname)  # 交叉验证导入
        X1,X0 = dataread(fname) #交叉验证导入
        y1=np.ones(len(X1))
        y0=np.zeros(len(X0))
        # X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=.1, random_state=0)#这个只跑一次，我现在想让他跑十次，所有的结果也都保存下来
        # X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=.1, random_state=0)
        # X_train = np.vstack((X0_train, X1_train))
        # X_test = np.vstack((X0_test, X1_test))
        # y_train = np.hstack((y0_train, y1_train))
        # y_test = np.hstack((y0_test, y1_test))
        kf = KFold(n_splits=10, shuffle=False, random_state=None)
        AA=1
        print(type(X1))
        # # 划分数据集
        for (train_index_0, test_index_0), (train_index_1, test_index_1) in zip(kf.split(X0), kf.split(X1)):
            # 对 x0 进行分割
            # print(type(train_index_0))
            # train_index_0=train_index_0.tolist()
            # test_index_0=test_index_0.tolist()
            X0_train, X0_test = X0[train_index_0], X0[test_index_0]
            y0_train, y0_test = y0[train_index_0], y0[test_index_0]

            # 对 x1 进行分割
            X1_train, X1_test = X1[train_index_1], X1[test_index_1]
            y1_train, y1_test = y1[train_index_1], y1[test_index_1]
        # for train1_index, test1_index in kf.split(X1):
        #     X1_train, X1_test = X1[train1_index], X1[test1_index]
        #     y1_train, y1_test = y1[train1_index], y1[test1_index]
        #     for train0_index, test0_index in kf.split(X0):
        #         X0_train, X0_test = X0[train0_index], X0[test0_index]
        #         y0_train, y0_test = y0[train0_index], y0[test0_index]
            X_train = np.vstack((X0_train, X1_train))
            X_test = np.vstack((X0_test, X1_test))
            y_train = np.hstack((y0_train, y1_train))
            y_test = np.hstack((y0_test, y1_test))


            fnameresult = 'D:/' + feature +str(AA)+ '@.txt'#
            fnamefig = 'D:/' + feature+str(AA) +'@.png'#+ str(AA)
            print(u'start 当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

            Linear_fpr, Linear_tpr, Linear_roc_auc = Linear(X_train, X_test, y_train, y_test)
            # SVC_fpr, SVC_tpr, SVC_roc_auc = SVC_T(X_train, X_test, y_train, y_test)
            CF_fpr, CF_tpr, CF_roc_auc = Cascade_Forest1(X_train, X_test, y_train, y_test)
            print(u'svc 当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            # SGD_fpr, SGD_tpr, SGD_roc_auc = SGD(X_train, X_test, y_train, y_test)
            # Neighbors_fpr, Neighbors_tpr, Neighbors_roc_auc = Neighbors(X_train, X_test, y_train, y_test)
            # GPC_fpr, GPC_tpr, GPC_roc_auc = GPC(X_train, X_test, y_train, y_test)
            GaussianNB_fpr, GaussianNB_tpr, GaussianNB_roc_auc = Gaussian_NB(X_train, X_test, y_train, y_test)
            # Bernoulli_fpr, Bernoulli_tpr, Bernoulli_roc_auc = Bernoulli_NB(X_train, X_test, y_train, y_test)
            Tree_fpr, Tree_tpr, Tree_roc_auc = DT(X_train, X_test, y_train, y_test)
            Bagging_fpr, Bagging_tpr, Bagging_roc_auc = Bagging(X_train, X_test, y_train, y_test)
            RandomForest_fpr, RandomForest_tpr, RandomForest_roc_auc = RandomForest(X_train, X_test, y_train, y_test)
            AdaBoost_fpr, AdaBoost_tpr, AdaBoost_roc_auc = AdaBoost(X_train, X_test, y_train, y_test)
            GradientBoosting_fpr, GradientBoosting_tpr, GradientBoosting_roc_auc = GradientBoosting(X_train, X_test, y_train,                                                                                                y_test)
            HGB_fpr, HGB_tpr, HGB_roc_auc = HistGradientBoosting(X_train, X_test, y_train, y_test)
            # CNNB_fpr, CNNB_tpr, CNNB_roc_auc = CNNBilstm(X_train, X_test, y_train, y_test)
            # CNNB_A_fpr, CNNB_A_tpr, CNNB_A_roc_auc = CNNBilstm_Attention(X_train, X_test, y_train, y_test)
            # TextCNN_fpr, TextCNN_tpr, TextCNN_roc_auc = TextCNN(X_train, X_test, y_train, y_test)
            # Graph_Code_fpr, Graph_Code_tpr, Graph_Code_roc_auc = Graph_Code(fname)
            MLPC_fpr, MLPC_tpr, MLPC_roc_auc = MLPClassifier_1(X_train, X_test, y_train, y_test)
            XG_fpr, XG_tpr, XG_roc_auc = XGboost(X_train, X_test, y_train, y_test)
            GBM_fpr, GBM_tpr, GBM_roc_auc = lightgbm(X_train, X_test, y_train, y_test)
            BLS_Cross_fpr, BLS_Cross_tpr, BLS_Cross_roc_auc  =BLS_Cross(X_train, X_test, y_train, y_test)

            # 绘图
            plt.figure(figsize=(10, 10))
            Roc("Linear Discriminant", Linear_fpr, Linear_tpr, Linear_roc_auc)
            # Roc("Support Vector Machine", SVC_fpr, SVC_tpr, SVC_roc_auc)
            Roc("Cascade Forest", CF_fpr, CF_tpr, CF_roc_auc)
            # Roc("Stochastic Gradient Descent", SGD_fpr, SGD_tpr, SGD_roc_auc)
            # Roc("K Nearest Neighbors",Neighbors_fpr, Neighbors_tpr, Neighbors_roc_auc)
            # Roc("Gaussian Processes", GPC_fpr, GPC_tpr, GPC_roc_auc)
            Roc("Gaussian Naive Bayes", GaussianNB_fpr, GaussianNB_tpr, GaussianNB_roc_auc)
            # Roc("Bernoulli Naive Bayes", Bernoulli_fpr, Bernoulli_tpr, Bernoulli_roc_auc)
            Roc("Decision Tree", Tree_fpr, Tree_tpr, Tree_roc_auc)
            Roc("Bagging", Bagging_fpr, Bagging_tpr, Bagging_roc_auc)
            Roc("Random Forest", RandomForest_fpr, RandomForest_tpr, RandomForest_roc_auc)
            Roc("AdaBoost", AdaBoost_fpr, AdaBoost_tpr, AdaBoost_roc_auc)
            Roc("Gradient Boosting", GradientBoosting_fpr, GradientBoosting_tpr, GradientBoosting_roc_auc)
            Roc("Hist Gradient Boosting", HGB_fpr, HGB_tpr, HGB_roc_auc)
            # Roc("CNNBilstm", CNNB_fpr, CNNB_tpr, CNNB_roc_auc)
            # Roc("CNNBilstm_Attention", CNNB_A_fpr, CNNB_A_tpr, CNNB_A_roc_auc)
            # Roc("TextCNN", TextCNN_fpr, TextCNN_tpr, TextCNN_roc_auc)
            # Roc("Graph_Code", Graph_Code_fpr, Graph_Code_tpr, Graph_Code_roc_auc)
            Roc("MLPClassifier_1", MLPC_fpr, MLPC_tpr, MLPC_roc_auc)
            Roc("XGboosting", XG_fpr, XG_tpr, XG_roc_auc)
            Roc("LightGBM", GBM_fpr, GBM_tpr, GBM_roc_auc)
            Roc("BLS", BLS_Cross_fpr, BLS_Cross_tpr, BLS_Cross_roc_auc)

            # #Roc("svm", fpr, tpr, roc_auc)
            plt.savefig(fnamefig, dpi=1000, bbox_inches='tight')
            print(u'end 当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            AA = AA + 1




        #plt.show()