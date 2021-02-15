import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils import data
from torch import nn
import time
from IPython import display
import seaborn as sns
import numpy as np
import random
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import cv2
import sys


if __name__ == "__main__":
    # load and preprocess data
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    csv_path = sys.argv[3]

    # load and preprocess data
    lr, epochs, batch_size, eta = 0.001, 150, 100, 100

    trans1 = transforms.Compose([transforms.ToTensor()])
    train_imgs = datasets.ImageFolder('../input/figure', transform=trans1)
    train_iter = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
    train_list = list(enumerate(train_iter))

    trans2 = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    test_imgs = datasets.ImageFolder('../input/stick-figure', transform=trans2)
    test_iter = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size, shuffle=False)
    test_list = list(enumerate(test_iter))

    # construct net
    class VGG(nn.Module):
        def __init__(self, input_channels, num_class):
            super().__init__()
            self.stage1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                                        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

            self.stage2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                        nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                        nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                        nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

            self.stage3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                        nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                        nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                        nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

            self.stage4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                        nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                        nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                        nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

            self.stage5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                        nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                        nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                        nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

            self.net = nn.Sequential(self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, nn.Flatten(),
                                     nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1024, 1024),
                                     nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1024, num_class))

        def forward(self, src, tar):
            y_src = self.net(src)
            x_src_mmd = self.stage1(src)
            x_tar_mmd = self.stage1(tar)
            x_src_mmd = self.stage2(x_src_mmd)
            x_tar_mmd = self.stage2(x_tar_mmd)
            x_src_mmd = self.stage2(x_src_mmd)
            x_tar_mmd = self.stage2(x_tar_mmd)

            return y_src, x_src_mmd, x_tar_mmd


    # calculate mmd loss
    def mix_rbf_mmd2(X, Y, sigma_list=[100], biased=True):
        K_XX, K_XY, K_YY, d = mix_rbf_kernel(X, Y, sigma_list)
        return mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


    def mix_rbf_kernel(X, Y, sigma_list):
        assert (X.size(0) == Y.size(0))
        m = X.size(0)

        Z = torch.cat((X, Y), 0)
        ZZT = torch.mm(Z, Z.t())
        diag_ZZT = torch.diag(ZZT).unsqueeze(1)
        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma ** 2)
            K += torch.exp(-gamma * exponent)

        return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


    def mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        m = K_XX.size(0)
        if const_diagonal is not False:
            diag_X = diag_Y = const_diagonal
            sum_diag_X = sum_diag_Y = m * const_diagonal
        else:
            diag_X = torch.diag(K_XX)
            diag_Y = torch.diag(K_YY)
            sum_diag_X = torch.sum(diag_X)
            sum_diag_Y = torch.sum(diag_Y)

        Kt_XX_sums = K_XX.sum(dim=1) - diag_X
        Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
        K_XY_sums_0 = K_XY.sum(dim=0)

        Kt_XX_sum = Kt_XX_sums.sum()
        Kt_YY_sum = Kt_YY_sums.sum()
        K_XY_sum = K_XY_sums_0.sum()

        if biased:
            mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                    + (Kt_YY_sum + sum_diag_Y) / (m * m)
                    - 2.0 * K_XY_sum / (m * m))
        else:
            mmd2 = (Kt_XX_sum / (m * (m - 1))
                    + Kt_YY_sum / (m * (m - 1))
                    - 2.0 * K_XY_sum / (m * m))

        return mmd2


    # train
    argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
    astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
    reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
    size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


    class Animator:
        def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                     ylim=None, xscale='linear', yscale='linear',
                     fmts=('y-', 'r-', 'g-.', 'b-'), nrows=1, ncols=1,
                     figsize=(6, 4)):
            if legend is None:
                legend = []
            display.set_matplotlib_formats('svg')
            self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
            if nrows * ncols == 1:
                self.axes = [self.axes, ]
            self.config_axes = lambda: set_axes(
                self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
            self.X, self.Y, self.fmts = None, None, fmts

        def add(self, x, y):
            if not hasattr(y, "__len__"):
                y = [y]
            n = len(y)
            if not hasattr(x, "__len__"):
                x = [x] * n
            if not self.X:
                self.X = [[] for _ in range(n)]
            if not self.Y:
                self.Y = [[] for _ in range(n)]
            for i, (a, b) in enumerate(zip(x, y)):
                if a is not None and b is not None:
                    self.X[i].append(a)
                    self.Y[i].append(b)
            self.axes[0].cla()
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                self.axes[0].plot(x, y, fmt)
            self.config_axes()
            display.clear_output(wait=True)


    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()


    def try_gpu(i=0):
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')


    def train_plot(dann, train_iter, test_list, num_epochs, opt, eta, device=try_gpu()):
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform_(m.weight)

        dann.net.apply(init_weights)

        print('training on', device)
        dann.net.to(device)
        optimizer = opt
        loss = nn.CrossEntropyLoss()
        test_batch_id = 0
        animator = Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'mmd loss', 'total loss'])

        best = 100
        for epoch in range(num_epochs):
            for i, (x_train, y) in enumerate(train_iter):
                dann.net.train()

                _, (x_test, _) = test_list[test_batch_id]
                x_train, x_test, y = x_train.to(device), x_test.to(device), y.to(device)

                y_src, x_src_mmd, x_tar_mmd = dann(x_train, x_test)
                train_loss = loss(y_src, y)

                mmd_loss = 0
                x_src_mmd = x_src_mmd.view(-1, 8 * 8)
                x_tar_mmd = x_tar_mmd.view(-1, 8 * 8)
                mmd_loss = eta * mix_rbf_mmd2(x_src_mmd, x_tar_mmd)

                total_loss = train_loss + mmd_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if total_loss < best:
                    best = total_loss
                    torch.save(dann.net.state_dict(), "Net.param")

                animator.add(epoch + i / len(train_iter), (train_loss, mmd_loss, total_loss))
                test_batch_id = (test_batch_id + 1) % len(test_list)

            print("Epoch: {}/{}, train loss: {}, mmd loss: {}, total loss: {}".format(epoch + 1, num_epochs, train_loss,
                                                                                      mmd_loss, total_loss))

        plt.savefig("Result.jpg")
        plt.show()
        torch.save(vgg.state_dict(), "Network.param")


    vgg = VGG(3, 9)
    train_plot(vgg, train_iter, test_list, epochs, torch.optim.Adam(vgg.net.parameters(), lr=lr), eta)

    # predict
    def predict():
        vgg.net.eval()
        test = pd.read_csv(csv_path)
        result = []
        for _, (x_test, _) in test_list:
            x_test = x_test.to(try_gpu())
            y_pred = vgg.net(x_test)
            result.extend(argmax(y_pred, axis=1).cpu().numpy().tolist())

        index = [i for i in range(0, len(test))]
        output = pd.DataFrame({'ID': index, 'label': result})
        output.to_csv('Answer.csv', index=None, encoding='utf8')

    predict()
