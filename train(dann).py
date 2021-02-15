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
    lr, epochs, batch_size, eta = 0.001, 200, 500, 0.1
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    csv_path = sys.argv[3]

    trans1 = transforms.Compose([transforms.Grayscale(), transforms.Lambda(lambda x: cv2.Canny(np.array(x), 150, 300)),
                                 transforms.ToPILImage(), transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15, fill=(0,)),
                                 transforms.ToTensor()])
    train_imgs = datasets.ImageFolder(train_path, transform=trans1)
    train_iter = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)

    trans2 = transforms.Compose([transforms.Grayscale(), transforms.Resize((32, 32)), transforms.ToTensor()])
    test_imgs = datasets.ImageFolder(test_path, transform=trans2)
    test_iter = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size, shuffle=False)

    # construct net
    class extractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
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

            self.conv = nn.Sequential(self.stage1, self.stage2, self.stage3, self.stage4, self.stage5)

        def forward(self, x):
            return self.conv(x).squeeze()


    class classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
                                        nn.Linear(1024, 9))

        def forward(self, src):
            return self.linear(src)


    class discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                        nn.Linear(512, 1))

        def forward(self, x):
            return self.linear(x)

    # train
    def try_gpu(i=0):
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')


    def train_plot(num_epochs, device=try_gpu()):
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform_(m.weight)

        fea.apply(init_weights)
        cla.apply(init_weights)
        dis.apply(init_weights)
        print('training on', device)
        fea.to(device)
        cla.to(device)
        dis.to(device)

        func1 = nn.CrossEntropyLoss()
        func2 = nn.BCEWithLogitsLoss()
        trainer_fea = torch.optim.Adam(fea.parameters())
        trainer_cla = torch.optim.Adam(cla.parameters())
        trainer_dis = torch.optim.Adam(dis.parameters())

        for epoch in range(num_epochs):
            loss_cla = 0
            loss_dis = 0
            for i, ((x_train, y), (x_test, _)) in enumerate(zip(train_iter, test_iter)):
                cla.train()
                dis.train()
                fea.train()
                x_train, x_test, y = x_train.to(device), x_test.to(device), y.to(device)

                mixed_data = torch.cat([x_train, x_test], dim=0)
                domain_label = torch.zeros([x_train.shape[0] + x_test.shape[0], 1]).cuda()
                domain_label[:x_train.shape[0]] = 1
                feature = fea(mixed_data)
                domain_logits = dis(feature.detach())
                loss_dis = func2(domain_logits, domain_label)
                loss_dis.backward()
                trainer_dis.step()

                class_logits = cla(feature[:x_train.shape[0]])
                domain_logits = dis(feature)
                loss_cla = func1(class_logits, y) - eta * func2(domain_logits, domain_label)
                loss_cla.backward()
                trainer_fea.step()
                trainer_cla.step()

                trainer_cla.zero_grad()
                trainer_fea.zero_grad()
                trainer_dis.zero_grad()

            losses_cla.append(loss_cla)
            losses_dis.append(loss_dis)

            print("Epoch: {}/{}, classifier loss: {}, discriminator loss: {}".format(epoch + 1, num_epochs, loss_cla,
                                                                                     loss_dis))

        plt.plot(losses_cla, label="Classifier loss")
        plt.plot(losses_dis, label="Discriminator loss")
        plt.legend(loc='best')
        plt.savefig("Result.jpg")
        plt.show()
        torch.save(fea.state_dict(), "Feature extractor.param")
        torch.save(cla.state_dict(), "Label predictor.param")
        torch.save(dis.state_dict(), "Domain classifier.param")


    losses_cla = []
    losses_dis = []
    fea = extractor()
    cla = classifier()
    dis = discriminator()
    train_plot(epochs)


    def feature_plot():
        for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(train_iter, test_iter)):
            source_data = source_data.to(try_gpu())
            target_data = target_data.to(try_gpu())
            res1 = fea(source_data).detach().cpu()
            res2 = fea(target_data).detach().cpu()
            if i == 0:
                x1 = res1
                x2 = res2
            elif i > 4:
                break
            else:
                x1 = torch.cat((x1, res1))
                x2 = torch.cat((x2, res2))

        X = torch.cat((x1, x2))
        out = TSNE(n_components=2).fit_transform(X)
        p1 = out.T[0]
        p2 = out.T[1]
        plt.figure(figsize=(10, 10))
        plt.scatter(p1[:2000], p2[:2000], label="source")
        plt.scatter(p1[2000:], p2[2000:], label="target")
        plt.legend(loc='best')
        plt.show()


    feature_plot()

    # predict
    argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)

    def predict():
        cla.eval()
        fea.eval()
        test = pd.read_csv(csv_path)
        result = []
        for _, (x_test, _) in enumerate(test_iter):
            x_test = x_test.to(try_gpu())
            y_pred = cla(fea(x_test))
            result.extend(argmax(y_pred, axis=1).cpu().numpy().tolist())

        index = [i for i in range(0, len(test))]
        output = pd.DataFrame({'ID': index, 'label': result})
        output.to_csv('Answer.csv', index=None, encoding='utf8')


    predict()