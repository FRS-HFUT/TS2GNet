import os
import numpy as np
import random
import torch.nn.functional as F
import torch
import torch.utils.data as dataf
import torch.nn as nn
from scipy import io
import HyperX
import losses
import metric
import scipy.io as sio
from torch.optim import lr_scheduler
import datetime
from TS2GNet import TS2GNet
import rasterio
from torch.optim.lr_scheduler import StepLR
from osgeo import gdal
import matplotlib.pyplot as plt
from PIL import Image

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(14)
code_start_time = datetime.datetime.now()  # 开始时间点
print("程序运行开始时间：", code_start_time)

def DrawResult(height, width, num_class, labels):
    palette = np.array([
        [37, 58, 150],
        [51, 181, 232],
        [112, 204, 216],
        [148, 204, 120],
        [188, 215, 78],
        [238, 234, 63],
        [244, 127, 33],
        [239, 71, 34],
        [238, 33, 35],
        [123, 18, 20]
    ])
    palette = palette[:num_class]
    palette = palette * 1.0 / 255
    X_result = np.zeros((labels.shape[0], 3))

    for i in range(1, num_class + 1):
        mask = (labels == i)
        X_result[mask, 0] = palette[i - 1, 0]  # 红色通道
        X_result[mask, 1] = palette[i - 1, 1]  # 绿色通道
        X_result[mask, 2] = palette[i - 1, 2]  # 蓝色通道

    X_result = np.reshape(X_result, (height, width, 3))
    return X_result  #


def padWithZeros(img, pad_width):
    return np.pad(img, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), 'constant')
# load data
DataPath = r'G:\NET\TS2GNet\stacking\stacking_10m.mat'
TRPath = r'G:\NET\TS2GNet\stacking\label_train_all.mat'
TSPath = r'G:\NET\TS2GNet\stacking\label_test_all.mat'

patchsize = 16
batchsize = 16  
EPOCH = 500
LR = 1e-4
Data = io.loadmat(DataPath)
TrLabel = io.loadmat(TRPath)
TsLabel = io.loadmat(TSPath)
Data = Data['image']  #

Data = Data.astype(np.float32)
TrLabel = TrLabel['value']
TsLabel = TsLabel['value']

labels_max = TrLabel.max()
labels_min = TrLabel.min()
print(f"Max label: {labels_max}, Min label: {labels_min}")
print('image shape: ', Data.shape)
print('Label shape: ', TrLabel.shape)

pad_width = np.floor(patchsize / 2)
pad_width = int(pad_width)
[m, n, l] = np.shape(Data)  #
class_number = np.max(TsLabel)

for i in range(l):
    Data[:, :, i] = (Data[:, :, i] - Data[:, :, i].min()) / (Data[:, :, i].max() - Data[:, :, i].min())
x = Data
x2 = np.empty((m + 2 * pad_width, n + 2 * pad_width, l), dtype='float32')

for i in range(l):
    temp = x[:, :, i]
    temp2 = np.pad(temp, pad_width, 'symmetric')
    x2[:, :, i] = temp2  #


[ind1, ind2] = np.where(TsLabel != 0)

TestNum = len(ind1)
TestPatch = np.empty((TestNum, l, patchsize + 1, patchsize + 1), dtype='float32')  # (42596,103,25,25)
TestLabel = np.empty(TestNum)
for i in range(len(ind1)):
    patch = x2[(ind1[i]):(ind1[i] + patchsize + 1), (ind2[i]):(ind2[i] + patchsize + 1), :]
    patch = np.transpose(patch, (2, 0, 1))
    TestPatch[i, :, :, :] = patch
    patchlabel = TsLabel[ind1[i], ind2[i]]
    TestLabel[i] = patchlabel
train_dataset = HyperX.dataLoad(x2, TrLabel, patch_size=patchsize, center_pixel=False, flip_augmentation=True,
                                mixture_augmentation=False)
train_loader = dataf.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
TestPatch = torch.from_numpy(TestPatch)
TestLabel = torch.from_numpy(TestLabel) - 1
TestLabel = TestLabel.long()
Classes = len(np.unique(TrLabel)) - 1
cnn = TS2GNet(classes=class_number, HSI_Data_Shape_H=m, HSI_Data_Shape_W=n, HSI_Data_Shape_C=l,
                              patch_size=patchsize + 1)
cnn.cuda()

total = sum([param.nelement() for param in cnn.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=7e-6)
# scheduler = StepLR(optimizer, step_size=300, gamma=0.2)# '''固定步长学习率优化'''
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200,350,450], gamma=0.5)  # '''指定步长学习率优化'''
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)# '''余弦退火学习率优化'''# optimize all cnn parameters
loss_fun1 = losses.ConstractiveLoss()
loss_fun2 = nn.CrossEntropyLoss(label_smoothing=0.05)
show_feature_map = []

save_dir = r'G:\NET\TS2GNet\pkl'
os.makedirs(save_dir, exist_ok=True)
BestAcc = 0.0  #
# #

start_training_time = datetime.datetime.now()
for epoch in range(EPOCH):
    for step, (images, points, labels) in enumerate(train_loader):
        images = images.cuda()
        points = points.cuda()
        labels = labels.cuda()

        features3, output = cnn(images, points)
        classifier_loss = loss_fun2(output, labels)
        total_loss = classifier_loss

        cnn.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            cnn.eval()
            pred_y = np.empty((len(TestLabel)), dtype='float32')
            number = len(TestLabel) // 2000
            for i in range(number):
                temp = TestPatch[i * 2000:(i + 1) * 2000, :, :, :]
                temp_points = temp[:, :, pad_width, pad_width]
                temp = temp.cuda()
                temp_points = temp_points.cuda()
                _, temp2 = cnn(temp, temp_points)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 2000:(i + 1) * 2000] = temp3.cpu()
                del temp, temp2, temp3, _, temp_points

            if (i + 1) * 2000 < len(TestLabel):
                temp = TestPatch[(i + 1) * 2000:len(TestLabel), :, :, :]
                temp_points = temp[:, :, pad_width, pad_width]
                temp_points = temp_points.cuda()
                temp = temp.cuda()
                _, temp2 = cnn(temp, temp_points)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 2000:len(TestLabel)] = temp3.cpu()
                del temp, temp2, temp3, _, temp_points

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
            current_lr = optimizer.param_groups[0]['lr']
            print('Epoch: ', epoch, '| classify loss: %.6f' % classifier_loss.data.cpu().numpy(),
                  '| test accuracy（OA）: %.6f' % accuracy,
                  '| learning rate: %.6f' % current_lr,
                  )

            if accuracy > BestAcc:
                filename = f'net_params_myNet_UP_OA{accuracy * 100:.2f}.pkl'
                filepath = os.path.join(save_dir, filename)
                torch.save(cnn.state_dict(), filepath)
                BestAcc = accuracy
            cnn.train()
    scheduler.step()

end_training_time = datetime.datetime.now()
def Draw(label_image, dataset_name, model_name, num_classes):
    height, width = label_image.shape
    img = DrawResult(height, width, num_classes, label_image.flatten())

    img_uint8 = (img * 255).astype(np.uint8)
    image_pil = Image.fromarray(img_uint8)
    png_path = f"{dataset_name}_{model_name}.png"
    tif_path = f"{dataset_name}_{model_name}.tif"
    image_pil.save(png_path, format='PNG', dpi=(300, 300))
    image_pil.save(tif_path, format='TIFF', dpi=(300, 300))
    print(f"Saved classification results as {png_path} and {tif_path}")
cnn.load_state_dict(torch.load(r'G:\NET\TS2GNet\best.pkl'))
cnn.eval()

pred_y = np.empty((len(TestLabel)), dtype='float32')
number = len(TestLabel) // 5000
for i in range(number):
    temp = TestPatch[i * 5000:(i + 1) * 5000, :, :, :]
    temp_points = temp[:, :, pad_width, pad_width]
    temp = temp.cuda()
    temp_points = temp_points.cuda()
    _, temp2 = cnn(temp, temp_points)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i * 5000:(i + 1) * 5000] = temp3.cpu()
    del temp, temp2, temp3, _, temp_points

if (i + 1) * 5000 < len(TestLabel):
    temp = TestPatch[(i + 1) * 5000:len(TestLabel), :, :, :]
    temp_points = temp[:, :, pad_width, pad_width]
    temp = temp.cuda()
    temp_points = temp_points.cuda()
    _, temp2 = cnn(temp, temp_points)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i + 1) * 5000:len(TestLabel)] = temp3.cpu()
    del temp, temp2, temp3, _, temp_points
pred_y = torch.from_numpy(pred_y).long()
Classes = np.unique(TestLabel)
precision = np.empty(len(Classes))
recall = np.empty(len(Classes))
f1_score = np.empty(len(Classes))


for i in range(len(Classes)):
    cla = Classes[i]
    TP = 0
    FP = 0
    FN = 0

    for j in range(len(TestLabel)):
        if TestLabel[j] == cla and pred_y[j] == cla:
            TP += 1
        if TestLabel[j] != cla and pred_y[j] == cla:
            FP += 1
        if TestLabel[j] == cla and pred_y[j] != cla:
            FN += 1

    precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0


print('-------------------')
for i in range(len(Classes)):
    print('|第%d类：' % (i + 1))
    print('生产者精度（Recall）： %.2f%%' % (recall[i] * 100))
    print('用户精度（Precision）： %.2f%%' % (precision[i] * 100))
    print('F1 分数： %.4f' % f1_score[i])
    print('-------------------')

results = metric.metrics(pred_y, TestLabel, n_classes=len(Classes))
print('%.2f' % results["Accuracy"], "The OA")
print('%.2f' % results["Kappa"], "The Kappa")

label_image = np.full((m, n), -1, dtype=np.int32)
label_image[ind1, ind2] = pred_y.numpy() + 1

code_end_time = datetime.datetime.now()
print("程序运行结束时间：", code_end_time)
print('程序运行总时长：', code_end_time - code_start_time)  # 运行时间，单位是  时:分:秒

