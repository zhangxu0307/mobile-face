
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile
import pandas as pd
import matplotlib.pyplot as plt

from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
from mobile_net import MobileNetV2
from resnet import ResNet34, ResNet18, ResNet50

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def alignment(src_img,src_pts):

    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i*n//n_folds:(i+1)*n//n_folds]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

#
def runNetonLFW(lfwPath, lfwLandmarkPath, lfwPairsPath):

    zfile = zipfile.ZipFile(lfwPath)
    predicts = []
    landmark = {}

    with open(lfwLandmarkPath) as f:
        landmark_lines = f.readlines()
    for line in landmark_lines:
        l = line.replace('\n', '').split('\t')
        landmark[l[0]] = [int(k) for k in l[1:]]

    with open(lfwPairsPath) as f:
        pairs_lines = f.readlines()[1:]

    for i in range(6000):
        p = pairs_lines[i].replace('\n', '').split('\t')

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))

        img1 = alignment(cv2.imdecode(np.frombuffer(zfile.read("lfw/" + name1), np.uint8), 1), landmark[name1])
        img2 = alignment(cv2.imdecode(np.frombuffer(zfile.read("lfw/" + name2), np.uint8), 1), landmark[name2])

        imglist = [img1, cv2.flip(img1, 1), img2, cv2.flip(img2, 1)]
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1, 3, 112, 96))
            imglist[i] = (imglist[i] - 127.5) / 128

        img = np.vstack(imglist)
        img = Variable(torch.from_numpy(img).float(), volatile=True).cuda()
        output = net(img)
        f = output.data
        f1, f2 = f[0], f[2]
        cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
        predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, cosdistance, sameflag))

    predicts = np.array(list(map(lambda line: line.strip('\n').split('\t'), predicts)))

    return predicts


# 绘制相似度分布直方图
def plotSimliarityHist(predicts, savePath):

    predDf = pd.DataFrame(predicts)
    predDf.columns = ["name1", "name2", "cosdistance", "sameflag"]

    pos = predDf[predDf["sameflag"] == '1']
    neg = predDf[predDf["sameflag"] == '0']

    hist = pd.to_numeric(pos["cosdistance"]).hist()
    fig1 = hist.get_figure()
    fig1.savefig(os.path.join(savePath, "pos.jpg"))

    hist = pd.to_numeric(neg["cosdistance"]).hist()
    fig2 = hist.get_figure()
    fig2.savefig(os.path.join(savePath, "neg.jpg"))


if __name__ == '__main__':

    lfwPath = 'data/LFW/lfw.zip'
    modelPath = 'model_file/resnet34_webface_align_m05.pt'
    lfwLandmarkPath = 'data/LFW/lfw_landmark.txt'
    lfwPairsPath = 'data/LFW/pairs.txt'
    histSavePath = "data/"

    modelName = modelPath.replace('.', '/').split('/')[1]
    print("model:", modelName)

    net = ResNet34(feature=True)
    net.load_state_dict(torch.load(modelPath))
    net.cuda()
    net.eval()
    net.feature = True
    print("load model finished")

    predicts = runNetonLFW(lfwPath, lfwLandmarkPath, lfwPairsPath)
    print("predict finished!")

    plotSimliarityHist(predicts, savePath=histSavePath)

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)

    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
