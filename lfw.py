import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch as th
from mobile_net import *
from detect_align import *

import os
from matplotlib.pyplot import plot, savefig
from glob import glob
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# 计算成对的余弦相似度
def calcCosSimilarityPairs(rep1, rep2):
    return np.dot(rep1, rep2.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep2, 2))

def example(modelPath):

    net = th.load(modelPath)

    imgPath1 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg"
    imgPath2 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0003.jpg"
    imgPath3 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/Abdel_Nasser_Assidi/Abdel_Nasser_Assidi_0001.jpg"
    imgPath4 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/Abdel_Nasser_Assidi/Abdel_Nasser_Assidi_0002.jpg"

    img1 = cv2.imread(imgPath1)
    img2 = cv2.imread(imgPath2)
    img3 = cv2.imread(imgPath3)
    img4 = cv2.imread(imgPath4)

    face1 = detectFace(img1, 96)
    face2 = detectFace(img2, 96)
    face3 = detectFace(img3, 96)
    face4 = detectFace(img4, 96)

    rep1 = net.getRep(face1)
    rep2 = net.getRep(face2)
    rep3 = net.getRep(face3)
    rep4 = net.getRep(face4)

    # 余弦相似度
    print(calcCosSimilarityPairs(rep1, rep2))
    print(calcCosSimilarityPairs(rep3, rep4))
    print(calcCosSimilarityPairs(rep1, rep3))
    print(calcCosSimilarityPairs(rep4, rep2))

# 获取负样本图像pair
def getNegPairsImg():

    negPairsTxt = pd.read_csv("data/LFW/negative_pairs.txt", sep="   ", header=0)
    imgPath = pd.read_csv("data/LFW/Path_lfw2.txt", header=-1)
    negPairsNum = len(negPairsTxt)
    print("neg pairs num:", negPairsNum)

    for i in range(negPairsNum):

        index1 = negPairsTxt.ix[i, "s1"]
        index2 = negPairsTxt.ix[i, "s2"]
        path1 = imgPath.ix[index1 - 1, 0]
        path2 = imgPath.ix[index2 - 1, 0]

        path1 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/" + path1
        path2 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/" + path2
        print(path1)
        print(path2)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        yield img1, img2

# 获取正样本图像pair
def getPosPairsImg():

    posPairsTxt = pd.read_csv("data/LFW/postive_pairs.txt", sep="   ", header=0)
    imgPath = pd.read_csv("data/LFW/Path_lfw2.txt", header=-1)
    posPairsNum = len(posPairsTxt)
    print("pos pairs num:", posPairsNum)

    for i in range(posPairsNum):
        index1 = posPairsTxt.ix[i, "s1"]
        index2 = posPairsTxt.ix[i, "s2"]
        path1 = imgPath.ix[index1 - 1, 0]
        path2 = imgPath.ix[index2 - 1, 0]

        path1 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/" + path1
        path2 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/" + path2
        print(path1)
        print(path2)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        yield img1, img2

# 测试LFW数据集相似度分布情况
def runLFW(modelPath, modelName):

    acc = 0
    posScore = []
    negScore = []

    net = th.load(modelPath)

    posGen = getPosPairsImg()
    for img1, img2 in posGen:
        try:
            face1 = detectFace(img1, cropSize=96)
            face2 = detectFace(img2, cropSize=96)
            rep1 = net.getRep(face1)
            rep2 = net.getRep(face2)
            score = calcCosSimilarityPairs(rep1, rep2)[0][0]
        except:
            continue
        print(score)
        posScore.append(score)

    posCsv = pd.DataFrame(posScore)
    posCsv.to_csv("data/pos_score_"+modelName+".csv", index=False)

    negGen = getNegPairsImg()
    for img1, img2 in negGen:
        try:
            face1 = detectFace(img1, cropSize=96)
            face2 = detectFace(img2, cropSize=96)
            rep1 = net.getRep(face1)
            rep2 = net.getRep(face2)
            score = calcCosSimilarityPairs(rep1, rep2)[0][0]
        except:
            continue
        print(score)
        negScore.append(score)

    negCsv = pd.DataFrame(negScore)
    negCsv.to_csv("data/neg_score_"+modelName+".csv", index=False)

# 绘制相似度分布直方图
def plotSimliarityHist(modelName): # 此处仍有bug，两个直方图会有混叠现象，只能一个个绘制

    # 绘制负样本对得分
    filePath = "data/neg_score_"+modelName+".csv"
    data = pd.read_csv(filePath)
    print(data)
    hist = data["0"].hist()
    fig1 = hist.get_figure()
    fig1.savefig('data/neg_score_' + modelName + ".jpg")

    # 绘制正样本对得分
    filePath = "data/pos_score_" + modelName + ".csv"
    data = pd.read_csv(filePath)
    print(data)
    hist = data["0"].hist()
    fig2 = hist.get_figure()
    fig2.savefig('data/pos_score_' + modelName + ".jpg")


# # 使用余弦相似度卡阈值计算准确率
# def runLFWScore(modelName, threshold):
#
#     acc = 0
#
#     negFile = open('data/lfw_neg_' + modelName + '.pkl', 'rb')
#     negPairs = pickle.load(negFile)
#
#     for pair in negPairs:
#         x1 = pair[0]
#         x2 = pair[1]
#         score = calcCosSimilarityPairs(x1, x2)
#         if score < threshold:
#             acc += 1
#
#     posFile = open('data/lfw_pos_' + modelName + '.pkl', 'rb')
#     posPairs = pickle.load(posFile)
#
#     for pair in posPairs:
#         x1 = pair[0]
#         x2 = pair[1]
#         score = calcCosSimilarityPairs(x1, x2)
#         if score > threshold:
#             acc += 1
#     print("lfw cos classify acc:", acc/(len(posPairs)+len(negPairs)))


if __name__ == '__main__':

    modelPath = "model_file/mobilefacev2_webface_align.pt"
    modelName = "resnet34"

    example(modelPath)

    runLFW(modelPath, modelName)
    plotSimliarityHist(modelName)

    # threshold = 0.5
    # runLFWScore(modelName, threshold)