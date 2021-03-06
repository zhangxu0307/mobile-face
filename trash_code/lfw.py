import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch as th
from detect_align import detectFace
from mobile_net import MobileNetV2
from resnet import ResNet34, ResNet18, ResNet50
from load_data import *
import time
from net_sphere import sphere20a, AngleLoss
from mobilenetv2_modify import MobileNetV2_modify
import torch
from AM_loss import AMLoss
import os
from matplotlib.pyplot import plot, savefig
from glob import glob
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# 计算成对的余弦相似度
def calcCosSimilarityPairs(rep1, rep2):
    return np.dot(rep1, rep2.T) / (np.linalg.norm(rep1, 2) * np.linalg.norm(rep2, 2))


def example(net, imgSize=(96, 96)):

    imgPath1 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg"
    imgPath2 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0003.jpg"
    imgPath3 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/Abdel_Nasser_Assidi/Abdel_Nasser_Assidi_0001.jpg"
    imgPath4 = "data/LFW/lfw-deepfunneled/lfw-deepfunneled/Abdel_Nasser_Assidi/Abdel_Nasser_Assidi_0002.jpg"

    img1 = cv2.imread(imgPath1)
    img2 = cv2.imread(imgPath2)
    img3 = cv2.imread(imgPath3)
    img4 = cv2.imread(imgPath4)

    face1 = detectFace(img1, imgSize)
    face2 = detectFace(img2, imgSize)
    face3 = detectFace(img3, imgSize)
    face4 = detectFace(img4, imgSize)

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
def getNegPairsImg(pairsTxT, filenameTxt, v):

    negPairsTxt = pd.read_csv(pairsTxT, sep="   ", header=0)
    imgPath = pd.read_csv(filenameTxt, header=-1)
    negPairsNum = len(negPairsTxt)
    print("neg pairs num:", negPairsNum)

    for i in range(negPairsNum):

        index1 = negPairsTxt.ix[i, "s1"]
        index2 = negPairsTxt.ix[i, "s2"]
        path1 = imgPath.ix[index1 - 1, 0]
        path2 = imgPath.ix[index2 - 1, 0]

        path1 = os.path.join(rootPath, path1)
        path2 = os.path.join(rootPath, path2)
        print(path1)
        print(path2)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        yield img1, img2

# 获取正样本图像pair
def getPosPairsImg(pairsTxT, filenameTxt, rootPath):

    posPairsTxt = pd.read_csv(pairsTxT, sep="   ", header=0)
    imgPath = pd.read_csv(filenameTxt, header=-1)
    posPairsNum = len(posPairsTxt)
    print("pos pairs num:", posPairsNum)

    for i in range(posPairsNum):
        index1 = posPairsTxt.ix[i, "s1"]
        index2 = posPairsTxt.ix[i, "s2"]
        path1 = imgPath.ix[index1 - 1, 0]
        path2 = imgPath.ix[index2 - 1, 0]

        path1 = os.path.join(rootPath, path1)
        path2 = os.path.join(rootPath, path2)
        print(path1)
        print(path2)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        yield img1, img2

# 测试LFW数据集相似度分布情况
def runLFWPosPairs(net, modelName, imgSize, posPairsTxT, filenameTxT, rootPath):

    posScore = []

    posGen = getPosPairsImg(posPairsTxT, filenameTxT, rootPath)
    for img1, img2 in posGen:
        try:
            face1 = detectFace(img1, cropSize=imgSize)
            face2 = detectFace(img2, cropSize=imgSize)

            rep1 = net.getRep(face1)
            rep2 = net.getRep(face2)

            score = calcCosSimilarityPairs(rep1, rep2)[0][0]
        except:
            continue
        print(score)
        # if score < 0.6:
        #     cv2.imshow("face1:", face1)
        #     cv2.imshow("face2:", face2)
        #     cv2.waitKey()
        posScore.append(score)

    posCsv = pd.DataFrame(posScore)
    posCsv.to_csv("data/pos_score_"+modelName+".csv", index=False)

# 测试LFW数据集相似度分布情况
def runLFWNegPairs(net, modelName, imgSize, negPairsTxT, filenameTxT, rootPath):

    negScore = []

    negGen = getNegPairsImg(negPairsTxT, filenameTxT, rootPath)
    for img1, img2 in negGen:
        try:
            face1 = detectFace(img1, cropSize=imgSize)
            face2 = detectFace(img2, cropSize=imgSize)

            rep1 = net.getRep(face1)
            rep2 = net.getRep(face2)

            score = calcCosSimilarityPairs(rep1, rep2)[0][0]
        except:
            continue
        print(score)
        # if score > 0.6:
        #     cv2.imshow("face1:", face1)
        #     cv2.imshow("face2:", face2)
        #     cv2.waitKey()
        negScore.append(score)

    negCsv = pd.DataFrame(negScore)
    negCsv.to_csv("data/neg_score_"+modelName+".csv", index=False)

# 绘制相似度分布直方图
def plotSimliarityHist(modelName): # 此处仍有bug，两个直方图会有混叠现象，只能一个个绘制

    # 绘制负样本对得分
    filePath = "data/neg_score_"+modelName+".csv"
    data = pd.read_csv(filePath)
    hist = data["0"].hist()
    fig1 = hist.get_figure()
    fig1.savefig('data/neg_score_' + modelName + ".jpg")

    # 绘制正样本对得分
    filePath = "data/pos_score_" + modelName + ".csv"
    data = pd.read_csv(filePath)
    hist = data["0"].hist()
    fig2 = hist.get_figure()
    fig2.savefig('data/pos_score_' + modelName + ".jpg")


# # 使用余弦相似度卡阈值计算准确率
def LFWAccScore(modelName, threshold):

    negCSV = os.path.join("data", 'neg_score_' + modelName + ".csv")
    posCSV = os.path.join("data", 'pos_score_' + modelName + ".csv")

    negScore = pd.read_csv(negCSV)
    posScore = pd.read_csv(posCSV)

    negScore = negScore["0"] <= threshold
    posScore = posScore["0"] > threshold

    acc = (negScore.sum()+posScore.sum())/(len(negScore)+len(posScore))

    print("lfw acc:", acc)


if __name__ == '__main__':

    posPairsTxT = "data/LFW/postive_pairs.txt"
    negPairsTxT = "data/LFW/negative_pairs.txt"
    filenameTxT = "data/LFW/Path_lfw2.txt"
    rootPath = "data/LFW/lfw-deepfunneled/lfw-deepfunneled"

    modelPath = "model_file/sphere20a_20171020.pth"
    modelName = modelPath.replace('.', '/').split('/')[1]
    print("model:", modelName)

    net = sphere20a(10574)
    net.load_state_dict(th.load(modelPath))
    net = net.cuda()
    net = net.eval()

    example(net, imgSize=(112, 96))
    runLFWPosPairs(net, modelName, imgSize=(112, 96), posPairsTxT=posPairsTxT, filenameTxT=filenameTxT, rootPath=rootPath)
    runLFWNegPairs(net, modelName, imgSize=(112, 96), negPairsTxT=negPairsTxT, filenameTxT=filenameTxT, rootPath=rootPath)
    plotSimliarityHist(modelName)

    threshold = 0.58
    LFWAccScore(modelName, threshold)