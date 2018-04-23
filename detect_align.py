import cv2
import dlib
import os
from PIL import Image
from MTCNN_pytorch.src import detect_faces

detector = dlib.get_frontal_face_detector()
# detector = dlib.cnn_face_detection_model_v1('model_file/mmod_human_face_detector.dat')
landmark_predictor = dlib.shape_predictor('model_file/dlib/shape_predictor_5_face_landmarks.dat')

def detectFace(img, cropSize=112):

    dets = detector(img, 1)

    if (len(dets) <= 0):
        raise ValueError("No face in img")

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(landmark_predictor(img, detection))

    images = dlib.get_face_chips(img, faces, size=cropSize)

    # cv2.imshow('image', images[0])
    # cv2.waitKey(0)

    return images[0]

# MTCNN检测face pytorch版本
def detect_MTCNN_pytorch(img, cropSize=112):

    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # opencv格式转PIL Image格式
    boundingBoxes, landmarks = detect_faces(image)

    # 按照置信度排序
    boundingBoxes = sorted(boundingBoxes, key=lambda x: x[-1], reverse=True)
    faceBoundingBox = boundingBoxes[0]

    # 过滤置信度低的人脸
    print(faceBoundingBox[-1])
    if faceBoundingBox[-1] < 0.98:
        raise ValueError("confidence is too low")

    x1 = int(faceBoundingBox[0])
    y1 = int(faceBoundingBox[1])
    x2 = int(faceBoundingBox[2])
    y2 = int(faceBoundingBox[3])
    #
    # for i in range(5):                       # 整理landmark为指定形式list
    #     landMarkList.append(landmarks[0][i])
    #     landMarkList.append(landmarks[0][i+5])

    faceImg = img[y1:y2, x1:x2]

    faceImg = cv2.resize(faceImg, (cropSize, cropSize)) # crop至指定尺寸

    # cv2.imshow('image', faceImg)
    # cv2.waitKey(0)

    return faceImg


def detectAllWebface(rootPath, saveRoot):

    failcnt = 0

    for root, dirs, files in os.walk(rootPath):
        for dir in dirs:
            for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir)):
                for file in subfiles:
                    imgPath = os.path.join(rootPath, dir, file)
                    print(imgPath)
                    img = cv2.imread(imgPath)
                    try:
                        faceImg = detectFace(img, cropSize=112) # 先用dilib检测，若失败，换用MTCNN，全部失败则跳过
                        print("dlib dtected")
                    except:
                        try:
                            faceImg = detect_MTCNN_pytorch(img, cropSize=112)
                            print("mtcnn dtected")
                        except:
                            failcnt += 1
                            print("detect fail:", failcnt)
                            continue

                    if dir not in os.listdir(saveRoot):
                        os.mkdir(os.path.join(saveRoot, dir))
                    saveImgPath = os.path.join(saveRoot, dir, file)
                    cv2.imwrite(saveImgPath, faceImg)


if __name__ == '__main__':

    webfaceRoot = "data/CASIA-WebFace/"
    saveRoot = "data/webface_detect_5_points/"

    detectAllWebface(webfaceRoot, saveRoot)