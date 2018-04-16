import cv2
import dlib
import os

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('model_file/shape_predictor_68_face_landmarks.dat')
img = cv2.imread('data/LFW/lfw-deepfunneled/lfw-deepfunneled/Abdul_Rahman/Abdul_Rahman_0001.jpg')

def detectFace(img, cropSize=224):

    dets = detector(img, 1)

    if (len(dets) < 0):
        raise ValueError("No face in img")

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(landmark_predictor(img, detection))

    images = dlib.get_face_chips(img, faces, size=cropSize)

    # cv2.imshow('image', images[0])
    # cv2.waitKey(0)
    return images[0]

def detectAllWebface(rootPath, saveRoot):

    for root, dirs, files in os.walk(rootPath):
        for dir in dirs:
            for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir)):
                for file in subfiles:
                    imgPath = os.path.join(rootPath, dir, file)
                    print(imgPath)
                    img = cv2.imread(imgPath)
                    try:
                        faceImg = detectFace(img, cropSize=224)
                    except:
                        continue
                    if dir not in os.listdir(saveRoot):
                        os.mkdir(os.path.join(saveRoot, dir))
                    saveImgPath = os.path.join(saveRoot, dir, file)
                    cv2.imwrite(saveImgPath, faceImg)


if __name__ == '__main__':

    webfaceRoot = "data/CASIA-WebFace/"
    saveRoot = "data/webface_detect/"

    detectAllWebface(webfaceRoot, saveRoot)