import sys
import os
import dlib
import glob
from skimage import io
import cv2
import open3d as o3d
import numpy as np
import math


# モデルの読み込み→回転→画像保存まで, ピンホールカメラの焦点距離を取得
def getRotateImage(objFile, dx, dy = 0, imageName = "faceimage.jpg", width=640, height=480):
    # read file
    mesh = o3d.io.read_triangle_mesh(objFile)

    # 回転行列の計算
    x = math.radians(dx)
    y = math.radians(dy)

    # x軸回転
    x = math.radians(dx)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(y), np.sin(y)],
                   [0, -np.sin(y), np.cos(y)]])
    mesh.rotate(Rx)

    # y軸回転
    Ry = np.array([[np.cos(x), 0, -np.sin(x)],
                   [0, 1, 0],
                   [np.sin(x), 0, np.cos(x)]])
    mesh.rotate(Ry)

    # 回転後表示
    vis = o3d.visualization.Visualizer()
    vis.create_window(width = width, height = height)
    vis.add_geometry(mesh)
    vis.run()

    # 画像保存
    vis.capture_screen_image("./faceRotateData/" + imageName)

    # ピンホールカメラモデルのデータに変換
    ctr = vis.get_view_control()
    pinhole = ctr.convert_to_pinhole_camera_parameters()
    pi = pinhole.intrinsic
    im = pi.intrinsic_matrix

    vis.destroy_window()

    return im


# 顔特徴点を取得する関数
def getFacialLandmarks(imageName):
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    face_path = "./faceRotateData/" + imageName

    # 特徴点を保存する配列
    points = [];

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for f in glob.glob(face_path):
        print("Processing file: {}".format(f))
        img = io.imread(f)
        dets = detector(img, 1)

        for k, d in enumerate(dets):

            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)

            # add part
            for i in range(0, 68):
                s = str(shape.part(i))
                s = s.replace("(", "")
                s = s.replace(")", "")
                s = s.replace(",", "")
                x, y = s.split(" ")
                points.append((int(x), int(y), 1))

    return points

# カメラ座標を取得
def getCameraCoordinate(points, im):

    cameraPoints = []

    for i in range(0, len(points)):
        array = np.array([points[i]])
        cameraPoint = np.dot(im, array.T)
        cameraPoints.append(cameraPoint.T.reshape(3,).tolist())

    return cameraPoints


if __name__ == "__main__":

    file = 'file'
    imageName = 'faceImage-15degree.jpg'

    # 画像サイズ
    width = 640
    height = 480

    # 15度x回転させた
    # 焦点距離を取得
    im = getRotateImage(file, 15, 0, imageName, width, height)

    # 顔特徴点取得
    points = getFacialLandmarks(imageName)

    # 顔特徴点のカメラ座標を取得
    cameraPoints = getCameraCoordinate(points, im)
