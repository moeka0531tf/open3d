import dlib
import glob
from skimage import io
import open3d as o3d
import numpy as np
import math


# モデルの読み込み→回転→画像保存まで, ピンホールカメラの焦点距離を取得
def getRotateImage(objFile, dx, dy = 0, imageName = "faceimage.jpg", width=640, height=480):
    # read file
    mesh = o3d.io.read_triangle_mesh(objFile)
    # 頂点座標取得
    # print(np.array(mesh.vertices))

    axis = mesh.get_axis_aligned_bounding_box()
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[0, 0, 0])
    # print(coordinate)

    # 回転行列の計算
    # ラジアンに変換
    x = math.radians(dx)
    y = math.radians(dy)

    # x軸回転
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(y), np.sin(y)],
                   [0, -np.sin(y), np.cos(y)]])
    mesh.rotate(Rx)

    # y軸回転
    Ry = np.array([[np.cos(x), 0, -np.sin(x)],
                   [0, 1, 0],
                   [np.sin(x), 0, np.cos(x)]])
    mesh.rotate(Ry)
    # 回転後の頂点座標取得
    # print(np.array(mesh.vertices))

    # 回転後表示
    vis = o3d.visualization.Visualizer()
    vis.create_window(width = width, height = height)
    vis.add_geometry(mesh)
    vis.add_geometry(coordinate)
    vis.run()

    # 画像保存
    vis.capture_screen_image("./faceRotateData/" + imageName)

    # ピンホールカメラモデルのデータに変換
    ctr = vis.get_view_control()
    pinhole = ctr.convert_to_pinhole_camera_parameters()
    # カメラ外部パラメータ
    extrinsic = pinhole.extrinsic
    # print(extrinsic)
    pi = pinhole.intrinsic
    im = pi.intrinsic_matrix
    # print(im)

    vis.destroy_window()

    return im, extrinsic

# 顔特徴点を取得する関数
def getFacialLandmarks(imageName):
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    face_path = "./faceRotateData/" + imageName

    # 特徴点を保存する配列
    points = [];

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for f in glob.glob(face_path):
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
def getCameraPoints(points, im):

    cameraPoints = []

    for i in range(0, len(points)):
        array = np.array([points[i]])
        cameraPoint = np.dot(np.linalg.inv(im), array.T)
        cameraPoints.append(cameraPoint.T.reshape(3,).tolist())

    return cameraPoints

# カメラ座標からワールド座標への変換
def getObjectPoints(dx, dy, cameraPoints, extrinsic, points):

    # ラジアンに変換
    x = math.radians(dx)
    y = math.radians(dy)

    focal_length = 415.69219382
    finalPoints = []

    # x軸回転
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(y), np.sin(y)],
                   [0, -np.sin(y), np.cos(y)]])

    # y軸回転
    Ry = np.array([[np.cos(x), 0, -np.sin(x)],
                   [0, 1, 0],
                   [np.sin(x), 0, np.cos(x)]])

    # 最終的な特徴点
    for i in range(0, len(cameraPoints)):
        array = np.array(points[i])
        array = np.insert(array, 3, 1)
        array = np.array([array])
        objectPoint = np.dot(np.linalg.inv(extrinsic), array.T)
        objectPoint = np.delete(objectPoint, 3, 0)

        temp = objectPoint.T.reshape(3, ).tolist()

        z = focal_length * temp[1] / points[i][1]
        finalPoints.append([temp[0], temp[1], z])

        # 正面向きの座標を取得する
        point = np.dot(np.linalg.inv(Ry), objectPoint)
        points.append(point.T.reshape(3,).tolist())

    # 描画用のポイントを整備する
    finalArray = np.array(finalPoints)

    print(finalArray)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(finalArray)
    o3d.visualization.draw_geometries([pcd])
    return finalArray


if __name__ == "__main__":

    file = 'file'
    imageName = 'faceImage-15degree.jpg'

    # 画像サイズ
    width = 640
    height = 480

    # 回転角度
    x = 0
    y = 0

    # 15度x回転させた
    # 焦点距離を取得
    im, extrinsic = getRotateImage(file, x, y, imageName, width, height)

    # 顔特徴点取得
    points = getFacialLandmarks(imageName)

    # 顔特徴点のカメラ座標を取得
    cameraPoints = getCameraPoints(points, im)

    # カメラ座標から実際の座標に変換
    objectPoints = getObjectPoints(x, y, cameraPoints, extrinsic, points)
    # print(objectPoints)

    point = np.array([objectPoints])
    # print(point)

