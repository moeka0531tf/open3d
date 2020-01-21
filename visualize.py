import sys
import open3d as o3d
import numpy as np

def file_read(file):
    print(mesh)
    # 頂点を表示
    print(np.asarray(mesh.vertices).shape[0])
    return mesh

def file_draw(file):
    o3d.visualization.draw_geometries([file], width=640, height=480)

if __name__ == "__main__":

    file = 'sample.obj'

    # read file
    mesh = o3d.io.read_triangle_mesh(file)

    # 頂点数
    vertices = np.asarray(mesh.vertices).shape[0]

    file_draw(mesh)

    # TriangleMesh -> PointCloud　に変換
    # 頂点数で統一が図れる
    pcd = mesh.sample_points_uniformly(vertices)

    # pointCloudの色の要素を取得できる
    print(np.asarray(pcd.colors))

    # 点の色を変更する(MeshからPCDにすると点の色がついていないため)
    pcd.paint_uniform_color([1, 0, 0])
    
    file_draw(pcd)
