import sys
import open3d as o3d
import numpy as np

# 通常の描写
def normal_visualize(file):
    o3d.visualization.draw_geometries([file], width=640, height=480)

if __name__ == "__main__":

    file = 'sample.obj'

    # read file
    mesh = o3d.io.read_triangle_mesh(file)

    # 頂点数
    vertices = np.asarray(mesh.vertices).shape[0]

    # TriangleMeshからPoinCloudへそのまま変換
    pcd = mesh.sample_points_poisson_disk(vertices)

    # TriangleMesh -> PointCloud　に2つの方法でダウンサンプリングして変換
    # 頂点数で統一が図れる
    # ダウンサンプリング時の頂点数
    vertices_ds = 5000
    pcd1 = mesh.sample_points_uniformly(vertices_ds)
    pcd2 = mesh.sample_points_poisson_disk(vertices_ds)

    # 点の色を変更
    pcd.paint_uniform_color([0, 0, 1])
    pcd1.paint_uniform_color([1, 0, 0])
    pcd2.paint_uniform_color([0, 1, 0])

    normal_visualize(pcd1)
    normal_visualize(pcd2)

    # PoinCloudでのダウンサンプリング
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    normal_visualize(downpcd)

    o3d.visualization.draw_geometries([pcd1, pcd2], width=640, height=480)


