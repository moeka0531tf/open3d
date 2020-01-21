import sys
import open3d as o3d
import numpy as np

# 通常の描写
def normal_visualize(file):
    o3d.visualization.draw_geometries([file], width=640, height=480)

# コールバック関数でオブジェクトを回転させる
def custom_draw_geometry_with_rotation(file):

     def rotate_view(vis):
         ctr = vis.get_view_control()
         ctr.rotate(10.0, 0.0)
         return False

     o3d.visualization.draw_geometries_with_animation_callback([file], rotate_view, width=640, height=480)

# オブジェクトを任意の方向に回転させて表示
def rotate_visualize(file, x=0, y=0):
     vis = o3d.visualization.Visualizer()
     vis.create_window()
     vis.add_geometry(file)

     ctr = vis.get_view_control()
     ctr.rotate(x, y)
     vis.run()
     vis.destroy_window()
     
if __name__ == "__main__":

    file = 'sample.obj'

    # read file
    mesh = o3d.io.read_triangle_mesh(file)

    # 頂点数
    vertices = np.asarray(mesh.vertices).shape[0]

    # メッシュファイルを表示
    normal_visualize(mesh)

    # TriangleMesh -> PointCloud　に変換
    # 頂点数で統一が図れる
    pcd = mesh.sample_points_uniformly(vertices)

    # pointCloudの色の要素を取得できる
    print(np.asarray(pcd.colors))

    # 点の色を変更する(MeshからPCDにすると点の色がついていないため)
    pcd.paint_uniform_color([1, 0, 0])

    # PointCloudを表示
    normal_visualize(pcd)

    # オブジェクトを回転させて表示
    custom_draw_geometry_with_rotation(pcd)

    # オブジェクトを任意の角度に回転させて表示
    rotate_visualize(pcd, 90, 0)