import sys
import numpy as np
import open3d as o3d

if __name__ ==  "__main__":

    file1 = './objFile/sample1.obj'
    file2 = './objFile/sample2.obj'

    mesh1 = o3d.io.read_triangle_mesh(file1)
    mesh2 = o3d.io.read_triangle_mesh(file2)

    vertices_ds = 2500
    pcd1 = mesh1.sample_points_poisson_disk(vertices_ds)
    pcd2 = mesh2.sample_points_poisson_disk(vertices_ds)

    pcd1.paint_uniform_color([1, 0, 0])
    pcd2.paint_uniform_color([0, 0, 1])

    kdt = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    o3d.geometry.PointCloud.estimate_normals(pcd1, search_param=kdt)
    o3d.geometry.PointCloud.estimate_normals(pcd2, search_param=kdt)

    o3d.visualization.draw_geometries([pcd1, pcd2], "two objects visualize", 640, 480)

    th = 0.0000000001
    T = np.asarray([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # 初期値の情報
    info = o3d.registration.evaluate_registration(pcd1, pcd2,
                                      max_correspondence_distance=th,
                                      transformation=T)

    info = o3d.registration.registration_icp(pcd1, pcd2,
                                 max_correspondence_distance=th,
                                 init=T,
                                 estimation_method=o3d.registration.TransformationEstimationPointToPoint())

    pcd1.transform(info.transformation)
    o3d.visualization.draw_geometries([pcd1, pcd2], "two objects icp", 640, 480)