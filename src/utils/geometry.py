import numpy as np 
import trimesh 
from trimesh import transformations as tf 
import matplotlib.pyplot  as plt 
from mpl_toolkits.mplot3d  import Axes3D 
'''
几何处理工具：计算距离、法线、变换等

'''
def compute_point_to_plane_distance(point, plane_point, plane_normal):
    plane_normal = plane_normal / np.linalg.norm(plane_normal) 
    return np.dot(point  - plane_point, plane_normal)
 
def compute_face_normals(vertices, faces):

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0 
    edge2 = v2 - v0 

    face_normals = np.cross(edge1,  edge2)
    face_normals /= np.linalg.norm(face_normals,  axis=1, keepdims=True)
    return face_normals 
 
def transform_points(points, translation=None, rotation=None, scale=None):
    #缩放
    if scale is not None:
        points *= scale 
    
    # 旋转
    if rotation is not None:
        if rotation.shape  == (3,):
            rotation = tf.euler_matrix(*rotation)[:3,  :3]
        points = np.dot(points,  rotation.T)
    
    # 平移 
    if translation is not None:
        points += translation 
    
    return points 
 
def compute_bounding_box(vertices):
    
    min_bound = np.min(vertices,  axis=0)
    max_bound = np.max(vertices,  axis=0)
    return min_bound, max_bound 
 
def check_point_in_face(point, face_vertices):
   
    v0, v1, v2 = face_vertices 
    vectors = np.array([v1  - v0, v2 - v0, point - v0])
    bary = np.linalg.lstsq(vectors.T,  np.array([1,  1, 1]), rcond=None)[0]
    return np.all(bary  >= 0) and np.sum(bary)  <= 1 + 1e-6 
 
def compute_mesh_centroid(mesh):
    return np.mean(mesh.vertices,  axis=0)
 
def generate_grid_points(bounds, resolution=0.1):
    """
    生成均匀分布的网格点
    """
    x = np.arange(bounds[0][0],  bounds[1][0], resolution)
    y = np.arange(bounds[0][1],  bounds[1][1], resolution)
    z = np.arange(bounds[0][2],  bounds[1][2], resolution)
    grid = np.array(np.meshgrid(x,  y, z)).T.reshape(-1,  3)
    return grid 
 
def compute_closest_point(mesh, query_point):
    """
    计算查询点到网格的最近点
    """
    _, closest_point = mesh.nearest.on_surface(query_point) 
    return closest_point 
 
def compute_convex_hull(points):
    """
    计算点集的凸包

    """
    hull = trimesh.convex.convex_hull(points) 
    return hull 
 
def visualize_bounding_box(ax, mesh, color="red", alpha=0.2):
    """
绘制网格的包围盒
    """
    min_bound, max_bound = compute_bounding_box(mesh.vertices) 
    edges = np.array([ 
        [min_bound, [max_bound[0], min_bound[1], min_bound[2]]],
        [[max_bound[0], min_bound[1], min_bound[2]], max_bound],
        [max_bound, [min_bound[0], max_bound[1], max_bound[2]]],
        [[min_bound[0], max_bound[1], max_bound[2]], min_bound],
    ])
    for edge in edges:
        ax.plot(*edge.T,  color=color, alpha=alpha)
 
if __name__ == "__main__":

    mesh = trimesh.load("output/optimized_cvt.obj") 
    face_normals = compute_face_normals(mesh.vertices,  mesh.faces) 

    centroid = compute_mesh_centroid(mesh)

    convex_hull = compute_convex_hull(mesh.vertices) 
    fig = plt.figure() 
    ax = fig.add_subplot(111,  projection='3d')
    visualize_bounding_box(ax, mesh)
    plt.show() 
