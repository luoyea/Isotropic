import numpy as np 
import trimesh 
from trimesh import remesh 
from scipy.spatial  import Delaunay 
'''
角度优化：基于质心沃罗诺伊镶嵌（CVT）和非钝角重网格化（NOB）
'''
def compute_triangle_angles(mesh):
    """计算网格中每个三角形的三个角度"""
    angles = []
    for face in mesh.faces:
        v0, v1, v2 = mesh.vertices[face]
        epsilon = 1e-8
        a = v1 - v0
        b = v2 - v0
        c = v2 - v1

        dot_bc = np.dot(b, c)
        norm_b = np.linalg.norm(b) + epsilon
        norm_c = np.linalg.norm(c) + epsilon
        cos_a = np.clip(dot_bc / (norm_b * norm_c), -1.0, 1.0)
        angle_a = np.degrees(np.arccos(cos_a))

        dot_ac = np.dot(a, c)
        norm_a = np.linalg.norm(a) + epsilon
        cos_b = np.clip(dot_ac / (norm_a * norm_c), -1.0, 1.0)
        angle_b = np.degrees(np.arccos(cos_b))
        angle_c = 180 - angle_a - angle_b
        angles.append([angle_a, angle_b, angle_c])
    return np.array(angles) 
 
def is_boundary_vertex(mesh, vertex_index):
    return mesh.vertex_neighbors[vertex_index].__len__() != mesh.vertex_faces[vertex_index].__len__()

def compute_weighted_centroid(vertices, face):
    a, b, c = vertices[face]
    centroid = (a + b + c) / 3.0
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    return centroid, area

def cvt_optimization(mesh, iterations=5):
    """CVT 优化"""
    vertices = mesh.vertices.copy()
    faces = mesh.faces
    is_boundary = np.array([is_boundary_vertex(mesh, i) for i in range(len(vertices))])

    for _ in range(iterations):
        new_vertices = np.zeros_like(vertices)
        weight_sum = np.zeros((len(vertices), 1))

        for face in faces:
            centroid, area = compute_weighted_centroid(vertices, face)
            for idx in face:
                new_vertices[idx] += centroid * area
                weight_sum[idx] += area

        # 更新非边界顶点位置
        for i in range(len(vertices)):
            if not is_boundary[i] and weight_sum[i] > 1e-6:
                vertices[i] = new_vertices[i] / weight_sum[i]

        # 更新 mesh 用于下次迭代邻接信息
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    optimized_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return optimized_mesh


def nob_optimization(mesh, beta_min=35.0, beta_max=86.0):
    """NOB"""
    optimized = remesh.barycentric(mesh,  count=len(mesh.vertices)) 
    angles = compute_triangle_angles(optimized)
    while np.any(angles  > beta_max):
        idx = np.where(angles  > beta_max)[0][0]
        face = optimized.faces[idx] 
        edge = optimized.edges_unique[idx] 
        new_vertex = np.mean(optimized.vertices[edge],  axis=0)
        optimized.vertices  = np.vstack([optimized.vertices,  new_vertex])
        optimized = remesh.subdivide(optimized) 
        
        # 重新计算角度 
        angles = compute_triangle_angles(optimized)
    
    return optimized 
 
def angle_optimization(input_path, output_path, method="cvt", beta_min=35.0, beta_max=86.0):
    mesh = trimesh.load(input_path) 
    
    if method == "cvt":
        optimized = cvt_optimization(mesh, beta_min=beta_min, beta_max=beta_max)
    elif method == "nob":
        optimized = nob_optimization(mesh, beta_min=beta_min, beta_max=beta_max)
    else:
        raise ValueError("Unsupported method. Choose 'cvt' or 'nob'.")
    optimized.export(output_path) 
    print(f"角度优化完成，顶点数：{len(optimized.vertices)} ，角度范围：{beta_min}-{beta_max}°")
 
if __name__ == "__main__":
    input_path = "output/Botijo_to_5k_input_uniform.obj" 
    output_path_cvt = "output/optimized_cvt.obj" 
    output_path_nob = "output/optimized_nob.obj" 
    
    # CVT 优化 
    angle_optimization(input_path, output_path_cvt, method="cvt", beta_min=35.0, beta_max=86.0)
    
    # NOB 优化 
    angle_optimization(input_path, output_path_nob, method="nob", beta_min=35.0, beta_max=86.0)
