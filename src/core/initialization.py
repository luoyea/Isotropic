import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from trimesh.sample import sample_surface_even

'''
初始化网格生成模块
'''

def compute_surface_area(mesh):
    return mesh.area

def compute_curvature_radii(mesh):
    vertex_normals = mesh.vertex_normals
    curvature = np.linalg.norm(vertex_normals, axis=1)
    curvature_radii = 1.0 / (curvature + 1e-8)
    return curvature_radii

def compute_sizing_function(mesh, N, uniform=True):
    """计算目标边长 L /自适应的 sizing function ρ(xi)"""
    surface_area = compute_surface_area(mesh)
    if uniform:
        L = np.sqrt((2 * surface_area) / (np.sqrt(3) * N))
        return np.full(len(mesh.vertices), L)
    else:
        r_i = compute_curvature_radii(mesh)
        rho = np.sqrt(6 * r_i - 3/2)
        hmin = np.min(rho)
        hmax = np.max(rho)
        rho = np.clip(rho, hmin, hmax)
        return rho

def adaptive_sample_by_curvature(mesh, target_vertices, rho):
    """自适应地从表面采样点"""
    face_rho = rho[mesh.faces].mean(axis=1)
    prob = face_rho / np.sum(face_rho)
    sampled_faces = np.random.choice(len(mesh.faces), size=target_vertices, p=prob)
    sampled_points = []

    for face_idx in sampled_faces:
        tri = mesh.triangles[face_idx]
        r1 = np.sqrt(np.random.rand())
        r2 = np.random.rand()
        a = (1 - r1)
        b = r1 * (1 - r2)
        c = r1 * r2
        point = a * tri[0] + b * tri[1] + c * tri[2]
        sampled_points.append(point)
    return np.array(sampled_points)

def initialize_mesh(input_path, output_path, target_vertices, uniform=True):
    mesh = trimesh.load(input_path, process=False)
    print(f"加载输入网格：{input_path}，顶点数：{len(mesh.vertices)}")
    sizing_function = compute_sizing_function(mesh, target_vertices, uniform=uniform)
    if uniform:
        print("执行均匀重采样...")
        points, _ = sample_surface_even(mesh, count=target_vertices)
    else:
        print("执行自适应重采样...")
        points = adaptive_sample_by_curvature(mesh, target_vertices, sizing_function)

    print("计算凸包构建初始网格...")
    hull = ConvexHull(points)
    remeshed = trimesh.Trimesh(vertices=points, faces=hull.simplices, process=False)

    remeshed.export(output_path)
    print(f"初始网格生成完成，顶点数：{len(remeshed.vertices)}")
    return remeshed

if __name__ == "__main__":
    input_path = "input/Botijo_to_5k_input.obj"
    output_path_uniform = "output/initial_uniform.obj"
    output_path_adaptive = "output/initial_adaptive.obj"

    initialize_mesh(input_path, output_path_uniform, target_vertices=1000, uniform=True)
    initialize_mesh(input_path, output_path_adaptive, target_vertices=1000, uniform=False)
