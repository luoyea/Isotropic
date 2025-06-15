import numpy as np 
import trimesh 
from trimesh import remesh 
'''
切向拉普拉斯平滑：(角度平滑)
    1. 计算每个顶点的邻居平均位置。
    2. 对每个顶点，计算其邻居顶点的平均位置，然后根据平滑系数更新顶点位置。
    3. 重复迭代，直到达到收敛条件或达到最大迭代次数。
'''
def laplacian_smoothing(mesh, iterations=10, lambda_=0.5):
    vertices = mesh.vertices.copy() 
    faces = mesh.faces  
    for _ in range(iterations):
        neighbors = [[] for _ in range(len(vertices))]
        for face in faces:
            for i in range(3):
                v = face[i]
                neighbors[v].extend([face[(i+1)%3], face[(i-1)%3]])
        
        # 计算平滑位置 
        for v in range(len(vertices)):
            if len(neighbors[v]) == 0:
                continue 
            avg = np.mean(vertices[neighbors[v]],  axis=0)
            vertices[v] = vertices[v] * (1 - lambda_) + avg * lambda_
    smoothed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return smoothed_mesh 
 
def tanh_smoothing(mesh, iterations=10, lambda_=0.5, epsilon=1e-3):
    """双曲正切平滑"""
    vertices = mesh.vertices.copy() 
    faces = mesh.faces  
    
    for _ in range(iterations):
        neighbors = [[] for _ in range(len(vertices))]
        for face in faces:
            for i in range(3):
                v = face[i]
                neighbors[v].extend([face[(i+1)%3], face[(i-1)%3]])
        
        for v in range(len(vertices)):
            if len(neighbors[v]) == 0:
                continue 
            avg = np.mean(vertices[neighbors[v]],  axis=0)
            delta = avg - vertices[v]
            norm = np.linalg.norm(delta)  + epsilon 
            vertices[v] += lambda_ * delta * np.tanh(1.0  / norm)
    
    smoothed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return smoothed_mesh 
 
def taubin_smoothing(mesh, iterations=10, lambda_=0.5, mu=0.5):
    """TAubin 平滑"""
    vertices = mesh.vertices.copy() 
    faces = mesh.faces  
    
    for _ in range(iterations):
        # 正向拉普拉斯平滑 
        neighbors = [[] for _ in range(len(vertices))]
        for face in faces:
            for i in range(3):
                v = face[i]
                neighbors[v].extend([face[(i+1)%3], face[(i-1)%3]])
        for v in range(len(vertices)):
            if len(neighbors[v]) == 0:
                continue 
            avg = np.mean(vertices[neighbors[v]],  axis=0)
            vertices[v] = vertices[v] * (1 - lambda_) + avg * lambda_
        
        # 逆向拉普拉斯平滑
        for v in range(len(vertices)):
            if len(neighbors[v]) == 0:
                continue 
            avg = np.mean(vertices[neighbors[v]],  axis=0)
            vertices[v] = vertices[v] * (1 + mu) - avg * mu 
    smoothed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return smoothed_mesh 
 
def adaptive_smoothing(mesh, iterations=10, lambda_=0.5, curvature_threshold=0.1):
    """自适应平滑"""
    vertices = mesh.vertices.copy() 
    faces = mesh.faces  
    vertex_normals = mesh.vertex_normals  
    curvature = np.linalg.norm(vertex_normals,  axis=1)
    curvature = curvature / np.max(curvature)
    
    for _ in range(iterations):
        neighbors = [[] for _ in range(len(vertices))]
        for face in faces:
            for i in range(3):
                v = face[i]
                neighbors[v].extend([face[(i+1)%3], face[(i-1)%3]])
        
        for v in range(len(vertices)):
            if len(neighbors[v]) == 0:
                continue 
            avg = np.mean(vertices[neighbors[v]],  axis=0)
            adjusted_lambda = lambda_ * (1 - curvature[v]) if curvature[v] > curvature_threshold else lambda_
            vertices[v] = vertices[v] * (1 - adjusted_lambda) + avg * adjusted_lambda 
    
    smoothed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return smoothed_mesh 
 
def smoothing(mesh, method="laplacian", **kwargs):  
    """执行网格平滑"""
    if method == "laplacian":
        smoothed = laplacian_smoothing(mesh, **kwargs)
    elif method == "tanh":
        smoothed = tanh_smoothing(mesh, **kwargs)
    elif method == "taubin":
        smoothed = taubin_smoothing(mesh, **kwargs)
    elif method == "adaptive":
        smoothed = adaptive_smoothing(mesh, **kwargs)
    else:
        raise ValueError("Unsupported method. Choose 'laplacian', 'tanh', 'taubin', or 'adaptive'.")
    
    return smoothed
