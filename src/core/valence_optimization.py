import numpy as np 
import trimesh 
from trimesh import remesh 
''''
顶点度数优化：拉普拉斯平滑与拓扑调整
'''
def compute_vertex_valence(mesh):
    """计算每个顶点的度数（连接的边数）."""
    # 统计每个顶点在 faces 中出现的次数 
    valence = np.zeros(len(mesh.vertices),  dtype=int)
    for face in mesh.faces: 
        for vertex in face:
            valence[vertex] += 1 
    return valence 
 
def laplacian_smoothing(mesh, iterations=10, lambda_=0.5):
    """基于拉普拉斯平滑优化顶点位置."""
    vertices = mesh.vertices.copy() 
    faces = mesh.faces  
    
    for _ in range(iterations):
        # 计算每个顶点的邻居平均位置 
        neighbors = [[] for _ in range(len(vertices))]
        for face in faces:
            for i in range(3):
                v = face[i]
                neighbors[v].extend([face[(i+1)%3], face[(i-1)%3]])
        
        # 去重并计算平滑位置 
        for v in range(len(vertices)):
            if len(neighbors[v]) == 0:
                continue 
            avg = np.mean(vertices[neighbors[v]],  axis=0)
            vertices[v] = vertices[v] * (1 - lambda_) + avg * lambda_
    
    # 生成平滑后的网格 
    smoothed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return smoothed_mesh 
 
def topology_adjustment(mesh, target_valence=6, max_iterations=20):
    """
    使用remesh进行拓扑调整
    """
    # 转换为面片网格格式
    mesh_process = mesh.copy()
    
    # 迭代式拓扑优化
    for _ in range(max_iterations):
        # 获取当前网格的顶点和面
        vertices = mesh_process.vertices
        faces = mesh_process.faces
        
        # 使用细分控制边长度
        new_vertices, new_faces = trimesh.remesh.subdivide_to_size(
            vertices=vertices,
            faces=faces,
            max_edge=np.mean(mesh.edges_unique_length) * 0.8,
            max_iter=5
        )
        mesh_process = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        
        # 使用标准二次误差度量简化网格
        mesh_process = mesh_process.simplify_quadric_decimation(
            face_count=int(len(faces) * 0.8)
        )
    
    return mesh_process
 
def taubin_smoothing(mesh, iterations=10, lambda_=0.5, mu=-0.53):
    """Taubin平滑算法"""
    vertices = mesh.vertices.copy()
    faces = mesh.faces
    
    for _ in range(iterations):
        # 正向平滑
        neighbors = [[] for _ in range(len(vertices))]
        for face in faces:
            for i in range(3):
                v = face[i]
                neighbors[v].extend([face[(i+1)%3], face[(i-1)%3]])
        
        for v in range(len(vertices)):
            if len(neighbors[v]) == 0:
                continue 
            avg = np.mean(vertices[neighbors[v]], axis=0)
            vertices[v] += lambda_ * (avg - vertices[v])
        
        # 反向..
        for v in range(len(vertices)):
            if len(neighbors[v]) == 0:
                continue 
            avg = np.mean(vertices[neighbors[v]], axis=0)
            vertices[v] += mu * (avg - vertices[v])
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def valence_optimization(input_path, output_path, method="smoothing", target_valence=6, smooth_type="laplacian"):
    """main：执行顶点度数优化"""
    if mesh is None:
        mesh = trimesh.load(input_path)
    
    if method == "smoothing":
        if smooth_type == "taubin":
            optimized = taubin_smoothing(mesh)
        else:  # 默认使用拉普拉斯
            optimized = laplacian_smoothing(mesh)
    elif method == "topology":
        optimized = topology_adjustment(mesh, target_valence=target_valence)
    else:
        raise ValueError("Unsupported method")
    if output_path is None:
        optimized.export(output_path)
    print(f"优化完成，方法：{method}，顶点数：{len(optimized.vertices)}")
 
if __name__ == "__main__":
    input_path = "output/Botijo_to_5k_input_cvt.obj" 
    output_path_smoothing = "output/valence_smoothed.obj" 
    output_path_topology = "output/valence_adjusted.obj" 
    
    # 拉普拉斯平滑优化 
    valence_optimization(input_path, output_path_smoothing, method="smoothing")
    
    # 拓扑调整优化 
    valence_optimization(input_path, output_path_topology, method="topology", target_valence=6)