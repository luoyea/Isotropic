import numpy as np 
import trimesh 
from trimesh import remesh 
from scipy.spatial  import Delaunay 
'''
角度优化：基于质心沃罗诺伊镶嵌（CVT）和非钝角重网格化（NOB）
'''
def compute_triangle_angles(mesh):
    """计算网格中每个三角形的三个角度（单位：度）."""
    angles = []
    for face in mesh.faces:
        v0, v1, v2 = mesh.vertices[face]
        
        # 添加数值稳定性检查
        epsilon = 1e-8
        a = v1 - v0
        b = v2 - v0
        c = v2 - v1
        
        # 计算角度（添加安全除法）
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
 
def cvt_optimization(mesh, num_samples=1000, beta_min=35.0, beta_max=86.0, max_iterations=10, tolerance=1e-4):
    """改进后的CVT优化（带角度约束）"""
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    # 保留原始顶点数
    original_vertex_count = len(mesh.vertices)
    
    for _ in range(max_iterations):
        voronoi_centers = []
        for i in range(len(vertices)):
            connected_faces = np.where(np.any(faces == i, axis=1))[0]
            if len(connected_faces) == 0:
                voronoi_centers.append(vertices[i])
                continue

            region_vertices = vertices[faces[connected_faces].flatten()]
            center = np.mean(region_vertices, axis=0)
            voronoi_centers.append(center)

        vertices = np.array(voronoi_centers)

        try:
            delaunay = Delaunay(vertices)
            faces = delaunay.simplices
        except:
            break

        # 角度约束处理
        new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        angles = compute_triangle_angles(new_mesh)
        
        # 过滤不符合角度约束的面
        valid_faces = []
        for idx, face_angles in enumerate(angles):
            if idx >= len(faces):
                continue  # 防止索引越界
            if np.all(face_angles > beta_min) and np.all(face_angles < beta_max):
                valid_faces.append(faces[idx])  # 直接使用当前索引
        
        # 更新拓扑结构
        if len(valid_faces) > 0:
            faces = np.array(valid_faces)
            print(f"迭代 {_}: 有效面数 {len(faces)}，角度达标率 {len(valid_faces)/len(angles)*100:.1f}%")
        else:
            print("警告：未找到符合角度约束的面，终止迭代")
            break

    # 保持顶点数量稳定
    if len(vertices) != original_vertex_count:
        vertices = vertices[:original_vertex_count]
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def nob_optimization(mesh, beta_min=30.0, beta_max=90.0):
    """非钝角重网格化（NOB），消除钝角和小角."""
    # 使用remesh 库进行初步优化 
    optimized = remesh.barycentric(mesh,  count=len(mesh.vertices)) 
    
    # 迭代消除钝角（>90°）
    angles = compute_triangle_angles(optimized)
    while np.any(angles  > beta_max):
        # 找到第一个钝角三角形 
        idx = np.where(angles  > beta_max)[0][0]
        face = optimized.faces[idx] 
        
        # 对钝角边进行分裂
        edge = optimized.edges_unique[idx] 
        new_vertex = np.mean(optimized.vertices[edge],  axis=0)
        optimized.vertices  = np.vstack([optimized.vertices,  new_vertex])
        optimized = remesh.subdivide(optimized) 
        
        # 重新计算角度 
        angles = compute_triangle_angles(optimized)
    
    return optimized 
 
def angle_optimization(input_path, output_path, method="cvt", beta_min=30.0, beta_max=90.0):
    """主函数：执行角度优化.""" 
    mesh = trimesh.load(input_path) 
    
    if method == "cvt":
        optimized = cvt_optimization(mesh, beta_min=beta_min, beta_max=beta_max)
    elif method == "nob":
        optimized = nob_optimization(mesh, beta_min=beta_min, beta_max=beta_max)
    else:
        raise ValueError("Unsupported method. Choose 'cvt' or 'nob'.")
    
    # 保存优化后的网格 
    optimized.export(output_path) 
    print(f"角度优化完成，顶点数：{len(optimized.vertices)} ，角度范围：{beta_min}-{beta_max}°")
 
if __name__ == "__main__":
    input_path = "output/Botijo_to_5k_input_uniform.obj" 
    output_path_cvt = "output/optimized_cvt.obj" 
    output_path_nob = "output/optimized_nob.obj" 
    
    # CVT 优化 
    angle_optimization(input_path, output_path_cvt, method="cvt", beta_min=35.0, beta_max=86.0)
    
    # NOB 优化 
    angle_optimization(input_path, output_path_nob, method="nob", beta_min=30.0, beta_max=90.0)