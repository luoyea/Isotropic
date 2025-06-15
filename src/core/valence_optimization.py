import numpy as np 
import trimesh 
from trimesh import remesh 
''''
顶点度数优化：拉普拉斯平滑与拓扑调整
'''
def compute_triangle_angles(vertices, face):
    a, b, c = vertices[face]
    ab = b - a
    ac = c - a
    bc = c - b
    angle_A = np.degrees(np.arccos(np.clip(np.dot(ab, ac) /
                         (np.linalg.norm(ab) * np.linalg.norm(ac)), -1, 1)))
    angle_B = np.degrees(np.arccos(np.clip(np.dot(-ab, bc) /
                         (np.linalg.norm(ab) * np.linalg.norm(bc)), -1, 1)))
    angle_C = 180.0 - angle_A - angle_B
    return angle_A, angle_B, angle_C

def compute_valence(mesh):
    valence = np.zeros(len(mesh.vertices), dtype=int)
    for face in mesh.faces:
        for v in face:
            valence[v] += 1
    return valence

def flip_edge(mesh, edge):
    face_indices = mesh.edge_faces[tuple(edge)]
    if len(face_indices) != 2:
        return False

    f1, f2 = mesh.faces[face_indices[0]], mesh.faces[face_indices[1]]
    vs = set(f1) | set(f2)
    if len(vs) != 4:
        return False

    vs = list(vs)
    vs.remove(edge[0])
    vs.remove(edge[1])
    v_opposite_1, v_opposite_2 = vs

    new_faces = [
        [v_opposite_1, edge[0], v_opposite_2],
        [v_opposite_1, v_opposite_2, edge[1]]
    ]
    mesh.faces[face_indices[0]] = new_faces[0]
    mesh.faces[face_indices[1]] = new_faces[1]
    return True

def valence_error(valence, target=6):
    return np.abs(valence - target)

def topology_adjustment(mesh, beta_min=30.0, beta_max=120.0, target_valence=6, max_iterations=10):
    """
    拓扑优化
    """
    mesh = mesh.copy()

    for iteration in range(max_iterations):
        changed = False
        valence = compute_valence(mesh)

        for edge in mesh.edges_unique:
            face_indices = mesh.edge_faces[tuple(edge)]
            if len(face_indices) != 2:
                continue

            face1 = mesh.faces[face_indices[0]]
            face2 = mesh.faces[face_indices[1]]
            all_vertices = list(set(face1.tolist() + face2.tolist()))
            if len(all_vertices) != 4:
                continue
            angles_before = []
            for fid in [face1, face2]:
                angles_before.extend(compute_triangle_angles(mesh.vertices, fid))
            worst_before = min(angles_before) < beta_min or max(angles_before) > beta_max

            valence_before = valence_error(valence[all_vertices], target_valence).sum()
            
            if not flip_edge(mesh, edge):
                continue

            face1_new = mesh.faces[face_indices[0]]
            face2_new = mesh.faces[face_indices[1]]
            angles_after = []
            for fid in [face1_new, face2_new]:
                angles_after.extend(compute_triangle_angles(mesh.vertices, fid))
            worst_after = min(angles_after) < beta_min or max(angles_after) > beta_max
            valence_new = compute_valence(mesh)
            valence_after = valence_error(valence_new[all_vertices], target_valence).sum()

            if (worst_after and not worst_before) or (valence_after >= valence_before):
                flip_edge(mesh, edge)
            else:
                changed = True
        if not changed:
            break
    return mesh
 
def laplacian_smoothing(mesh, iterations=10, lambda_=0.5):
    """基于拉普拉斯平滑优化顶点位置"""
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
            vertices[v] = vertices[v] * (1 - lambda_) + avg * lambda_
    smoothed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return smoothed_mesh 
 
 
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
    """执行顶点度数优化"""
    if mesh is None:
        mesh = trimesh.load(input_path)
    
    if method == "smoothing":
        if smooth_type == "taubin":
            optimized = taubin_smoothing(mesh)
        else: 
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
