import trimesh
import numpy as np

# 判断一个三角形的最大角和最小角
def triangle_angles(vertices, face):
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    ab, bc, ca = b - a, c - b, a - c
    angles = [
        angle_between(-ca, ab),
        angle_between(-ab, bc),
        angle_between(-bc, ca),
    ]
    return np.max(angles), np.min(angles)

def angle_between(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

# 插入一个点到给定边的中点（返回更新后的网格）
def split_edge(mesh, edge):
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()

    v1, v2 = edge
    midpoint = (mesh.vertices[v1] + mesh.vertices[v2]) / 2.0
    midpoint_idx = len(vertices)
    vertices.append(midpoint.tolist())

    new_faces = []
    for face in faces[:]:
        if v1 in face and v2 in face:
            other = [idx for idx in face if idx != v1 and idx != v2][0]
            faces.remove(face)
            new_faces.append([v1, midpoint_idx, other])
            new_faces.append([midpoint_idx, v2, other])

    faces.extend(new_faces)
    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)

# 折叠边，删除一条边上的一个点并合并到另一个点
def collapse_edge(mesh, edge):
    v1, v2 = edge
    vertices = mesh.vertices.copy()
    faces = mesh.faces.tolist()

    vertices[v1] = (vertices[v1] + vertices[v2]) / 2.0
    updated_faces = []

    for face in faces:
        if v2 in face:
            if v1 in face:
                continue  # 删除重复点的面
            else:
                face = [v1 if idx == v2 else idx for idx in face]
        updated_faces.append(face)

    new_faces = [f for f in updated_faces if len(set(f)) == 3]
    return trimesh.Trimesh(vertices=vertices, faces=np.array(new_faces), process=False)

# 从三角形中获取最长边
def find_longest_edge(mesh, face):
    verts = mesh.vertices[face]
    d01 = np.linalg.norm(verts[0] - verts[1])
    d12 = np.linalg.norm(verts[1] - verts[2])
    d20 = np.linalg.norm(verts[2] - verts[0])
    if d01 >= d12 and d01 >= d20:
        return (face[0], face[1])
    elif d12 >= d01 and d12 >= d20:
        return (face[1], face[2])
    else:
        return (face[2], face[0])

# 从三角形中获取最短边
def find_shortest_edge(mesh, face):
    verts = mesh.vertices[face]
    d01 = np.linalg.norm(verts[0] - verts[1])
    d12 = np.linalg.norm(verts[1] - verts[2])
    d20 = np.linalg.norm(verts[2] - verts[0])
    if d01 <= d12 and d01 <= d20:
        return (face[0], face[1])
    elif d12 <= d01 and d12 <= d20:
        return (face[1], face[2])
    else:
        return (face[2], face[0])

# 主流程函数：角度优化操作
def isotropic(mesh, beta_min=35.0, beta_max=86.0, max_iter=5):
    for _ in range(max_iter):
        faces_to_split = []
        faces_to_collapse = []
        print(f"当前迭代：{_ + 1}/{max_iter}")
        for face in mesh.faces:
            max_angle, min_angle = triangle_angles(mesh.vertices, face)
            if max_angle > beta_max:
                faces_to_split.append(find_longest_edge(mesh, face))
            elif min_angle < beta_min:
                faces_to_collapse.append(find_shortest_edge(mesh, face))

        for edge in faces_to_split:
            mesh = split_edge(mesh, edge)
        for edge in faces_to_collapse:
            mesh = collapse_edge(mesh, edge)

    return mesh
