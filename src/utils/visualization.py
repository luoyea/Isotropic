import numpy as np
import trimesh
from matplotlib.colors import to_rgba

def visualize_angles(mesh, beta_min=30.0, beta_max=90.0):
    """标记异常角度三角形"""
    colors = np.ones((len(mesh.faces), 4)) * [0.7, 0.7, 0.7, 1.0]  # 默认灰色
    
    for i, face in enumerate(mesh.faces):
        # 计算三角形角度
        a, b, c = mesh.vertices[face]
        angles = [
            np.degrees(np.arccos(np.dot((b-a)/np.linalg.norm(b-a), (c-a)/np.linalg.norm(c-a)))),
            np.degrees(np.arccos(np.dot((c-b)/np.linalg.norm(c-b), (a-b)/np.linalg.norm(a-b)))),
            np.degrees(np.arccos(np.dot((a-c)/np.linalg.norm(a-c), (b-c)/np.linalg.norm(b-c))))
        ]
        
        max_angle = max(angles)
        min_angle = min(angles)
        
        # 标记超标三角形
        if max_angle > beta_max:
            colors[i] = to_rgba("lightcoral")  # 浅红色标记最大角超标
        elif min_angle < beta_min:
            colors[i] = to_rgba("lightblue")    # 浅蓝色标记最小角超标
    
    # 创建可视化网格
    colored_mesh = mesh.copy()
    colored_mesh.visual.face_colors = colors
    return colored_mesh
