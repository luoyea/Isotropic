import numpy as np 
import trimesh 
import os 
'''
网格文件读写工具：加载、保存和导出网格信息

'''
def load_mesh(file_path, file_type=None):
    try:
        mesh = trimesh.load(file_path,  file_type=file_type)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("文件格式不支持或非三角网格。")
        return mesh 
    except Exception as e:
        raise RuntimeError(f"加载网格失败：{str(e)}")
 
def save_mesh(mesh, output_path, file_type=None, **kwargs):
    try: 
        if file_type is None:
            _, ext = os.path.splitext(output_path) 
            file_type = ext[1:].lower()
        
        mesh.export(output_path,  file_type=file_type, **kwargs)
        print(f"网格已保存至：{output_path}")
    except Exception as e:
        raise RuntimeError(f"保存网格失败：{str(e)}")
 
def export_mesh_info(mesh, output_dir, prefix="mesh_info"):
    boundary_edges = []
    for edge_idx in range(len(mesh.edges)):
        # 获取与边相关联的面数
        face_count = len(np.where(mesh.face_adjacency_edges == edge_idx)[0])
        if face_count == 1:  # 边界边只属于一个面
            boundary_edges.append(mesh.edges[edge_idx])
    
    info = {
        "顶点数": len(mesh.vertices), 
        "面数": len(mesh.faces),
        "边界边数": len(boundary_edges),
        "平均面面积": mesh.area_faces.mean(), 
        "凸包体积": mesh.convex_hull.volume,
        "质心": mesh.center_mass.tolist(), 
        "对角线长度": mesh.extents.tolist(), 
        "边界框": {
            "最小值": mesh.bounds[0].tolist(), 
            "最大值": mesh.bounds[1].tolist() 
        }
    }
    
    # 保存为文本文件 
    output_path = os.path.join(output_dir,  f"{prefix}_info.txt") 
    with open(output_path, "w") as f:
        for key, value in info.items(): 
            f.write(f"{key}:  {value}\n")
    print(f"网格信息已保存至：{output_path}")
 
if __name__ == "__main__":
    input_path = "output/Botijo_to_5k_input.obj" 
    output_dir = "output/"
    mesh = load_mesh(input_path)

    save_mesh(mesh, os.path.join(output_dir,  "mesh_export.stl")) 
    export_mesh_info(mesh, output_dir, prefix="final_mesh")