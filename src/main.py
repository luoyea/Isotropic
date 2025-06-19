import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))    # 项目根目录 
sys.path.append(str(Path(__file__).parent))           # src 目录 
import os 
import argparse 
from utils import parameters 
from utils.parameters  import load_params, get_smoothing_params, get_topology_params, get_cvt_params, get_general_params ,get_nob_params, get_isotropic_params
from utils.mesh_io   import load_mesh, save_mesh, export_mesh_info 
from core.smoothing   import smoothing 
from utils.geometry   import compute_convex_hull, compute_mesh_centroid 
from src.core.valence_optimization   import topology_adjustment 
from src.core.angle_optimization   import cvt_optimization,nob_optimization,isotropic
import numpy as np 
import trimesh 
from utils.visualization  import visualize_angles 
from core.initialization  import initialize_mesh 
 
def parse_args():
    parser = argparse.ArgumentParser(description="网格处理工具")
    parser.add_argument("--input",   type=str, required=True, help="输入网格路径")
    parser.add_argument("--output_dir",   type=str, default="output/", help="输出目录")
    parser.add_argument("--smoothing",   type=str, choices=["laplacian", "tanh", "taubin", "adaptive"], help="平滑方法")
    parser.add_argument("--topology",   action="store_true", help="是否执行拓扑优化")
    parser.add_argument("--cvt",   action="store_true", help="是否执行 CVT 优化")
    parser.add_argument("--config",   type=str, help="自定义参数配置文件路径")
    parser.add_argument("--initialize",  action="store_true", help="是否执行初始化网格生成")
    parser.add_argument("--nob",   action="store_true", help="是否执行NOB优化") 
    parser.add_argument("--isotropic",   action="store_true", help="是否执行等距优化")
    return parser.parse_args() 
 
def process_mesh(mesh, params):
    """执行网格处理流程."""
    # 平滑处理 
    if params["smoothing"]:
        smoothing_params = params["smoothing"].copy()
        method = smoothing_params.pop("method")
        print(f"执行 {method} 平滑...")
        mesh = smoothing(mesh, method=method, **smoothing_params)
    
    # 拓扑优化 
    if params["topology"]:
        print("执行拓扑优化...")
        mesh = topology_adjustment(mesh, **params["topology"])
    
    # CVT 优化 
    if params["cvt"]:
        print("执行 CVT 优化...")
        cvt_params = params["cvt"].copy()
        # 移除属于NOB优化的参数
        cvt_params.pop("num_samples", None) 
        cvt_params.pop("max_iterations", None) 
        cvt_params.pop("tolerance", None)
        mesh = cvt_optimization(
            mesh,
            iterations=cvt_params.pop("iterations", 5),
            beta_min=cvt_params.pop("beta_min", 35.0),
            beta_max=cvt_params.pop("beta_max", 86.0),
            **cvt_params 
        )
    # NOB 优化
    if params["nob"]:
        print("执行NOB优化...")
        nob_params = params["nob"].copy()
        nob_params.pop("num_samples", None)
        nob_params.pop("tolerance", None)
        nob_params.pop("max_iterations", None)
        
        mesh = nob_optimization(
            mesh,
            beta_min=nob_params.pop("beta_min", 35.0),
            beta_max=nob_params.pop("beta_max", 86.0),
            max_iter=nob_params.pop("max_iter", 10),
            **nob_params
        )
    if params["isotropic"]:
        print("执行等距优化...")
        isotropic_params = params["isotropic"].copy()
        mesh = isotropic(mesh, 
            beta_min=isotropic_params.pop("beta_min", 35.0),
            beta_max=isotropic_params.pop("beta_max", 86.0),
            max_iter=isotropic_params.pop("max_iter", 5),
            **isotropic_params
        )
        return mesh 
 
def analyze_angles(mesh, beta_min=35.0, beta_max=86.0):
    """统计三角形角度分布"""
    triangles = mesh.triangles  
    angles = []
    for tri in triangles:
        a, b, c = tri[1]-tri[0], tri[2]-tri[1], tri[0]-tri[2]
        angles.extend([ 
            np.degrees(np.arccos(np.dot(-a,  c)/(np.linalg.norm(a)*np.linalg.norm(c)))), 
            np.degrees(np.arccos(np.dot(-b,  a)/(np.linalg.norm(b)*np.linalg.norm(a)))), 
            np.degrees(np.arccos(np.dot(-c,  b)/(np.linalg.norm(c)*np.linalg.norm(b)))) 
        ])
    
    angles = np.array(angles) 
    over_max = np.sum(angles  > beta_max) / len(angles)
    under_min = np.sum(angles  < beta_min) / len(angles)
    return over_max * 100, under_min * 100 
 
def main():
    args = parse_args()
    params = load_params(args.config) 
    
    # 在加载网格后添加初始化逻辑 
    if args.initialize: 
        print("执行初始化网格生成...")
        # 获取目标顶点数参数 
        target_vertices = params["general"].get("target_vertices", 500000)
        # 初始化 
        mesh = initialize_mesh(
            input_path=args.input, 
            output_path=os.path.join(args.output_dir,  "initialized.obj"), 
            target_vertices=target_vertices,
            uniform=True 
        )
    else:
        # 原有加载逻辑 
        mesh = load_mesh(args.input) 
    
    # 处理流程 
    processed_params = {
        "smoothing": get_smoothing_params(args.smoothing)  if args.smoothing  else None,
        "topology": get_topology_params() if args.topology  else None,  

        "cvt": get_cvt_params() if args.cvt  else None,
        "nob": get_nob_params() if args.nob  else None,  
        "isotropic":get_isotropic_params() if args.isotropic else None  # 假设 isotropic 优化不在此处处理
    }
    processed_mesh = process_mesh(mesh, processed_params)
    
    # 保存结果 
    base_name = os.path.splitext(os.path.basename(args.input))[0]  
    output_path = os.path.join(args.output_dir,   f"{base_name}_processed.obj")  
    save_mesh(processed_mesh, output_path)
    print(f"处理后的网格已保存至：{output_path}")
    
    # 导出信息 
    export_mesh_info(processed_mesh, args.output_dir,   prefix=base_name)
    
    # 可选：计算凸包并保存 
    convex_hull = compute_convex_hull(processed_mesh.vertices)  
    save_mesh(convex_hull, os.path.join(args.output_dir,   f"{base_name}_convex_hull.obj"))  
    
    # 质量分析
    beta_min = get_cvt_params().get("beta_min", 35.0)
    beta_max = get_cvt_params().get("beta_max", 86.0)
    
    over_max_percent, under_min_percent = analyze_angles(processed_mesh, beta_min=beta_min, beta_max=beta_max)
    
    print(f"角度大于 {beta_max} 度的比例: {over_max_percent:.2f}%")
    print(f"角度小于 {beta_min} 度的比例: {under_min_percent:.2f}%")
    
if __name__ == "__main__":
    main()
