import json 
import os 
'''
参数配置模块：加载和管理网格处理算法的参数

'''
DEFAULT_PARAMS = {
    "smoothing": {
        "laplacian": {
            "iterations": 10,
            "lambda_": 0.5,
            "method": "laplacian"
        },
        "tanh": {
            "iterations": 15,
            "lambda_": 0.3,
            "epsilon": 1e-3,
            "method": "tanh"
        },
        "taubin": {
            "iterations": 15,
            "lambda_": 0.3,
            "mu": 0.5,
            "method": "taubin"
        },
        "adaptive": {
            "iterations": 15,
            "lambda_": 0.3,
            "curvature_threshold": 0.1,
            "method": "adaptive"
        }
    },
    
        "topology": {
            "target_valence": 6,
            "max_iterations": 20,
        },
        "cvt": {
            "num_samples": 900000,
            "max_iterations": 20,
            "tolerance": 1e-6,
            "beta_min": 30.0,  
            "beta_max": 90.0  
        },
        "general": {
            "output_dir": "output/",
            "verbose": True 
        }
    
}
 
def load_params(config_path=None):
    params = DEFAULT_PARAMS.copy() 
    
    if config_path and os.path.exists(config_path): 
        with open(config_path, "r") as f:
            custom_params = json.load(f) 
        for category in custom_params:
            if category in params:
                params[category].update(custom_params[category])
    
    return params 
 
def get_smoothing_params(method="laplacian"):
    """获取平滑算法参数."""
    return load_params()["smoothing"][method]
 
def get_topology_params():
    """获取拓扑优化参数."""
    params = load_params()["topology"].copy()
    params.pop("method", None)
    return params

def get_cvt_params():
    """获取CVT优化参数."""
    return load_params()["cvt"].copy()
 
def get_general_params():
    """获取通用参数."""
    return load_params()["general"]
 
if __name__ == "__main__":
    params = load_params()
    # 拉普拉斯平滑参数 
    laplacian_params = get_smoothing_params("laplacian")
    print(f"拉普拉斯平滑迭代次数：{laplacian_params['iterations']}")
    # 拓扑优化参数 
    topology_params = get_topology_params()
    print(f"目标度数：{topology_params['target_valence']}")
    custom_params = {
        "smoothing": {
            "taubin": {
                "lambda_": 0.6,
                "mu": 0.4 
            }
        },
        "general": {
            "output_dir": "custom_output/"
        }
    }
    with open("config.json",  "w") as f:
        json.dump(custom_params,  f)
    
    params = load_params("config.json") 
    print(f"自定义 TAubin 平滑 lambda: {params['smoothing']['taubin']['lambda_']}")