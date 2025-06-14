"""
拓扑优化后处理分析系统 v1.2
功能：三维网格质量分析与可视化 
"""
 
import numpy as np 
from mayavi import mlab 
import stl
from stl import mesh 
 
class TopologyAnalyzer:
    def __init__(self, stl_file, beta_min=30, beta_max=150):
        self.mesh  = mesh.Mesh.from_file(stl_file) 
        self.beta_min  = beta_min 
        self.beta_max  = beta_max 
        self.stats  = {
            "total_triangles": len(self.mesh.vectors), 
            "bad_min_angle": 0,
            "bad_max_angle": 0 
        }
        
    def calculate_triangle_angles(self, vertices):
        """向量法计算三角形三个角度"""
        a = vertices[1]()  - vertices[0]()
        b = vertices[2]()  - vertices[1]()
        c = vertices[0]()  - vertices[2]()
        
        angles = []
        for vec1, vec2 in [(a, b), (b, c), (c, a)]:
            cos_theta = np.dot(vec1,  vec2) / (
                np.linalg.norm(vec1)  * np.linalg.norm(vec2)  + 1e-8
            )
            angles.append(np.degrees(np.arccos(np.clip(cos_theta,  -1.0, 1.0))))
        return angles 
 
    def analyze_quality(self):
        """遍历所有三角形进行质量分析"""
        for tri in self.mesh.vectors: 
            angles = self.calculate_triangle_angles(tri) 
            if min(angles) < self.beta_min: 
                self.stats["bad_min_angle"]  += 1 
            if max(angles) > self.beta_max: 
                self.stats["bad_max_angle"]  += 1 
 
    def visualize(self):
        """三维可视化带质量标记的网格"""
        figure = mlab.figure(size=(1024,  768), bgcolor=(1, 1, 1))
        
        colors = []
        for tri in self.mesh.vectors: 
            angles = self.calculate_triangle_angles(tri) 
            if max(angles) > self.beta_max: 
                colors.append((1,  0.5, 0.5))  # 浅红色 
            elif min(angles) < self.beta_min: 
                colors.append((0.5,  0.5, 1))  # 浅蓝色 
            else:
                colors.append((0.8,  0.8, 0.8))  # 正常灰色 
        
        mlab.triangular_mesh( 
            self.mesh.points[:,  0],
            self.mesh.points[:,  1],
            self.mesh.points[:,  2],
            self.mesh.faces, 
            scalars=np.arange(len(colors)), 
            colormap='viridis',
            opacity=0.8 
        )
        
        # 添加颜色条和统计信息 
        mlab.colorbar(title='Quality  Index', orientation='vertical')
        mlab.text(0.05,  0.95, 
                 f"Min Angle Violations: {self.stats['bad_min_angle']}/{self.stats['total_triangles']}\n" 
                 f"Max Angle Violations: {self.stats['bad_max_angle']}/{self.stats['total_triangles']}", 
                 width=0.4, color=(0, 0, 0))
        
        mlab.show() 
 
if __name__ == "__main__":
    analyzer = TopologyAnalyzer("D:\科研\重大-测试\\b3\output\Botijo_to_5k_input_colored.obj") 
    analyzer.analyze_quality() 
    analyzer.visualize() 