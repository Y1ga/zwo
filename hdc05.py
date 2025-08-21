import cv2
import numpy as np
import glob
import os
from scipy.optimize import least_squares
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

class BSplineDistortionModel:
    def __init__(self, grid_size=(10, 8), degree=3):
        """
        B样条畸变模型
        
        Args:
            grid_size: 控制点网格大小 (nx, ny)
            degree: B样条的度数
        """
        self.grid_size = grid_size
        self.degree = degree
        self.nx, self.ny = grid_size
        
        # 创建控制点网格
        self.create_control_grid()
        
        # B样条参数（控制点的偏移）
        self.control_points_x = np.zeros((self.ny, self.nx))  # x方向偏移
        self.control_points_y = np.zeros((self.ny, self.nx))  # y方向偏移
        
    def create_control_grid(self):
        """创建控制点网格和节点向量"""
        # 节点向量（均匀分布）
        self.knots_x = np.linspace(0, 1, self.nx + self.degree + 1)
        self.knots_y = np.linspace(0, 1, self.ny + self.degree + 1)
        
        # 控制点在归一化坐标中的位置
        self.grid_x = np.linspace(0, 1, self.nx)
        self.grid_y = np.linspace(0, 1, self.ny)
        
    def normalize_coordinates(self, x, y, img_size):
        """将像素坐标归一化到[0,1]"""
        width, height = img_size
        x_norm = np.asarray(x) / (width - 1)
        y_norm = np.asarray(y) / (height - 1)
        return x_norm, y_norm
    
    def denormalize_coordinates(self, x_norm, y_norm, img_size):
        """将归一化坐标转换回像素坐标"""
        width, height = img_size
        x = np.asarray(x_norm) * (width - 1)
        y = np.asarray(y_norm) * (height - 1)
        return x, y
    
    def evaluate_bspline_2d(self, x_norm, y_norm):
        """
        评估2D B样条在给定点的值
        
        Args:
            x_norm, y_norm: 归一化坐标 (可以是标量或数组)
            
        Returns:
            displacement_x, displacement_y: 在该点的位移
        """
        # 确保坐标在有效范围内
        x_norm = np.clip(x_norm, 0, 1)
        y_norm = np.clip(y_norm, 0, 1)
        
        # 初始化位移数组
        if np.isscalar(x_norm):
            displacement_x = 0.0
            displacement_y = 0.0
        else:
            displacement_x = np.zeros_like(x_norm)
            displacement_y = np.zeros_like(y_norm)
        
        # 计算B样条基函数
        for i in range(self.nx):
            for j in range(self.ny):
                # 计算B样条基函数值
                basis_x = self.bspline_basis(x_norm, i, self.degree, self.knots_x)
                basis_y = self.bspline_basis(y_norm, j, self.degree, self.knots_y)
                
                # 处理标量和数组情况
                if np.isscalar(basis_x) and np.isscalar(basis_y):
                    basis = basis_x * basis_y
                else:
                    basis = np.multiply(basis_x, basis_y)
                
                # 累加控制点的贡献
                displacement_x += basis * self.control_points_x[j, i]
                displacement_y += basis * self.control_points_y[j, i]
        
        return displacement_x, displacement_y
    
    def bspline_basis(self, t, i, p, knots):
        """
        计算B样条基函数N_{i,p}(t)
        
        Args:
            t: 参数值 (可以是标量或数组)
            i: 控制点索引
            p: 度数
            knots: 节点向量
            
        Returns:
            基函数值
        """
        # 处理数组输入
        if np.isscalar(t):
            if p == 0:
                return 1.0 if knots[i] <= t < knots[i+1] else 0.0
        else:
            # 对于数组输入，逐元素计算
            if p == 0:
                result = np.zeros_like(t)
                mask = (knots[i] <= t) & (t < knots[i+1])
                result[mask] = 1.0
                return result
        
        # 递归计算
        left_term = 0
        right_term = 0
        
        # 左侧项
        if knots[i+p] != knots[i]:
            left_term = (t - knots[i]) / (knots[i+p] - knots[i]) * self.bspline_basis(t, i, p-1, knots)
        
        # 右侧项
        if knots[i+p+1] != knots[i+1]:
            right_term = (knots[i+p+1] - t) / (knots[i+p+1] - knots[i+1]) * self.bspline_basis(t, i+1, p-1, knots)
        
        return left_term + right_term
    
    def apply_distortion(self, x, y, img_size):
        """
        应用B样条畸变校正
        
        Args:
            x, y: 原始坐标 (可以是标量或数组)
            img_size: 图像尺寸 (width, height)
            
        Returns:
            x_corrected, y_corrected: 校正后的坐标
        """
        # 处理标量输入
        is_scalar = np.isscalar(x) and np.isscalar(y)
        
        if is_scalar:
            x = np.array([x])
            y = np.array([y])
        
        # 归一化坐标
        x_norm, y_norm = self.normalize_coordinates(x, y, img_size)
        
        # 评估B样条位移
        dx, dy = self.evaluate_bspline_2d(x_norm, y_norm)
        
        # 应用位移（在归一化坐标系中）
        x_norm_corrected = x_norm + dx
        y_norm_corrected = y_norm + dy
        
        # 转换回像素坐标
        x_corrected, y_corrected = self.denormalize_coordinates(
            x_norm_corrected, y_norm_corrected, img_size
        )
        
        # 如果输入是标量，返回标量
        if is_scalar:
            return x_corrected[0], y_corrected[0]
        else:
            return x_corrected, y_corrected
    
    def get_parameters(self):
        """获取B样条参数向量"""
        return np.concatenate([self.control_points_x.flatten(), 
                              self.control_points_y.flatten()])
    
    def set_parameters(self, params):
        """设置B样条参数"""
        n_params = self.nx * self.ny
        self.control_points_x = params[:n_params].reshape(self.ny, self.nx)
        self.control_points_y = params[n_params:].reshape(self.ny, self.nx)
    
    def visualize_distortion_field(self, img_size, save_path=None):
        """可视化畸变场"""
        width, height = img_size
        
        # 创建网格点
        x_grid = np.linspace(0, width-1, 20)
        y_grid = np.linspace(0, height-1, 20)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # 逐点计算畸变（避免数组处理问题）
        DX = np.zeros_like(X)
        DY = np.zeros_like(Y)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x_corrected, y_corrected = self.apply_distortion(X[i, j], Y[i, j], img_size)
                DX[i, j] = x_corrected - X[i, j]
                DY[i, j] = y_corrected - Y[i, j]
        
        # 绘制畸变场
        plt.figure(figsize=(12, 8))
        plt.quiver(X, Y, DX, DY, scale=10, scale_units='xy', angles='xy', alpha=0.7)
        plt.title('B-spline Distortion Field')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # 添加颜色条显示位移大小
        magnitude = np.sqrt(DX**2 + DY**2)
        plt.scatter(X, Y, c=magnitude, s=20, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Distortion Magnitude (pixels)')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"畸变场可视化已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()  # 避免内存泄漏

class ManualCameraCalibration:
    def __init__(self, use_bspline=False, bspline_grid_size=(8, 6)):
        """手写相机标定类"""
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.use_bspline = use_bspline
        
        # B样条畸变模型
        if use_bspline:
            self.bspline_model = BSplineDistortionModel(grid_size=bspline_grid_size)
        else:
            self.bspline_model = None
            
    def project_points_manual(self, object_points, rvec, tvec, camera_matrix, 
                             dist_coeffs=None, img_size=None):
        """
        手动实现3D点到2D图像点的投影（支持B样条畸变）
        """
        # 确保输入格式正确
        if object_points.ndim == 2 and object_points.shape[1] == 3:
            points_3d = object_points.T  # (3, N)
        else:
            points_3d = object_points.reshape(-1, 3).T
            
        rvec = rvec.flatten()
        tvec = tvec.flatten()
        
        # 1. 旋转向量转旋转矩阵 (Rodrigues公式)
        R = self.rodrigues_manual(rvec)
        
        # 2. 3D点变换到相机坐标系
        camera_points = R @ points_3d + tvec.reshape(3, 1)  # (3, N)
        
        # 3. 透视投影到归一化图像平面
        x = camera_points[0] / camera_points[2]
        y = camera_points[1] / camera_points[2]
        
        # 4. 畸变校正
        if self.use_bspline and self.bspline_model is not None and img_size is not None:
            # B样条畸变校正
            # 先应用内参得到像素坐标
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
            
            u_undist = fx * x + cx
            v_undist = fy * y + cy
            
            # 应用B样条畸变
            u, v = self.bspline_model.apply_distortion(u_undist, v_undist, img_size)
            
            return np.column_stack([u, v])
            
        elif dist_coeffs is not None and len(dist_coeffs) >= 4:
            # 传统多项式畸变
            x, y = self.apply_distortion(x, y, dist_coeffs)
        
        # 5. 应用相机内参
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        u = fx * x + cx
        v = fy * y + cy
        
        return np.column_stack([u, v])
    
    def apply_distortion(self, x, y, dist_coeffs):
        """应用传统径向和切向畸变"""
        k1, k2, p1, p2 = dist_coeffs[:4]
        k3 = dist_coeffs[4] if len(dist_coeffs) > 4 else 0
        
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2**3
        
        # 径向畸变
        radial_factor = 1 + k1 * r2 + k2 * r4 + k3 * r6
        
        # 切向畸变
        dx_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        dy_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
        
        x_dist = x * radial_factor + dx_tangential
        y_dist = y * radial_factor + dy_tangential
        
        return x_dist, y_dist
    
    def rodrigues_manual(self, rvec):
        """手动实现Rodrigues公式：旋转向量转旋转矩阵"""
        theta = np.linalg.norm(rvec)
        
        if theta < 1e-6:
            return np.eye(3)
        
        # 单位旋转轴
        k = rvec / theta
        
        # 反对称矩阵
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        
        # Rodrigues公式
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        
        return R
    
    def estimate_homography_dlt(self, world_points, image_points):
        """使用DLT估计单应性矩阵（改进版）"""
        n_points = len(world_points)
        
        if n_points < 4:
            raise ValueError(f"DLT需要至少4个点，但只有{n_points}个")
        
        # 数据归一化以提高数值稳定性
        # 归一化图像点
        img_mean = np.mean(image_points, axis=0)
        img_std = np.std(image_points, axis=0)
        img_std[img_std < 1e-8] = 1.0  # 防止除零
        
        img_points_norm = (image_points - img_mean) / img_std
        
        # 归一化世界点
        world_mean = np.mean(world_points[:, :2], axis=0)
        world_std = np.std(world_points[:, :2], axis=0)
        world_std[world_std < 1e-8] = 1.0
        
        world_points_norm = (world_points[:, :2] - world_mean) / world_std
        
        # 构建系数矩阵A (2N x 9)
        A = np.zeros((2 * n_points, 9))
        
        for i in range(n_points):
            X, Y = world_points_norm[i, 0], world_points_norm[i, 1]  # Z = 0
            u, v = img_points_norm[i, 0], img_points_norm[i, 1]
            
            A[2*i] = [-X, -Y, -1, 0, 0, 0, u*X, u*Y, u]
            A[2*i+1] = [0, 0, 0, -X, -Y, -1, v*X, v*Y, v]
        
        # SVD求解Ah = 0
        try:
            U, s, Vt = np.linalg.svd(A)
            
            # 检查SVD的数值稳定性
            if len(s) > 0 and s[0] / s[-1] > 1e12:
                print(f"警告: DLT系数矩阵条件数过大: {s[0]/s[-1]:.2e}")
            
            h = Vt[-1]
            H_norm = h.reshape(3, 3)
            
            # 反归一化
            # T_img^(-1) * H_norm * T_world
            T_img = np.array([
                [img_std[0], 0, img_mean[0]],
                [0, img_std[1], img_mean[1]],
                [0, 0, 1]
            ])
            
            T_world_inv = np.array([
                [1/world_std[0], 0, -world_mean[0]/world_std[0]],
                [0, 1/world_std[1], -world_mean[1]/world_std[1]],
                [0, 0, 1]
            ])
            
            H = T_img @ H_norm @ T_world_inv
            
            # 归一化使H[2,2] = 1
            if abs(H[2, 2]) > 1e-8:
                H = H / H[2, 2]
            else:
                print("警告: H[2,2]接近零，单应性矩阵可能无效")
            
            return H
            
        except np.linalg.LinAlgError as e:
            print(f"DLT SVD求解失败: {e}")
            # 降级到简单方法
            return self.estimate_homography_simple(world_points, image_points)
    
    def estimate_homography_simple(self, world_points, image_points):
        """简单的单应性矩阵估计（降级方案）"""
        # 使用OpenCV作为备选方案
        try:
            world_2d = world_points[:, :2].astype(np.float32)
            img_2d = image_points.astype(np.float32)
            H, _ = cv2.findHomography(world_2d, img_2d, cv2.RANSAC)
            if H is not None:
                return H
        except:
            pass
        
        # 如果OpenCV也失败，返回单位矩阵
        print("警告: 无法估计单应性矩阵，使用单位矩阵")
        return np.eye(3)
    
    def estimate_camera_matrix_zhang(self, homographies):
        """使用张正友标定法估计相机内参矩阵（改进版）"""
        n_views = len(homographies)
        
        print(f"使用 {n_views} 个单应性矩阵估计内参...")
        
        # 构建约束方程组
        V = []
        
        for idx, H in enumerate(homographies):
            # 归一化单应性矩阵，确保数值稳定
            H = H / H[2, 2]
            
            def v_ij(i, j):
                hi1, hi2, hi3 = H[i, :]
                hj1, hj2, hj3 = H[j, :]
                
                return np.array([
                    hi1 * hj1,
                    hi1 * hj2 + hi2 * hj1,
                    hi2 * hj2,
                    hi3 * hj1 + hi1 * hj3,
                    hi3 * hj2 + hi2 * hj3,
                    hi3 * hj3
                ])
            
            # 两个约束：h1^T * ω * h2 = 0 和 h1^T * ω * h1 = h2^T * ω * h2
            v12 = v_ij(0, 1)  # h1^T * ω * h2 = 0
            v11_v22 = v_ij(0, 0) - v_ij(1, 1)  # h1^T * ω * h1 - h2^T * ω * h2 = 0
            
            V.append(v12)
            V.append(v11_v22)
            
            print(f"  视图 {idx+1}: H条件数 = {np.linalg.cond(H):.2e}")
        
        V = np.array(V)
        print(f"约束矩阵V形状: {V.shape}, 条件数: {np.linalg.cond(V):.2e}")
        
        # 检查是否有足够的约束
        if V.shape[0] < 6:
            raise ValueError(f"约束不足: 需要至少6个约束，但只有{V.shape[0]}个")
        
        # SVD求解，添加数值稳定性检查
        try:
            U, s, Vt = np.linalg.svd(V)
            print(f"SVD奇异值: {s}")
            
            # 检查奇异值的条件数
            if s[0] / s[-1] > 1e12:
                print("警告: 约束矩阵条件数过大，可能导致数值不稳定")
            
            b = Vt[-1]
            print(f"解向量b: {b}")
            
        except np.linalg.LinAlgError as e:
            print(f"SVD求解失败: {e}")
            # 降级方案：使用简化的内参估计
            return self.estimate_camera_matrix_simple(homographies)
        
        # 从b恢复内参矩阵，添加数值检查
        B11, B12, B22, B13, B23, B33 = b
        
        print(f"B矩阵元素: B11={B11:.6f}, B12={B12:.6f}, B22={B22:.6f}")
        print(f"            B13={B13:.6f}, B23={B23:.6f}, B33={B33:.6f}")
        
        # 检查分母是否接近零
        denominator = B11 * B22 - B12**2
        if abs(denominator) < 1e-10:
            print("警告: 分母接近零，使用简化方法")
            return self.estimate_camera_matrix_simple(homographies)
        
        # 计算内参
        try:
            cy = (B12 * B13 - B11 * B23) / denominator
            lam = B33 - (B13**2 + cy * (B12 * B13 - B11 * B23)) / B11
            
            print(f"中间计算: cy={cy:.2f}, lambda={lam:.6f}")
            
            # 检查lambda是否为正
            if lam <= 0:
                print(f"警告: lambda={lam} <= 0, 使用简化方法")
                return self.estimate_camera_matrix_simple(homographies)
            
            if B11 <= 0:
                print(f"警告: B11={B11} <= 0, 使用简化方法")
                return self.estimate_camera_matrix_simple(homographies)
            
            fx = np.sqrt(lam / B11)
            
            # 检查fy的分母
            fy_denominator = lam * B11 / denominator
            if fy_denominator <= 0:
                print(f"警告: fy分母={fy_denominator} <= 0, 使用简化方法")
                return self.estimate_camera_matrix_simple(homographies)
            
            fy = np.sqrt(fy_denominator)
            skew = -B12 * fx**2 * fy / lam
            cx = skew * cy / fy - B13 * fx**2 / lam
            
            camera_matrix = np.array([
                [fx, skew, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            # 检查结果的合理性
            if not np.all(np.isfinite(camera_matrix)):
                print("警告: 内参包含无穷大或NaN，使用简化方法")
                return self.estimate_camera_matrix_simple(homographies)
            
            if fx < 100 or fx > 2000 or fy < 100 or fy > 2000:
                print(f"警告: 焦距不合理 fx={fx:.2f}, fy={fy:.2f}，使用简化方法")
                return self.estimate_camera_matrix_simple(homographies)
            
            print(f"计算得到的内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, skew={skew:.6f}")
            return camera_matrix
            
        except (ValueError, RuntimeWarning) as e:
            print(f"内参计算出错: {e}")
            return self.estimate_camera_matrix_simple(homographies)
    
    def estimate_camera_matrix_simple(self, homographies):
        """简化的内参估计方法（降级方案）"""
        print("使用简化方法估计内参...")
        
        # 假设主点在图像中心，无偏斜
        img_center_x, img_center_y = 320, 240  # 对于640x480图像
        
        # 从第一个单应性矩阵估计焦距
        H = homographies[0]
        H = H / H[2, 2]  # 归一化
        
        # 使用单应性矩阵的性质估计焦距
        # ||h1|| ≈ ||h2|| ≈ 1/f (在归一化坐标下)
        h1_norm = np.linalg.norm(H[:2, 0])
        h2_norm = np.linalg.norm(H[:2, 1])
        
        # 估计焦距
        f_estimate = (h1_norm + h2_norm) / 2
        fx = fy = 1.0 / f_estimate if f_estimate > 0 else 800.0  # 默认焦距
        
        # 调整到合理范围
        fx = max(400, min(1200, fx * 400))  # 假设焦距在400-1200范围
        fy = fx  # 假设fx = fy
        
        camera_matrix = np.array([
            [fx, 0, img_center_x],
            [0, fy, img_center_y],
            [0, 0, 1]
        ])
        
        print(f"简化方法得到的内参: fx={fx:.2f}, fy={fy:.2f}, cx={img_center_x}, cy={img_center_y}")
        return camera_matrix
    
    def decompose_homography(self, H, camera_matrix):
        """从单应性矩阵分解出旋转和平移"""
        # 归一化单应性矩阵
        H = H / np.linalg.norm(H[:, 0])
        
        # K^(-1) * H
        K_inv = np.linalg.inv(camera_matrix)
        H_norm = K_inv @ H
        
        # 提取旋转和平移
        h1 = H_norm[:, 0]
        h2 = H_norm[:, 1]
        h3 = H_norm[:, 2]
        
        # 旋转矩阵的第三列
        r1 = h1 / np.linalg.norm(h1)
        r2 = h2 / np.linalg.norm(h1)
        r3 = np.cross(r1, r2)
        
        # 构建旋转矩阵并正交化
        R = np.column_stack([r1, r2, r3])
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        # 确保det(R) = 1
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
            
        # 平移向量
        t = h3 / np.linalg.norm(h1)
        
        # 旋转矩阵转旋转向量
        rvec = self.rotation_matrix_to_rodrigues(R)
        
        return rvec, t
    
    def rotation_matrix_to_rodrigues(self, R):
        """旋转矩阵转旋转向量"""
        trace = np.trace(R)
        
        if abs(trace - 3) < 1e-6:
            return np.zeros(3)
        
        theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if abs(theta) < 1e-6:
            return np.zeros(3)
        
        if abs(theta - np.pi) < 1e-6:
            # 特殊情况：theta ≈ π
            diag = np.diag(R)
            k = np.argmax(diag)
            
            v = np.zeros(3)
            v[k] = np.sqrt((R[k, k] + 1) / 2)
            
            for i in range(3):
                if i != k:
                    v[i] = R[k, i] / (2 * v[k])
            
            rvec = theta * v / np.linalg.norm(v)
        else:
            # 一般情况
            v = np.array([R[2, 1] - R[1, 2], 
                         R[0, 2] - R[2, 0], 
                         R[1, 0] - R[0, 1]])
            v = v / (2 * np.sin(theta))
            rvec = theta * v
        
        return rvec

    def calibrate_camera_manual(self, object_points_list, image_points_list, image_size):
        """手动实现相机标定（改进版，支持B样条）"""
        n_views = len(object_points_list)
        
        print(f"开始手动标定，共 {n_views} 个视图...")
        if self.use_bspline:
            print("使用B样条畸变模型")
        else:
            print("使用传统多项式畸变模型")
        
        # 检查输入数据
        for i, (obj_pts, img_pts) in enumerate(zip(object_points_list, image_points_list)):
            if len(obj_pts) != len(img_pts):
                raise ValueError(f"视图{i+1}: 3D点和2D点数量不匹配")
            if len(obj_pts) < 4:
                raise ValueError(f"视图{i+1}: 点数量不足，需要至少4个点")
            
            print(f"  视图{i+1}: {len(obj_pts)} 个点")
        
        # 步骤1: 计算每个视图的单应性矩阵
        print("1. 计算单应性矩阵...")
        homographies = []
        valid_views = []
        
        for i in range(n_views):
            try:
                H = self.estimate_homography_dlt(object_points_list[i], image_points_list[i])
                
                # 检查单应性矩阵的有效性
                if np.all(np.isfinite(H)) and abs(np.linalg.det(H)) > 1e-6:
                    homographies.append(H)
                    valid_views.append(i)
                    print(f"  视图{i+1}: 单应性矩阵计算成功")
                else:
                    print(f"  视图{i+1}: 单应性矩阵无效，跳过")
                    
            except Exception as e:
                print(f"  视图{i+1}: 单应性矩阵计算失败: {e}")
        
        if len(homographies) < 3:
            raise ValueError(f"有效视图数量不足: {len(homographies)}/3")
        
        print(f"成功计算 {len(homographies)} 个有效单应性矩阵")
        
        # 步骤2: 使用张正友方法估计内参
        print("2. 估计相机内参矩阵...")
        try:
            camera_matrix = self.estimate_camera_matrix_zhang(homographies)
        except Exception as e:
            print(f"张正友方法失败: {e}")
            # 使用简化方法
            camera_matrix = self.estimate_camera_matrix_simple(homographies)
        
        print(f"初始内参矩阵:\n{camera_matrix}")
        
        # 步骤3: 计算每个视图的外参
        print("3. 计算外参...")
        rvecs = []
        tvecs = []
        
        for i, view_idx in enumerate(valid_views):
            try:
                rvec, tvec = self.decompose_homography(homographies[i], camera_matrix)
                rvecs.append(rvec)
                tvecs.append(tvec)
                print(f"  视图{view_idx+1}: 外参计算成功")
            except Exception as e:
                print(f"  视图{view_idx+1}: 外参计算失败: {e}")
                # 使用默认值
                rvecs.append(np.zeros(3))
                tvecs.append(np.array([0, 0, 500]))
        
        # 步骤4: 非线性优化精化所有参数
        print("4. 非线性优化...")
        
        try:
            # 只使用有效视图的数据
            valid_object_points = [object_points_list[i] for i in valid_views]
            valid_image_points = [image_points_list[i] for i in valid_views]
            
            if self.use_bspline:
                # 使用B样条畸变的优化
                params = self.pack_parameters_bspline(camera_matrix, rvecs, tvecs)
                
                # 计算初始重投影误差
                initial_residuals = self.reprojection_residuals_bspline(
                    params, valid_object_points, valid_image_points, image_size
                )
                initial_rms = np.sqrt(np.mean(initial_residuals**2))
                print(f"  初始RMS误差: {initial_rms:.4f} pixels")
                
                # 优化
                result = least_squares(
                    self.reprojection_residuals_bspline,
                    params,
                    args=(valid_object_points, valid_image_points, image_size),
                    method='lm',
                    max_nfev=2000,
                    ftol=1e-8,
                    xtol=1e-8
                )
                
                if result.success:
                    print(f"  B样条优化成功: {result.message}")
                else:
                    print(f"  B样条优化警告: {result.message}")
                
                # 解包优化结果
                camera_matrix_opt, rvecs_opt, tvecs_opt = self.unpack_parameters_bspline(
                    result.x, len(valid_views)
                )
                dist_coeffs_opt = None  # B样条模型不使用传统畸变系数
                
            else:
                # 传统优化
                params = self.pack_parameters(camera_matrix, np.zeros(5), rvecs, tvecs)
                
                # 计算初始重投影误差
                initial_residuals = self.reprojection_residuals(params, valid_object_points, valid_image_points)
                initial_rms = np.sqrt(np.mean(initial_residuals**2))
                print(f"  初始RMS误差: {initial_rms:.4f} pixels")
                
                # 优化
                result = least_squares(
                    self.reprojection_residuals,
                    params,
                    args=(valid_object_points, valid_image_points),
                    method='lm',
                    max_nfev=1000,
                    ftol=1e-8,
                    xtol=1e-8
                )
                
                if result.success:
                    print(f"  传统优化成功: {result.message}")
                else:
                    print(f"  传统优化警告: {result.message}")
                
                # 解包优化结果
                camera_matrix_opt, dist_coeffs_opt, rvecs_opt, tvecs_opt = self.unpack_parameters(
                    result.x, len(valid_views)
                )
            
        except Exception as e:
            print(f"优化失败: {e}")
            # 使用初始估计
            camera_matrix_opt = camera_matrix
            dist_coeffs_opt = np.zeros(5) if not self.use_bspline else None
            rvecs_opt = rvecs
            tvecs_opt = tvecs
        
        # 计算最终重投影误差
        total_error = 0
        total_points = 0
        
        valid_object_points = [object_points_list[i] for i in valid_views]
        valid_image_points = [image_points_list[i] for i in valid_views]
        
        for i in range(len(valid_views)):
            try:
                projected = self.project_points_manual(
                    valid_object_points[i], rvecs_opt[i], tvecs_opt[i], 
                    camera_matrix_opt, dist_coeffs_opt, image_size
                )
                
                if projected is not None and len(projected) == len(valid_image_points[i]):
                    error = np.linalg.norm(valid_image_points[i] - projected, axis=1)
                    total_error += np.sum(error**2)
                    total_points += len(error)
                    
            except Exception as e:
                print(f"计算重投影误差时出错(视图{i+1}): {e}")
        
        if total_points > 0:
            rms_error = np.sqrt(total_error / total_points)
        else:
            rms_error = float('inf')
            print("警告: 无法计算重投影误差")
        
        # 保存结果
        self.camera_matrix = camera_matrix_opt
        self.dist_coeffs = dist_coeffs_opt
        self.rvecs = rvecs_opt
        self.tvecs = tvecs_opt
        
        model_type = "B样条" if self.use_bspline else "传统多项式"
        print(f"{model_type}标定完成! RMS重投影误差: {rms_error:.4f} pixels")
        
        return rms_error, camera_matrix_opt, dist_coeffs_opt, rvecs_opt, tvecs_opt
    
    def pack_parameters(self, camera_matrix, dist_coeffs, rvecs, tvecs):
        """打包参数向量"""
        params = []
        
        # 内参: fx, fy, cx, cy
        params.extend([camera_matrix[0, 0], camera_matrix[1, 1], 
                      camera_matrix[0, 2], camera_matrix[1, 2]])
        
        # 畸变系数: k1, k2, p1, p2, k3
        params.extend(dist_coeffs[:5])
        
        # 外参: rvec + tvec for each view
        for rvec, tvec in zip(rvecs, tvecs):
            params.extend(rvec)
            params.extend(tvec)
        
        return np.array(params)
    
    def unpack_parameters(self, params, n_views):
        """解包参数向量"""
        # 内参
        fx, fy, cx, cy = params[:4]
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # 畸变系数
        dist_coeffs = params[4:9]
        
        # 外参
        rvecs = []
        tvecs = []
        for i in range(n_views):
            start_idx = 9 + i * 6
            rvec = params[start_idx:start_idx+3]
            tvec = params[start_idx+3:start_idx+6]
            rvecs.append(rvec)
            tvecs.append(tvec)
        
        return camera_matrix, dist_coeffs, rvecs, tvecs
    
    def reprojection_residuals(self, params, object_points_list, image_points_list):
        """计算重投影残差"""
        n_views = len(object_points_list)
        
        # 解包参数
        camera_matrix, dist_coeffs, rvecs, tvecs = self.unpack_parameters(params, n_views)
        
        residuals = []
        
        for i in range(n_views):
            # 投影3D点
            projected = self.project_points_manual(
                object_points_list[i], rvecs[i], tvecs[i], 
                camera_matrix, dist_coeffs
            )
            
            # 计算残差
            diff = image_points_list[i] - projected
            residuals.extend(diff.flatten())
        
        return np.array(residuals)
    
    def pack_parameters_bspline(self, camera_matrix, rvecs, tvecs):
        """打包B样条标定的参数向量"""
        params = []
        
        # 内参: fx, fy, cx, cy
        params.extend([camera_matrix[0, 0], camera_matrix[1, 1], 
                      camera_matrix[0, 2], camera_matrix[1, 2]])
        
        # B样条控制点
        bspline_params = self.bspline_model.get_parameters()
        params.extend(bspline_params)
        
        # 外参: rvec + tvec for each view
        for rvec, tvec in zip(rvecs, tvecs):
            params.extend(rvec)
            params.extend(tvec)
        
        return np.array(params)
    
    def unpack_parameters_bspline(self, params, n_views):
        """解包B样条标定的参数向量"""
        # 内参
        fx, fy, cx, cy = params[:4]
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # B样条参数
        n_bspline = self.bspline_model.nx * self.bspline_model.ny * 2
        bspline_params = params[4:4+n_bspline]
        self.bspline_model.set_parameters(bspline_params)
        
        # 外参
        rvecs = []
        tvecs = []
        start_idx = 4 + n_bspline
        for i in range(n_views):
            rvec = params[start_idx + i*6:start_idx + i*6 + 3]
            tvec = params[start_idx + i*6 + 3:start_idx + i*6 + 6]
            rvecs.append(rvec)
            tvecs.append(tvec)
        
        return camera_matrix, rvecs, tvecs
    
    def reprojection_residuals_bspline(self, params, object_points_list, image_points_list, image_size):
        """计算B样条模型的重投影残差"""
        n_views = len(object_points_list)
        
        # 解包参数
        camera_matrix, rvecs, tvecs = self.unpack_parameters_bspline(params, n_views)
        
        residuals = []
        
        for i in range(n_views):
            # 投影3D点
            projected = self.project_points_manual(
                object_points_list[i], rvecs[i], tvecs[i], 
                camera_matrix, None, image_size  # B样条不需要传统畸变系数
            )
            
            # 计算残差
            diff = image_points_list[i] - projected
            residuals.extend(diff.flatten())
        
        return np.array(residuals)
    
    def save_bspline_model(self, filename='bspline_distortion.npz'):
        """保存B样条畸变模型"""
        if self.bspline_model is not None:
            np.savez(filename,
                     grid_size=self.bspline_model.grid_size,
                     degree=self.bspline_model.degree,
                     control_points_x=self.bspline_model.control_points_x,
                     control_points_y=self.bspline_model.control_points_y,
                     knots_x=self.bspline_model.knots_x,
                     knots_y=self.bspline_model.knots_y)
            print(f"B样条模型已保存到: {filename}")
    
    def load_bspline_model(self, filename='bspline_distortion.npz'):
        """加载B样条畸变模型"""
        data = np.load(filename)
        grid_size = tuple(data['grid_size'])
        degree = int(data['degree'])
        
        self.bspline_model = BSplineDistortionModel(grid_size, degree)
        self.bspline_model.control_points_x = data['control_points_x']
        self.bspline_model.control_points_y = data['control_points_y']
        self.bspline_model.knots_x = data['knots_x']
        self.bspline_model.knots_y = data['knots_y']
        
        print(f"B样条模型已从 {filename} 加载")

class StereoCalibrationWithManual:
    def __init__(self, chessboard_size=(9, 6), square_size=25.0, use_bspline=False):
        """带手写标定功能的双目标定类"""
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.use_bspline = use_bspline
        
        # 准备棋盘格角点的世界坐标
        self.prepare_object_points()
        
        # 存储检测到的角点
        self.object_points = []
        self.img_points_left = []
        self.img_points_right = []
        
        # 手写标定器
        self.manual_calib_left = ManualCameraCalibration(use_bspline=use_bspline)
        self.manual_calib_right = ManualCameraCalibration(use_bspline=use_bspline)
        
        # 传统标定器用于对比
        self.manual_calib_left_traditional = ManualCameraCalibration(use_bspline=False)
        self.manual_calib_right_traditional = ManualCameraCalibration(use_bspline=False)
        
    def prepare_object_points(self):
        """准备棋盘格的3D世界坐标"""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 
                               0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        self.objp = objp
        
    def find_chessboard_corners(self, left_images_path, right_images_path):
        """在左右相机图像中寻找棋盘格角点"""
        left_images = sorted(glob.glob(left_images_path))
        right_images = sorted(glob.glob(right_images_path))
        
        if len(left_images) != len(right_images):
            raise ValueError("左右相机图像数量不匹配")
        
        print(f"找到 {len(left_images)} 对立体图像")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        valid_pairs = 0
        
        for left_path, right_path in zip(left_images, right_images):
            # 读取图像
            img_left = cv2.imread(left_path)
            img_right = cv2.imread(right_path)
            
            if img_left is None or img_right is None:
                print(f"无法读取图像: {left_path} 或 {right_path}")
                continue
                
            # 转换为灰度图
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            
            # 寻找棋盘格角点
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.chessboard_size, None)
            
            # 如果两个相机都找到了角点
            if ret_left and ret_right:
                # 精细化角点位置
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
                
                # 存储角点
                self.object_points.append(self.objp)
                self.img_points_left.append(corners_left.reshape(-1, 2))
                self.img_points_right.append(corners_right.reshape(-1, 2))
                
                valid_pairs += 1
                print(f"图像对 {valid_pairs}: {os.path.basename(left_path)} - 角点检测成功")
            else:
                print(f"图像对: {os.path.basename(left_path)} - 角点检测失败")
        
        print(f"成功检测到 {valid_pairs} 对有效图像")
        
        if valid_pairs < 10:
            print("警告: 有效图像对数量较少，标定精度可能不高")
            
    def calibrate_cameras_manual_with_bspline_comparison(self, img_size):
        """使用手写算法分别标定左右相机，并比较B样条与传统方法"""
        if len(self.object_points) == 0:
            raise ValueError("没有找到有效的角点，请先运行 find_chessboard_corners")
        
        results = {}
        
        # 传统多项式畸变标定
        print("\n" + "="*60)
        print("传统多项式畸变标定")
        print("="*60)
        
        print("\n=== 手写算法标定左相机（传统畸变） ===")
        ret_left_trad, mtx_left_trad, dist_left_trad, rvecs_left_trad, tvecs_left_trad = \
            self.manual_calib_left_traditional.calibrate_camera_manual(
                self.object_points, self.img_points_left, img_size
            )
        
        print("\n=== 手写算法标定右相机（传统畸变） ===")
        ret_right_trad, mtx_right_trad, dist_right_trad, rvecs_right_trad, tvecs_right_trad = \
            self.manual_calib_right_traditional.calibrate_camera_manual(
                self.object_points, self.img_points_right, img_size
            )
        
        results['traditional'] = {
            'left': {
                'ret': ret_left_trad,
                'camera_matrix': mtx_left_trad,
                'dist_coeffs': dist_left_trad,
                'rvecs': rvecs_left_trad,
                'tvecs': tvecs_left_trad
            },
            'right': {
                'ret': ret_right_trad,
                'camera_matrix': mtx_right_trad,
                'dist_coeffs': dist_right_trad,
                'rvecs': rvecs_right_trad,
                'tvecs': tvecs_right_trad
            }
        }
        
        # B样条畸变标定
        if self.use_bspline:
            print("\n" + "="*60)
            print("B样条畸变标定")
            print("="*60)
            
            print("\n=== 手写算法标定左相机（B样条畸变） ===")
            ret_left_bsp, mtx_left_bsp, dist_left_bsp, rvecs_left_bsp, tvecs_left_bsp = \
                self.manual_calib_left.calibrate_camera_manual(
                    self.object_points, self.img_points_left, img_size
                )
            
            print("\n=== 手写算法标定右相机（B样条畸变） ===")
            ret_right_bsp, mtx_right_bsp, dist_right_bsp, rvecs_right_bsp, tvecs_right_bsp = \
                self.manual_calib_right.calibrate_camera_manual(
                    self.object_points, self.img_points_right, img_size
                )
            
            results['bspline'] = {
                'left': {
                    'ret': ret_left_bsp,
                    'camera_matrix': mtx_left_bsp,
                    'dist_coeffs': dist_left_bsp,  # None for B-spline
                    'rvecs': rvecs_left_bsp,
                    'tvecs': tvecs_left_bsp
                },
                'right': {
                    'ret': ret_right_bsp,
                    'camera_matrix': mtx_right_bsp,
                    'dist_coeffs': dist_right_bsp,  # None for B-spline
                    'rvecs': rvecs_right_bsp,
                    'tvecs': tvecs_right_bsp
                }
            }
            
            # 保存B样条模型
            self.manual_calib_left.save_bspline_model('left_bspline_model.npz')
            self.manual_calib_right.save_bspline_model('right_bspline_model.npz')
            
            # 可视化畸变场
            print("\n生成畸变场可视化...")
            self.manual_calib_left.bspline_model.visualize_distortion_field(
                img_size, 'left_distortion_field.png'
            )
            self.manual_calib_right.bspline_model.visualize_distortion_field(
                img_size, 'right_distortion_field.png'
            )
        
        return results
    
    def calibrate_cameras_opencv(self, img_size):
        """使用OpenCV算法分别标定左右相机"""
        if len(self.object_points) == 0:
            raise ValueError("没有找到有效的角点，请先运行 find_chessboard_corners")
        
        print("\n=== OpenCV算法标定左相机 ===")
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            self.object_points, self.img_points_left, img_size, None, None
        )
        print(f"OpenCV左相机标定误差: {ret_left:.4f} pixels")
        
        print("\n=== OpenCV算法标定右相机 ===")
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            self.object_points, self.img_points_right, img_size, None, None
        )
        print(f"OpenCV右相机标定误差: {ret_right:.4f} pixels")
        
        return {
            'left': {
                'ret': ret_left,
                'camera_matrix': mtx_left,
                'dist_coeffs': dist_left,
                'rvecs': rvecs_left,
                'tvecs': tvecs_left
            },
            'right': {
                'ret': ret_right,
                'camera_matrix': mtx_right,
                'dist_coeffs': dist_right,
                'rvecs': rvecs_right,
                'tvecs': tvecs_right
            }
        }
    
    def compare_distortion_models(self, traditional_results, bspline_results):
        """对比传统畸变模型和B样条畸变模型的结果"""
        print("\n" + "="*80)
        print("畸变模型对比分析")
        print("="*80)
        
        for camera in ['left', 'right']:
            print(f"\n{camera.upper()}相机:")
            print("-" * 40)
            
            trad = traditional_results[camera]
            bspl = bspline_results[camera]
            
            print(f"重投影误差:")
            print(f"  传统多项式畸变: {trad['ret']:.4f} pixels")
            print(f"  B样条畸变:      {bspl['ret']:.4f} pixels")
            
            improvement = (trad['ret'] - bspl['ret']) / trad['ret'] * 100
            if improvement > 0:
                print(f"  B样条改进:      {improvement:.2f}%")
            else:
                print(f"  传统方法更好:   {-improvement:.2f}%")
            
            print(f"\n内参矩阵差异:")
            trad_params = [trad['camera_matrix'][0,0], trad['camera_matrix'][1,1], 
                          trad['camera_matrix'][0,2], trad['camera_matrix'][1,2]]
            bspl_params = [bspl['camera_matrix'][0,0], bspl['camera_matrix'][1,1], 
                          bspl['camera_matrix'][0,2], bspl['camera_matrix'][1,2]]
            
            for i, param_name in enumerate(['fx', 'fy', 'cx', 'cy']):
                diff_pct = abs(trad_params[i] - bspl_params[i]) / trad_params[i] * 100
                print(f"  {param_name}: {diff_pct:.3f}%")
        
        print(f"\n总体分析:")
        avg_trad_error = (traditional_results['left']['ret'] + traditional_results['right']['ret']) / 2
        avg_bspl_error = (bspline_results['left']['ret'] + bspline_results['right']['ret']) / 2
        overall_improvement = (avg_trad_error - avg_bspl_error) / avg_trad_error * 100
        
        print(f"  平均重投影误差:")
        print(f"    传统方法: {avg_trad_error:.4f} pixels")
        print(f"    B样条方法: {avg_bspl_error:.4f} pixels")
        print(f"    整体改进: {overall_improvement:.2f}%")
        
        if overall_improvement > 5:
            print("  结论: B样条畸变模型显著优于传统多项式模型")
        elif overall_improvement > 1:
            print("  结论: B样条畸变模型略优于传统多项式模型")
        elif overall_improvement > -1:
            print("  结论: 两种畸变模型性能相当")
        else:
            print("  结论: 传统多项式模型在此数据集上表现更好")
    
    def compare_calibration_results(self, manual_results, opencv_results):
        """对比手写算法和OpenCV的标定结果"""
        print("\n" + "="*60)
        print("标定结果对比")
        print("="*60)
        
        for camera in ['left', 'right']:
            print(f"\n{camera.upper()}相机:")
            print("-" * 30)
            
            manual = manual_results[camera]
            opencv = opencv_results[camera]
            
            print(f"重投影误差:")
            print(f"  手写算法: {manual['ret']:.4f} pixels")
            print(f"  OpenCV:   {opencv['ret']:.4f} pixels")
            
            print(f"\n内参矩阵 (fx, fy, cx, cy):")
            manual_params = [manual['camera_matrix'][0,0], manual['camera_matrix'][1,1], 
                           manual['camera_matrix'][0,2], manual['camera_matrix'][1,2]]
            opencv_params = [opencv['camera_matrix'][0,0], opencv['camera_matrix'][1,1], 
                           opencv['camera_matrix'][0,2], opencv['camera_matrix'][1,2]]
            
            print(f"  手写算法: [{manual_params[0]:.2f}, {manual_params[1]:.2f}, {manual_params[2]:.2f}, {manual_params[3]:.2f}]")
            print(f"  OpenCV:   [{opencv_params[0]:.2f}, {opencv_params[1]:.2f}, {opencv_params[2]:.2f}, {opencv_params[3]:.2f}]")
            
            if manual['dist_coeffs'] is not None:
                print(f"\n畸变系数 (k1, k2, p1, p2, k3):")
                print(f"  手写算法: {manual['dist_coeffs']}")
                print(f"  OpenCV:   {opencv['dist_coeffs'].flatten()}")
            
            # 计算参数差异百分比
            fx_diff = abs(manual_params[0] - opencv_params[0]) / opencv_params[0] * 100
            fy_diff = abs(manual_params[1] - opencv_params[1]) / opencv_params[1] * 100
            cx_diff = abs(manual_params[2] - opencv_params[2]) / opencv_params[2] * 100
            cy_diff = abs(manual_params[3] - opencv_params[3]) / opencv_params[3] * 100
            
            print(f"\n参数差异百分比:")
            print(f"  fx: {fx_diff:.2f}%")
            print(f"  fy: {fy_diff:.2f}%")
            print(f"  cx: {cx_diff:.2f}%")
            print(f"  cy: {cy_diff:.2f}%")

def main():
    """使用真实图像数据进行双目标定对比，包括B样条畸变"""
    
    # 初始化双目标定（启用B样条）
    stereo_calib = StereoCalibrationWithManual(
        chessboard_size=(9, 6), 
        square_size=25.0,
        use_bspline=True  # 启用B样条
    )
    
    # 设置你的图像路径
    left_images_path = "./images/left/*.jpg"
    right_images_path = "./images/right/*.jpg"
    
    try:
        # 寻找角点
        print("正在检测棋盘格角点...")
        stereo_calib.find_chessboard_corners(left_images_path, right_images_path)
        
        # 获取图像尺寸
        first_img = cv2.imread(glob.glob(left_images_path)[0])
        if first_img is not None:
            img_size = (first_img.shape[1], first_img.shape[0])
            print(f"图像尺寸: {img_size}")
        else:
            raise ValueError("无法读取图像，请检查路径")
        
        # 进行传统方法和B样条方法的对比标定
        print("\n开始畸变模型对比标定...")
        results = stereo_calib.calibrate_cameras_manual_with_bspline_comparison(img_size)
        
        # OpenCV标定用于对比
        print("\n开始OpenCV算法标定...")
        opencv_results = stereo_calib.calibrate_cameras_opencv(img_size)
        
        # 分析对比结果
        if 'bspline' in results:
            stereo_calib.compare_distortion_models(
                results['traditional'], 
                results['bspline']
            )
        
        # 对比手写传统方法与OpenCV
        print("\n" + "="*60)
        print("手写传统方法 vs OpenCV 对比")
        print("="*60)
        stereo_calib.compare_calibration_results(results['traditional'], opencv_results)
        
        # 保存结果
        save_data = {
            'traditional_results': results['traditional'],
            'opencv_results': opencv_results
        }
        
        if 'bspline' in results:
            save_data['bspline_results'] = results['bspline']
        
        np.savez('distortion_model_comparison.npz', **save_data)
        
        print(f"\n所有标定结果已保存!")
        print("文件包括:")
        print("- distortion_model_comparison.npz: 对比结果")
        if 'bspline' in results:
            print("- left_bspline_model.npz: 左相机B样条模型")  
            print("- right_bspline_model.npz: 右相机B样条模型")
            print("- left_distortion_field.png: 左相机畸变场可视化")
            print("- right_distortion_field.png: 右相机畸变场可视化")
        
    except Exception as e:
        print(f"标定过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查:")
        print("1. 图像路径是否正确")
        print("2. 棋盘格尺寸参数是否匹配")
        print("3. 图像中是否包含完整的棋盘格")

if __name__ == "__main__":
    main()