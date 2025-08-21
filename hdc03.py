import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.linalg import svd, norm
import matplotlib.pyplot as plt

class ManualCameraCalibration:
    def __init__(self):
        self.object_points = []  # 3D世界坐标点
        self.image_points = []   # 2D图像坐标点
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = []
        self.tvecs = []
        
    def add_points(self, obj_pts, img_pts):
        """添加一组对应点"""
        self.object_points.append(np.array(obj_pts, dtype=np.float32))
        self.image_points.append(np.array(img_pts, dtype=np.float32))
    
    def estimate_homography(self, obj_pts_2d, img_pts):
        """
        估计单应性矩阵 H
        将3D点投影到Z=0平面，然后计算2D-2D的单应性
        """
        # 构建线性方程组 Ah = 0
        n = len(obj_pts_2d)
        A = np.zeros((2 * n, 9))
        
        for i in range(n):
            x, y = obj_pts_2d[i]
            u, v = img_pts[i]
            
            # 第一行: -x -y -1 0 0 0 ux uy u
            A[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
            # 第二行: 0 0 0 -x -y -1 vx vy v  
            A[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
        
        # SVD求解
        U, S, Vt = svd(A)
        h = Vt[-1, :]  # 最小奇异值对应的向量
        H = h.reshape(3, 3)
        
        return H
    
    def estimate_intrinsic_parameters(self):
        """
        使用张正友标定法估计内参矩阵
        """
        homographies = []
        
        # 为每个棋盘格图像计算单应性矩阵
        for obj_pts, img_pts in zip(self.object_points, self.image_points):
            # 将3D点投影到Z=0平面 (假设棋盘格在Z=0平面)
            obj_pts_2d = obj_pts[:, :2]  # 取x,y坐标
            H = self.estimate_homography(obj_pts_2d, img_pts)
            homographies.append(H)
        
        # 构建约束矩阵求解内参
        V = []
        for H in homographies:
            # 构建v_ij向量
            def v_ij(H, i, j):
                return np.array([
                    H[0,i] * H[0,j],
                    H[0,i] * H[1,j] + H[1,i] * H[0,j],
                    H[1,i] * H[1,j],
                    H[2,i] * H[0,j] + H[0,i] * H[2,j],
                    H[2,i] * H[1,j] + H[1,i] * H[2,j],
                    H[2,i] * H[2,j]
                ])
            
            # 添加两个约束: v_12^T * b = 0 和 (v_11 - v_22)^T * b = 0
            V.append(v_ij(H, 0, 1))  # v_12
            V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))  # v_11 - v_22
        
        V = np.array(V)
        
        # SVD求解b
        U, S, Vt = svd(V)
        b = Vt[-1, :]  # 最小奇异值对应的向量
        
        # 从b向量恢复内参矩阵
        B11, B12, B22, B13, B23, B33 = b
        
        # 计算内参
        v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
        lambda_val = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
        
        alpha = np.sqrt(lambda_val / B11)
        beta = np.sqrt(lambda_val * B11 / (B11 * B22 - B12**2))
        gamma = -B12 * alpha**2 * beta / lambda_val
        u0 = gamma * v0 / beta - B13 * alpha**2 / lambda_val
        
        # 构建内参矩阵
        K = np.array([
            [alpha, gamma, u0],
            [0, beta, v0],
            [0, 0, 1]
        ])
        
        return K
    
    def estimate_extrinsic_parameters(self, K, H):
        """从单应性矩阵和内参估计外参"""
        K_inv = np.linalg.inv(K)
        
        # h1, h2, h3是H的三列
        h1 = H[:, 0]
        h2 = H[:, 1] 
        h3 = H[:, 2]
        
        # 计算lambda
        lambda1 = 1.0 / norm(K_inv @ h1)
        lambda2 = 1.0 / norm(K_inv @ h2)
        lambda_avg = (lambda1 + lambda2) / 2.0
        
        # 计算旋转矩阵的前两列
        r1 = lambda_avg * K_inv @ h1
        r2 = lambda_avg * K_inv @ h2
        r3 = np.cross(r1, r2)  # 第三列通过叉积得到
        
        # 构建旋转矩阵
        R = np.column_stack([r1, r2, r3])
        
        # 确保R是正交矩阵 (通过SVD修正)
        U, S, Vt = svd(R)
        R = U @ Vt
        
        # 计算平移向量
        t = lambda_avg * K_inv @ h3
        
        # 转换为旋转向量
        rvec, _ = cv2.Rodrigues(R)
        
        return rvec.flatten(), t
    
    def project_points(self, obj_pts, rvec, tvec, K, dist_coeffs=None):
        """
        手动投影3D点到2D图像平面
        """
        # 旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 相机坐标系下的点
        pts_cam = (R @ obj_pts.T + tvec.reshape(-1, 1)).T
        
        # 投影到图像平面 (针孔相机模型)
        x = pts_cam[:, 0] / pts_cam[:, 2]
        y = pts_cam[:, 1] / pts_cam[:, 2]
        
        # 应用径向畸变 (如果提供了畸变系数)
        if dist_coeffs is not None:
            k1, k2, p1, p2, k3 = dist_coeffs if len(dist_coeffs) >= 5 else list(dist_coeffs) + [0] * (5 - len(dist_coeffs))
            
            r2 = x**2 + y**2
            r4 = r2**2
            r6 = r2 * r4
            
            # 径向畸变
            radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
            x_distorted = x * radial
            y_distorted = y * radial
            
            # 切向畸变
            x_distorted += 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
            y_distorted += p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
            
            x, y = x_distorted, y_distorted
        
        # 应用内参矩阵
        u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
        v = K[1, 1] * y + K[1, 2]
        
        return np.column_stack([u, v])
    
    def reprojection_error(self, params, obj_points_all, img_points_all, n_images):
        """
        计算重投影误差 (用于非线性优化)
        """
        # 解包参数
        # 内参: fx, fy, cx, cy, skew
        # 畸变: k1, k2, p1, p2, k3
        # 外参: 每个图像6个参数 (3个旋转 + 3个平移)
        
        fx, fy, cx, cy, skew = params[:5]
        k1, k2, p1, p2, k3 = params[5:10]
        
        K = np.array([
            [fx, skew, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        
        errors = []
        
        for i in range(n_images):
            # 提取第i个图像的外参
            start_idx = 10 + i * 6
            rvec = params[start_idx:start_idx+3]
            tvec = params[start_idx+3:start_idx+6]
            
            # 计算重投影点
            proj_pts = self.project_points(obj_points_all[i], rvec, tvec, K, dist_coeffs)
            
            # 计算误差
            error = (proj_pts - img_points_all[i]).flatten()
            errors.extend(error)
        
        return np.array(errors)
    
    def calibrate_camera_manual(self, image_size):
        """
        手动实现相机标定
        """
        if len(self.object_points) == 0:
            raise ValueError("没有标定数据")
        
        print("步骤1: 估计内参矩阵...")
        # 使用张正友方法估计内参
        K_init = self.estimate_intrinsic_parameters()
        print(f"初始内参矩阵:\n{K_init}")
        
        print("步骤2: 估计每个图像的外参...")
        # 估计每个图像的外参
        rvecs_init = []
        tvecs_init = []
        
        for i, (obj_pts, img_pts) in enumerate(zip(self.object_points, self.image_points)):
            # 计算单应性矩阵
            obj_pts_2d = obj_pts[:, :2]
            H = self.estimate_homography(obj_pts_2d, img_pts)
            
            # 估计外参
            rvec, tvec = self.estimate_extrinsic_parameters(K_init, H)
            rvecs_init.append(rvec)
            tvecs_init.append(tvec)
        
        print("步骤3: 非线性优化所有参数...")
        
        # 准备初始参数向量
        # [fx, fy, cx, cy, skew, k1, k2, p1, p2, k3, rvec1, tvec1, rvec2, tvec2, ...]
        initial_params = []
        
        # 内参 (5个)
        initial_params.extend([K_init[0,0], K_init[1,1], K_init[0,2], K_init[1,2], K_init[0,1]])
        
        # 畸变系数 (5个) - 初始为0
        initial_params.extend([0, 0, 0, 0, 0])
        
        # 外参 (每个图像6个)
        for rvec, tvec in zip(rvecs_init, tvecs_init):
            initial_params.extend(rvec)
            initial_params.extend(tvec)
        
        initial_params = np.array(initial_params)
        
        # 非线性优化
        result = least_squares(
            self.reprojection_error,
            initial_params,
            args=(self.object_points, self.image_points, len(self.object_points)),
            verbose=2,
            max_nfev=1000
        )
        
        # 提取优化后的参数
        optimized_params = result.x
        
        fx, fy, cx, cy, skew = optimized_params[:5]
        k1, k2, p1, p2, k3 = optimized_params[5:10]
        
        self.camera_matrix = np.array([
            [fx, skew, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3])
        
        # 提取优化后的外参
        self.rvecs = []
        self.tvecs = []
        for i in range(len(self.object_points)):
            start_idx = 10 + i * 6
            rvec = optimized_params[start_idx:start_idx+3]
            tvec = optimized_params[start_idx+3:start_idx+6]
            self.rvecs.append(rvec)
            self.tvecs.append(tvec)
        
        # 计算重投影误差
        final_errors = self.reprojection_error(
            optimized_params, self.object_points, self.image_points, len(self.object_points)
        )
        rms_error = np.sqrt(np.mean(final_errors**2))
        
        print(f"\n标定完成!")
        print(f"RMS重投影误差: {rms_error:.4f} pixels")
        print(f"最终内参矩阵:\n{self.camera_matrix}")
        print(f"畸变系数: {self.dist_coeffs}")
        
        return rms_error, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs
    
    def validate_calibration(self):
        """验证标定结果"""
        if self.camera_matrix is None:
            print("请先进行标定")
            return
        
        print("\n=== 标定结果验证 ===")
        total_error = 0
        total_points = 0
        
        for i, (obj_pts, img_pts) in enumerate(zip(self.object_points, self.image_points)):
            # 重投影
            proj_pts = self.project_points(
                obj_pts, self.rvecs[i], self.tvecs[i], 
                self.camera_matrix, self.dist_coeffs
            )
            
            # 计算误差
            errors = np.sqrt(np.sum((proj_pts - img_pts)**2, axis=1))
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            
            print(f"图像 {i+1}: 平均误差 {mean_error:.3f} px, 最大误差 {max_error:.3f} px")
            
            total_error += np.sum(errors**2)
            total_points += len(errors)
        
        rms_total = np.sqrt(total_error / total_points)
        print(f"总体RMS误差: {rms_total:.4f} pixels")

def demo_manual_calibration():
    """演示手动标定的使用"""
    
    # 创建标定对象
    calib = ManualCameraCalibration()
    
    # 模拟一些标定数据 (实际使用时从棋盘格检测中获得)
    print("生成模拟标定数据...")
    
    # 棋盘格参数
    pattern_size = (9, 6)
    square_size = 25.0  # mm
    
    # 生成棋盘格3D坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 模拟相机参数
    true_K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    true_dist = np.array([0.1, -0.2, 0.001, 0.002, 0.1])
    
    # 生成多个视角的数据
    np.random.seed(42)
    for i in range(15):
        # 随机生成相机位姿
        rvec = np.random.uniform(-0.3, 0.3, 3)
        tvec = np.array([
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100), 
            np.random.uniform(400, 600)
        ])
        
        # 投影到图像平面
        imgpts, _ = cv2.projectPoints(objp, rvec, tvec, true_K, true_dist)
        imgpts = imgpts.reshape(-1, 2)
        
        # 添加一些噪声
        imgpts += np.random.normal(0, 0.5, imgpts.shape)
        
        calib.add_points(objp, imgpts)
    
    print(f"添加了 {len(calib.object_points)} 组标定数据")
    
    # 执行手动标定
    image_size = (640, 480)
    rms_error, K, dist, rvecs, tvecs = calib.calibrate_camera_manual(image_size)
    
    # 验证结果
    calib.validate_calibration()
    
    # 与真实值比较
    print(f"\n=== 与真实值比较 ===")
    print(f"真实内参:\n{true_K}")
    print(f"估计内参:\n{K}")
    print(f"内参误差: {np.abs(K - true_K).max():.2f}")
    
    print(f"真实畸变: {true_dist}")
    print(f"估计畸变: {dist}")
    print(f"畸变误差: {np.abs(dist - true_dist).max():.4f}")

if __name__ == "__main__":
    demo_manual_calibration()