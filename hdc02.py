import cv2
import numpy as np
import glob
import os
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

class ManualCameraCalibration:
    def __init__(self):
        """手写相机标定类"""
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
    def project_points_manual(self, object_points, rvec, tvec, camera_matrix, dist_coeffs=None):
        """
        手动实现3D点到2D图像点的投影
        
        Args:
            object_points: 3D世界坐标点 (N, 3)
            rvec: 旋转向量 (3,) 或 (3, 1)
            tvec: 平移向量 (3,) 或 (3, 1)
            camera_matrix: 相机内参矩阵 (3, 3)
            dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3] (可选)
            
        Returns:
            projected_points: 投影后的2D点 (N, 2)
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
        if dist_coeffs is not None and len(dist_coeffs) >= 4:
            x, y = self.apply_distortion(x, y, dist_coeffs)
        
        # 5. 应用相机内参
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        u = fx * x + cx
        v = fy * y + cy
        
        return np.column_stack([u, v])
    
    def rodrigues_manual(self, rvec):
        """
        手动实现Rodrigues公式：旋转向量转旋转矩阵
        """
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
    
    def apply_distortion(self, x, y, dist_coeffs):
        """应用径向和切向畸变"""
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
        """手动实现相机标定（改进版）"""
        n_views = len(object_points_list)
        
        print(f"开始手动标定，共 {n_views} 个视图...")
        
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
            
            # 初始参数向量
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
                print(f"  优化成功: {result.message}")
            else:
                print(f"  优化警告: {result.message}")
            
            # 解包优化结果
            camera_matrix_opt, dist_coeffs_opt, rvecs_opt, tvecs_opt = self.unpack_parameters(
                result.x, len(valid_views)
            )
            
        except Exception as e:
            print(f"优化失败: {e}")
            # 使用初始估计
            camera_matrix_opt = camera_matrix
            dist_coeffs_opt = np.zeros(5)
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
                    camera_matrix_opt, dist_coeffs_opt
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
        
        print(f"手动标定完成! RMS重投影误差: {rms_error:.4f} pixels")
        
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

class StereoCalibrationWithManual:
    def __init__(self, chessboard_size=(9, 6), square_size=25.0):
        """
        带手写标定功能的双目标定类
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # 准备棋盘格角点的世界坐标
        self.prepare_object_points()
        
        # 存储检测到的角点
        self.object_points = []
        self.img_points_left = []
        self.img_points_right = []
        
        # 手写标定器
        self.manual_calib_left = ManualCameraCalibration()
        self.manual_calib_right = ManualCameraCalibration()
        
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
            
    def calibrate_cameras_manual(self, img_size):
        """使用手写算法分别标定左右相机"""
        if len(self.object_points) == 0:
            raise ValueError("没有找到有效的角点，请先运行 find_chessboard_corners")
        
        print("\n=== 手写算法标定左相机 ===")
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = self.manual_calib_left.calibrate_camera_manual(
            self.object_points, self.img_points_left, img_size
        )
        
        print("\n=== 手写算法标定右相机 ===")
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = self.manual_calib_right.calibrate_camera_manual(
            self.object_points, self.img_points_right, img_size
        )
        
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
                'tvecs': tvecs_right
            },
            'right': {
                'ret': ret_right,
                'camera_matrix': mtx_right,
                'dist_coeffs': dist_right,
                'rvecs': rvecs_right,
                'tvecs': tvecs_right
            }
        }
    
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
    """使用真实图像数据进行双目标定对比"""
    
    # 初始化双目标定
    stereo_calib = StereoCalibrationWithManual(chessboard_size=(9, 6), square_size=25.0)
    
    # 设置你的图像路径
    left_images_path = "./images/left/*.jpg"
    right_images_path = "./images/right/*.jpg"
    
    try:
        # 寻找角点
        print("正在检测棋盘格角点...")
        stereo_calib.find_chessboard_corners(left_images_path, right_images_path)
        
        # 获取图像尺寸 (需要根据你的实际图像调整)
        first_img = cv2.imread(glob.glob(left_images_path)[0])
        if first_img is not None:
            img_size = (first_img.shape[1], first_img.shape[0])
            print(f"图像尺寸: {img_size}")
        else:
            raise ValueError("无法读取图像，请检查路径")
        
        # 手写算法标定
        print("\n开始手写算法标定...")
        manual_results = stereo_calib.calibrate_cameras_manual(img_size)
        
        # OpenCV算法标定
        print("\n开始OpenCV算法标定...")
        opencv_results = stereo_calib.calibrate_cameras_opencv(img_size)
        
        # 对比结果
        stereo_calib.compare_calibration_results(manual_results, opencv_results)
        
        # 保存结果
        np.savez('manual_calibration_results.npz', 
                manual_left_camera_matrix=manual_results['left']['camera_matrix'],
                manual_left_dist_coeffs=manual_results['left']['dist_coeffs'],
                manual_right_camera_matrix=manual_results['right']['camera_matrix'], 
                manual_right_dist_coeffs=manual_results['right']['dist_coeffs'],
                opencv_left_camera_matrix=opencv_results['left']['camera_matrix'],
                opencv_left_dist_coeffs=opencv_results['left']['dist_coeffs'],
                opencv_right_camera_matrix=opencv_results['right']['camera_matrix'],
                opencv_right_dist_coeffs=opencv_results['right']['dist_coeffs'])
        
        print(f"\n标定结果已保存到: manual_calibration_results.npz")
        
    except Exception as e:
        print(f"标定过程中出现错误: {e}")
        print("请检查:")
        print("1. 图像路径是否正确")
        print("2. 棋盘格尺寸参数是否匹配")
        print("3. 图像中是否包含完整的棋盘格")

if __name__ == "__main__":
    main()