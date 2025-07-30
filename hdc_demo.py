import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# --- 1. 设置参数 ---

# 棋盘格标定板的尺寸 (内角点数量)
# 【请务必根据你的标定板修改这里!】 例如一个 9x6 的棋盘格，其内角点是 8x5
chessboard_size = (8, 6)
# 棋盘格每个方格的物理尺寸 (mm)
square_size = 25.0

# 准备世界坐标系中的点 (0,0,0), (25,0,0), (50,0,0) ...
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size
print(objp)
# 用于存储所有图像的世界坐标点和图像像素坐标点
objpoints = []  # 3D points in real world space
imgpoints_l = []  # 2D points in left image plane
imgpoints_r = []  # 2D points in right image plane

# ===================================================================
# 图像路径 (已修改为 .jpg)
left_images_path = 'images/left/*.jpg'
right_images_path = 'images/right/*.jpg'
# ===================================================================

# 获取图像列表，并排序以确保左右图像对齐
left_images = sorted(glob.glob(left_images_path))
right_images = sorted(glob.glob(right_images_path))

if not left_images or not right_images:
    print("错误：未找到任何图像。请检查图像路径。")
    print(f"搜索路径: '{left_images_path}' 和 '{right_images_path}'")
    exit()

# 创建用于保存结果的目录
output_dir = "calibration_results"
os.makedirs(output_dir, exist_ok=True)


# --- 2. 角点检测与可视化 (已修正逻辑) ---
print("开始角点检测...")

# 用于确保只可视化一次的标志
visualization_done = False

for i, (left_img_path, right_img_path) in enumerate(zip(left_images, right_images)):
    img_l = cv2.imread(left_img_path)
    img_r = cv2.imread(right_img_path)
    
    # 检查图像是否加载成功
    if img_l is None or img_r is None:
        print(f"警告: 无法加载图像对 {os.path.basename(left_img_path)} / {os.path.basename(right_img_path)}。跳过。")
        continue

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)

    # 如果左右图像都成功找到了角点
    if ret_l and ret_r:
        print(f"在图像对 {i+1} 中成功找到角点。")
        objpoints.append(objp)

        # 亚像素级角点精炼
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        imgpoints_l.append(corners_l2)
        corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        imgpoints_r.append(corners_r2)
        
        # 可视化第一对成功的图像
        if not visualization_done:
            print("正在生成角点检测可视化图...")
            drawn_img_l = cv2.drawChessboardCorners(img_l.copy(), chessboard_size, corners_l2, ret_l)
            drawn_img_r = cv2.drawChessboardCorners(img_r.copy(), chessboard_size, corners_r2, ret_r)
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle('Corner Detection Visualization (First Successful Pair)')
            
            axes[0].imshow(cv2.cvtColor(drawn_img_l, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Left Camera - Corners Detected')
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(drawn_img_r, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Right Camera - Corners Detected')
            axes[1].axis('off')
            
            plt.savefig(os.path.join(output_dir, "corner_detection.png"))
            plt.show()
            visualization_done = True # 将标志设为True，防止再次绘图

    else:
        print(f"警告: 在图像对 {i+1} 中未能找到完整的角点。跳过此对。")

if not objpoints:
    print("\n错误：在任何图像对中都未能找到足够的角点。标定失败。")
    print("请检查 chessboard_size 参数设置和图像质量。")
    exit()

print(f"\n在 {len(objpoints)}/{len(left_images)} 个图像对中成功找到角点。")


# --- 3. 单目标定 (获取初始内参) ---
print("\n开始为每个相机进行单目标定...")
img_shape = gray_l.shape[::-1]
# 重投影均方根误差 (RMS Reprojection Error): ret
# 相机内参矩阵 (Camera Intrinsic Matrix)，也称为 K 矩阵
# 畸变系数 (Distortion Coefficients): dist
# 旋转向量 (Rotation Vectors): rvecs
# 平移向量 (Translation Vectors): tvecs
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)
print("单目标定完成。")
print("左相机内参矩阵 (K_left):\n", mtx_l)
print("\n右相机内参矩阵 (K_right):\n", mtx_r)


# --- 4. 立体标定 (计算左右相机相对位姿) ---
print("\n开始立体标定...")
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret_stereo, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, img_shape,
    criteria=criteria_stereo, flags=flags)
print("立体标定完成。")
print("旋转矩阵 (R):\n", R)
print("平移向量 (T) (单位: mm):\n", T)
print("\n立体标定重投影均方根误差 (RMS Error): ", ret_stereo)


# # --- 5. 立体校正与可视化 ---
# print("\n开始立体校正...")
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
#     M1, d1, M2, d2, img_shape, R, T)
# left_map1, left_map2 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, img_shape, cv2.CV_16SC2)
# right_map1, right_map2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2, img_shape, cv2.CV_16SC2)
# print("正在生成立体校正可视化图...")
# img_l_orig = cv2.imread(left_images[0])
# img_r_orig = cv2.imread(right_images[0])
# dst_l = cv2.remap(img_l_orig, left_map1, left_map2, cv2.INTER_LINEAR)
# dst_r = cv2.remap(img_r_orig, right_map1, right_map2, cv2.INTER_LINEAR)
# h, w = dst_l.shape[:2]
# combined_rectified_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
# combined_rectified_img[:, :w, :] = dst_l
# combined_rectified_img[:, w:, :] = dst_r
# for i in range(20, h, 25):
#     cv2.line(combined_rectified_img, (0, i), (w * 2, i), (0, 255, 0), 1)
# plt.figure(figsize=(15, 7))
# plt.imshow(cv2.cvtColor(combined_rectified_img, cv2.COLOR_BGR2RGB))
# plt.title('Stereo Rectified Image with Epipolar Lines')
# plt.axis('off')
# plt.savefig(os.path.join(output_dir, "rectified_image.png"))
# plt.show()


# # --- 6. 保存标定结果 ---
# print("\n正在保存标定结果...")
# calibration_data = {
#     'M1': M1, 'd1': d1, 'M2': M2, 'd2': d2, 'R': R, 'T': T, 'E': E, 'F': F,
#     'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q, 'image_size': img_shape,
#     'valid_roi1': validPixROI1, 'valid_roi2': validPixROI2, 'reprojection_error': ret_stereo
# }
# np.savez(os.path.join(output_dir, "stereo_calibration.npz"), **calibration_data)
# print(f"标定数据已保存至 {os.path.join(output_dir, 'stereo_calibration.npz')}")
# print("\n标定流程成功结束！")