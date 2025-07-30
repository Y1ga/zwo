import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

def find_chessboard_corners(left_images_path, right_images_path, chessboard_size, square_size, output_dir, visualize=True):
    """
    检测所有图像对中的棋盘格角点。

    Args:
        left_images_path (str): 左相机图像的 glob 路径。
        right_images_path (str): 右相机图像的 glob 路径。
        chessboard_size (tuple): 棋盘格的内角点数 (width, height)。
        square_size (float): 棋盘格方块的物理尺寸。
        output_dir (str): 可视化结果的保存目录。
        visualize (bool): 是否可视化第一对成功检测的角点。

    Returns:
        tuple: (objpoints, imgpoints_l, imgpoints_r, img_shape)
               如果失败则返回 (None, None, None, None)。
    """
    print("开始角点检测...")
    
    # 准备世界坐标系中的点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    left_images = sorted(glob.glob(left_images_path))
    right_images = sorted(glob.glob(right_images_path))

    if not left_images or not right_images or len(left_images) != len(right_images):
        print("错误：图像路径无效或左右图像数量不匹配。")
        return None, None, None, None

    img_shape = None
    visualization_done = not visualize

    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        img_l = cv2.imread(left_path)
        img_r = cv2.imread(right_path)
        if img_l is None or img_r is None:
            print(f"警告: 无法加载图像对 {os.path.basename(left_path)} / {os.path.basename(right_path)}。")
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray_l.shape[::-1]

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)

        if ret_l and ret_r:
            print(f"在图像对 {i+1}/{len(left_images)} 中成功找到角点。")
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            imgpoints_l.append(corners_l2)
            corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            imgpoints_r.append(corners_r2)

            if not visualization_done:
                drawn_img_l = cv2.drawChessboardCorners(img_l.copy(), chessboard_size, corners_l2, ret_l)
                drawn_img_r = cv2.drawChessboardCorners(img_r.copy(), chessboard_size, corners_r2, ret_r)
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                axes[0].imshow(cv2.cvtColor(drawn_img_l, cv2.COLOR_BGR2RGB))
                axes[0].set_title('Left Camera - Corners Detected')
                axes[0].axis('off')
                axes[1].imshow(cv2.cvtColor(drawn_img_r, cv2.COLOR_BGR2RGB))
                axes[1].set_title('Right Camera - Corners Detected')
                axes[1].axis('off')
                plt.savefig(os.path.join(output_dir, "corner_detection.png"))
                plt.show()
                visualization_done = True
        else:
            print(f"警告: 在图像对 {i+1}/{len(left_images)} 中未能找到完整的角点。")
            
    if not objpoints:
        return None, None, None, None
        
    print(f"\n在 {len(objpoints)}/{len(left_images)} 个图像对中成功找到角点。")
    return objpoints, imgpoints_l, imgpoints_r, img_shape

def run_stereo_calibration(objpoints, imgpoints_l, imgpoints_r, img_shape):
    """
    执行立体标定。

    Args:
        objpoints (list): 世界坐标点列表。
        imgpoints_l (list): 左相机图像点列表。
        imgpoints_r (list): 右相机图像点列表。
        img_shape (tuple): 图像尺寸 (width, height)。

    Returns:
        dict: 包含所有标定结果的字典。
    """
    print("\n开始执行立体标定...")
    # 先进行单目标定以获取内参初始值
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

    # 立体标定
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = 0 # 允许函数优化内参
    ret_stereo, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r, img_shape,
        criteria=criteria_stereo, flags=flags
    )

    print("立体标定完成。RMS Error:", ret_stereo)
    
    calibration_data = {
        'mtx_l': M1, 'dist_l': d1,
        'mtx_r': M2, 'dist_r': d2,
        'R': R, 'T': T, 'E': E, 'F': F,
        'img_shape': img_shape, 'rms_error': ret_stereo
    }
    return calibration_data

def perform_and_visualize_rectification(calibration_data, left_img_path, right_img_path, output_dir):
    """
    执行立体校正并可视化结果。

    Args:
        calibration_data (dict): 标定结果字典。
        left_img_path (str): 用于测试的左图路径。
        right_img_path (str): 用于测试的右图路径。
        output_dir (str): 可视化结果的保存目录。

    Returns:
        dict: 包含所有校正参数的字典。
    """
    print("\n开始立体校正与可视化检验...")
    M1, d1, M2, d2, R, T, img_shape = [calibration_data[k] for k in ('mtx_l', 'dist_l', 'mtx_r', 'dist_r', 'R', 'T', 'img_shape')]
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(M1, d1, M2, d2, img_shape, R, T, alpha=0.9)
    
    map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, img_shape, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, img_shape, cv2.CV_16SC2)
    
    img_l_test = cv2.imread(left_img_path)
    img_r_test = cv2.imread(right_img_path)

    rectified_l = cv2.remap(img_l_test, map1_l, map2_l, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(img_r_test, map1_r, map2_r, cv2.INTER_LINEAR)
    
    rectified_pair = np.hstack((rectified_l, rectified_r))
    for i in range(20, rectified_pair.shape[0], 50):
        cv2.line(rectified_pair, (0, i), (rectified_pair.shape[1], i), (0, 255, 0), 1)
        
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(rectified_pair, cv2.COLOR_BGR2RGB))
    plt.title('Rectified Stereo Pair with Epipolar Lines')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "rectification_visualization.png"))
    plt.show()

    rectification_data = {
        'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
        'map1_l': map1_l, 'map2_l': map2_l,
        'map1_r': map1_r, 'map2_r': map2_r
    }
    return rectification_data

def save_parameters(filepath, data):
    """将参数字典保存到 .npz 文件。"""
    print(f"\n正在保存参数到 '{filepath}'...")
    np.savez(filepath, **data)
    print("保存成功。")

def main():
    """主执行函数"""
    # --- 1. 定义配置 ---
    config = {
        "chessboard_size": (8, 6),
        "square_size": 25.0,
        "left_images_path": 'images/left/*.jpg',
        "right_images_path": 'images/right/*.jpg',
        "output_dir": "calibration_results"
    }
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # --- 2. 查找角点 ---
    objpoints, imgpoints_l, imgpoints_r, img_shape = find_chessboard_corners(
        config["left_images_path"], config["right_images_path"],
        config["chessboard_size"], config["square_size"], config["output_dir"]
    )
    if objpoints is None:
        print("角点检测失败，程序终止。")
        return

    # --- 3. 执行标定 ---
    calibration_data = run_stereo_calibration(objpoints, imgpoints_l, imgpoints_r, img_shape)
    
    # --- 4. 保存核心标定参数 ---
    save_parameters(os.path.join(config["output_dir"], "stereo_calibration_params.npz"), calibration_data)

    # --- 5. 执行校正并保存校正参数 ---
    # 使用第一对图像进行可视化
    left_images = sorted(glob.glob(config["left_images_path"]))
    right_images = sorted(glob.glob(config["right_images_path"]))
    rectification_data = perform_and_visualize_rectification(
        calibration_data, left_images[0], right_images[0], config["output_dir"]
    )
    save_parameters(os.path.join(config["output_dir"], "stereo_rectification_params.npz"), rectification_data)
    
    print("\n标定与校正流程全部完成！")

if __name__ == '__main__':
    main()