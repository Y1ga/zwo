#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
振旺相机控制界面 - 增强版（优化串口性能和曝光显示，新增手动移动ROI功能）
功能：曝光、增益设置，实时视频流显示，RAW16图片保存，灰度值统计，串口通信（收发）
优化：解决发送延迟问题，增强清空接收区功能，优化曝光时间显示（us/ms/s自动转换）
新增：智能文件命名(Violet索引_曝光时间.png)，智能路径选择，会话管理，手动移动ROI功能
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import numpy as np
from PIL import Image, ImageTk
import datetime
import os
import sys

# 系统信息库
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告：未安装psutil库，智能路径选择功能将使用简化模式")

# 串口通信
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("警告：未安装pyserial库，串口功能将不可用")

# matplotlib相关导入
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.style as mplstyle
import matplotlib
mplstyle.use('fast')  # 优化性能

# 配置matplotlib中文字体支持
try:
    import platform
    system = platform.system()
    if system == "Windows":
        # Windows系统字体配置
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
    elif system == "Darwin":
        # macOS系统字体配置
        matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Hiragino Sans GB', 'DejaVu Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
    else:
        # Linux系统字体配置
        matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
    
    # 禁用matplotlib的字体警告
    matplotlib.rcParams['axes.unicode_minus'] = False
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
except Exception as e:
    print(f"matplotlib字体配置警告: {e}")

# 假设zwoasi模块在同一目录或已安装
try:
    import zwoasi as asi
except ImportError:
    print("错误：未找到zwoasi模块，请确保zwoasi库已正确安装")
    sys.exit(1)


class ZWOCameraController:
    def __init__(self, root):
        self.root = root
        self.root.title("振旺相机控制界面 - 增强版（优化串口性能和曝光显示，新增手动移动ROI功能）")
        self.root.geometry("1700x1000")
        
        # 配置tkinter和matplotlib中文字体以避免字体警告
        try:
            import platform
            import tkinter.font as tkFont
            
            system = platform.system()
            if system == "Windows":
                # Windows系统字体配置 - 修复字体配置方式
                try:
                    # 测试字体是否可用
                    test_font = tkFont.Font(family='Microsoft YaHei', size=9)
                    default_font_family = 'Microsoft YaHei'
                except:
                    # 如果微软雅黑不可用，尝试其他字体
                    try:
                        test_font = tkFont.Font(family='SimHei', size=9)
                        default_font_family = 'SimHei'
                    except:
                        default_font_family = 'Arial'  # 最后回退到Arial
                
                # 配置所有默认字体
                for font_name in ['TkDefaultFont', 'TkTextFont', 'TkFixedFont', 'TkMenuFont', 'TkHeadingFont', 'TkCaptionFont', 'TkSmallCaptionFont', 'TkIconFont', 'TkTooltipFont']:
                    try:
                        font = tkFont.nametofont(font_name)
                        font.configure(family=default_font_family, size=9)
                    except Exception as e:
                        print(f"配置字体 {font_name} 失败: {e}")
                        
            elif system == "Darwin":
                # macOS系统字体配置
                try:
                    test_font = tkFont.Font(family='PingFang SC', size=9)
                    default_font_family = 'PingFang SC'
                except:
                    try:
                        test_font = tkFont.Font(family='Arial Unicode MS', size=9)
                        default_font_family = 'Arial Unicode MS'
                    except:
                        default_font_family = 'Arial'
                
                for font_name in ['TkDefaultFont', 'TkTextFont', 'TkFixedFont', 'TkMenuFont', 'TkHeadingFont', 'TkCaptionFont', 'TkSmallCaptionFont', 'TkIconFont', 'TkTooltipFont']:
                    try:
                        font = tkFont.nametofont(font_name)
                        font.configure(family=default_font_family, size=9)
                    except Exception as e:
                        print(f"配置字体 {font_name} 失败: {e}")
                        
            else:
                # Linux系统字体配置
                try:
                    test_font = tkFont.Font(family='WenQuanYi Micro Hei', size=9)
                    default_font_family = 'WenQuanYi Micro Hei'
                except:
                    try:
                        test_font = tkFont.Font(family='DejaVu Sans', size=9)
                        default_font_family = 'DejaVu Sans'
                    except:
                        default_font_family = 'Arial'
                
                for font_name in ['TkDefaultFont', 'TkTextFont', 'TkFixedFont', 'TkMenuFont', 'TkHeadingFont', 'TkCaptionFont', 'TkSmallCaptionFont', 'TkIconFont', 'TkTooltipFont']:
                    try:
                        font = tkFont.nametofont(font_name)
                        font.configure(family=default_font_family, size=9)
                    except Exception as e:
                        print(f"配置字体 {font_name} 失败: {e}")
            
            print(f"已配置默认字体: {default_font_family}")
            
            # 禁用tkinter的字体警告
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
            
        except Exception as e:
            print(f"字体设置失败，使用系统默认字体: {e}")
        
        # 相机相关变量
        self.camera = None
        self.is_capturing = False
        self.capture_thread = None
        self.current_frame = None
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # 相机属性
        self.max_width = 1920
        self.max_height = 1080
        self.supported_bins = [1, 2, 4]
        
        # ROI拖拽控制相关变量
        self.roi_move_step = 10  # 保留，用于微调
        self.roi_move_fast_step = 50
        
        # ROI Canvas相关变量
        self.roi_dragging = False
        self.roi_drag_start_x = 0
        self.roi_drag_start_y = 0
        self.roi_rect_id = None
        self.roi_circle_id = None
        self.sensor_rect_id = None
        self.canvas_scale_x = 1.0  # Canvas坐标到传感器坐标的缩放比例
        self.canvas_scale_y = 1.0
        
        # 文件保存相关变量
        self.file_index = 1  # 文件索引，从1开始
        self.current_session_folder = None  # 当前会话文件夹
        
        # 自动曝光相关变量
        self.auto_exposure_enabled = False
        self.target_gray_value = 60000  # 16位图像的目标灰度值（约92%满量程）
        self.safe_exposure_time = 10000  # 安全曝光时间（微秒），默认10ms
        self.auto_exposure_adjustment_interval = 10  # 每10帧检查一次自动曝光（更快响应）
        self.auto_exposure_counter = 0
        self.last_auto_exposure_time = 0
        self.exposure_adjustment_factor = 1.2  # 曝光调整因子
        self.auto_exposure_history = []  # 记录最近几次的调整历史，用于智能调整
        
        # 灰度统计相关变量
        self.gray_min = 0
        self.gray_max = 0
        self.gray_mean = 0
        self.histogram_data = None
        self.stats_update_counter = 0
        
        # 串口相关变量
        self.serial_port = None
        self.is_serial_open = False
        self.serial_thread = None
        self.serial_buffer = []
        self.max_buffer_size = 1000
        self.receive_byte_count = 0  # 新增：接收字节计数
        
        # 串口发送相关变量
        self.send_stats = {
            'total_sent': 0,
            'last_send_time': 0,
            'send_errors': 0,
            'send_history': []  # 新增：发送时间历史记录
        }
        
        # 串口优化选项
        self.auto_clear_buffer = tk.BooleanVar(value=True)  # 新增：自动清空缓冲区
        self.clear_before_send = tk.BooleanVar(value=True)  # 新增：发送前清空
        self.performance_mode = tk.BooleanVar(value=False)  # 新增：性能模式
        
        # 默认保存路径
        self.save_path = self.get_smart_save_path()
        
        # 创建GUI界面
        self.create_widgets()
        
        # 初始化相机
        self.initialize_camera()
        
        # 创建初始会话文件夹
        self.get_current_session_folder()
        
        # 初始化串口端口列表
        if SERIAL_AVAILABLE:
            self.refresh_serial_ports()
            # 自动打开最新的串口
            self.auto_open_latest_serial()
        
        # 初始化ROI Canvas
        self.update_roi_canvas()
    
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 左侧控制面板
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 中间视频显示区域
        middle_frame = ttk.Frame(main_frame)
        middle_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 右侧统计面板
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === 左侧面板内容 ===
        # 相机控制面板
        control_frame = ttk.LabelFrame(left_frame, text="相机控制", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 曝光控制
        ttk.Label(control_frame, text="曝光时间 (μs):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.exposure_var = tk.StringVar(value="10000")
        exposure_entry = ttk.Entry(control_frame, textvariable=self.exposure_var, width=10)
        exposure_entry.grid(row=0, column=1, padx=(0, 5))
        exposure_entry.bind('<Return>', self.set_exposure)

        # 新增：曝光时间格式化显示标签
        self.exposure_display_label = ttk.Label(control_frame, text="", width=10, foreground="blue", anchor=tk.W)
        self.exposure_display_label.grid(row=0, column=2, padx=(0, 5))
        
        ttk.Button(control_frame, text="设置曝光", command=self.set_exposure).grid(row=0, column=3, padx=(0, 10))
        
        # 自动曝光控制
        self.auto_exposure_var = tk.BooleanVar(value=False)
        self.auto_exposure_checkbox = ttk.Checkbutton(control_frame, text="自动曝光", variable=self.auto_exposure_var, command=self.toggle_auto_exposure)
        self.auto_exposure_checkbox.grid(row=0, column=4, padx=(10, 5))
        
        # 目标灰度值设置
        ttk.Label(control_frame, text="目标灰度:").grid(row=0, column=5, sticky=tk.W, padx=(0, 5))
        self.target_gray_var = tk.StringVar(value="60000")  # 对于16位图像，默认目标灰度值
        target_entry = ttk.Entry(control_frame, textvariable=self.target_gray_var, width=8)
        target_entry.grid(row=0, column=6, padx=(0, 5))
        target_entry.bind('<Return>', self.update_target_gray)
        
        # 安全曝光时间设置
        ttk.Label(control_frame, text="安全曝光(ms):").grid(row=0, column=7, sticky=tk.W, padx=(10, 5))
        self.safe_exposure_var = tk.StringVar(value="10")  # 默认10ms
        safe_exp_entry = ttk.Entry(control_frame, textvariable=self.safe_exposure_var, width=6)
        safe_exp_entry.grid(row=0, column=8, padx=(0, 5))
        safe_exp_entry.bind('<Return>', self.update_safe_exposure)
        
        # 增益控制 (向右移动)
        ttk.Label(control_frame, text="增益:").grid(row=0, column=9, sticky=tk.W, padx=(10, 5))
        self.gain_var = tk.StringVar(value="0")
        gain_entry = ttk.Entry(control_frame, textvariable=self.gain_var, width=6)
        gain_entry.grid(row=0, column=10, padx=(0, 5))
        gain_entry.bind('<Return>', self.set_gain)
        
        ttk.Button(control_frame, text="设置增益", command=self.set_gain).grid(row=0, column=11)

        # 分辨率和ROI控制面板
        roi_frame = ttk.LabelFrame(left_frame, text="分辨率和ROI设置", padding="10")
        roi_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 第一行：分辨率设置
        ttk.Label(roi_frame, text="分辨率:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.width_var = tk.StringVar(value="640")
        width_entry = ttk.Entry(roi_frame, textvariable=self.width_var, width=6)
        width_entry.grid(row=0, column=1, padx=(0, 2))
        
        ttk.Label(roi_frame, text="×").grid(row=0, column=2, padx=2)
        
        self.height_var = tk.StringVar(value="480")
        height_entry = ttk.Entry(roi_frame, textvariable=self.height_var, width=6)
        height_entry.grid(row=0, column=3, padx=(2, 5))
        
        # Binning设置
        ttk.Label(roi_frame, text="Binning:").grid(row=0, column=4, sticky=tk.W, padx=(10, 5))
        self.binning_var = tk.StringVar(value="1")
        binning_combo = ttk.Combobox(roi_frame, textvariable=self.binning_var, width=4, values=["1", "2", "4"])
        binning_combo.grid(row=0, column=5, padx=(0, 5))
        binning_combo.state(['readonly'])
        
        ttk.Button(roi_frame, text="应用分辨率", command=self.apply_resolution).grid(row=0, column=6, padx=(5, 0))
        
        # 第二行：ROI位置设置
        ttk.Label(roi_frame, text="ROI起始:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        
        ttk.Label(roi_frame, text="X:").grid(row=1, column=1, sticky=tk.E, pady=(10, 0))
        self.roi_x_var = tk.StringVar(value="0")
        roi_x_entry = ttk.Entry(roi_frame, textvariable=self.roi_x_var, width=6)
        roi_x_entry.grid(row=1, column=2, padx=(2, 5), pady=(10, 0))
        
        ttk.Label(roi_frame, text="Y:").grid(row=1, column=3, sticky=tk.E, pady=(10, 0))
        self.roi_y_var = tk.StringVar(value="0")
        roi_y_entry = ttk.Entry(roi_frame, textvariable=self.roi_y_var, width=6)
        roi_y_entry.grid(row=1, column=4, padx=(2, 5), pady=(10, 0))
        
        ttk.Button(roi_frame, text="居中ROI", command=self.center_roi).grid(row=1, column=5, padx=(5, 0), pady=(10, 0))
        ttk.Button(roi_frame, text="应用ROI", command=self.apply_roi).grid(row=1, column=6, padx=(5, 0), pady=(10, 0))
        
        # === 新增：ROI可视化拖拽控制面板 ===
        roi_visual_frame = ttk.LabelFrame(roi_frame, text="ROI可视化拖拽控制", padding="10")
        roi_visual_frame.grid(row=2, column=0, columnspan=7, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 第一行：控制说明和选项
        control_info_frame = ttk.Frame(roi_visual_frame)
        control_info_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        info_text = "操作说明: 拖拽红色圆点移动ROI位置，双击圆点居中ROI"
        ttk.Label(control_info_frame, text=info_text, font=("Arial", 9), foreground="blue").pack(side=tk.LEFT)
        
        ttk.Button(control_info_frame, text="居中ROI", command=self.center_roi).pack(side=tk.RIGHT, padx=(10, 0))
        
        # 第二行：ROI预览Canvas
        canvas_frame = ttk.Frame(roi_visual_frame)
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建ROI预览Canvas
        self.roi_canvas_width = 300
        self.roi_canvas_height = 200
        self.roi_canvas = tk.Canvas(canvas_frame, width=self.roi_canvas_width, height=self.roi_canvas_height, 
                                   bg='lightgray', relief='sunken', bd=2)
        self.roi_canvas.pack(padx=5, pady=5)
        
        # ROI拖拽相关变量
        self.roi_dragging = False
        self.roi_drag_start_x = 0
        self.roi_drag_start_y = 0
        self.roi_rect_id = None
        self.roi_circle_id = None
        self.sensor_rect_id = None
        
        # 绑定鼠标事件
        self.roi_canvas.bind("<Button-1>", self.on_roi_canvas_click)
        self.roi_canvas.bind("<B1-Motion>", self.on_roi_canvas_drag)
        self.roi_canvas.bind("<ButtonRelease-1>", self.on_roi_canvas_release)
        self.roi_canvas.bind("<Double-Button-1>", self.on_roi_canvas_double_click)
        
        # 第三行：ROI信息显示
        roi_info_frame = ttk.Frame(roi_visual_frame)
        roi_info_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # ROI位置信息
        ttk.Label(roi_info_frame, text="ROI位置信息:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        info_text_frame = ttk.Frame(roi_info_frame)
        info_text_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_text_frame, text="X坐标:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.roi_pos_x_label = ttk.Label(info_text_frame, text="0", font=("Arial", 9, "bold"), foreground="red")
        self.roi_pos_x_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(info_text_frame, text="Y坐标:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.roi_pos_y_label = ttk.Label(info_text_frame, text="0", font=("Arial", 9, "bold"), foreground="red")
        self.roi_pos_y_label.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(info_text_frame, text="宽度:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self.roi_width_label = ttk.Label(info_text_frame, text="640", font=("Arial", 9))
        self.roi_width_label.grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(info_text_frame, text="高度:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5))
        self.roi_height_label = ttk.Label(info_text_frame, text="480", font=("Arial", 9))
        self.roi_height_label.grid(row=3, column=1, sticky=tk.W)
        
        # 传感器信息
        ttk.Label(roi_info_frame, text="传感器信息:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        
        sensor_info_frame = ttk.Frame(roi_info_frame)
        sensor_info_frame.pack(fill=tk.X)
        
        ttk.Label(sensor_info_frame, text="最大分辨率:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.sensor_size_label = ttk.Label(sensor_info_frame, text="1920×1080", font=("Arial", 9))
        self.sensor_size_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(sensor_info_frame, text="当前Binning:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.current_binning_label = ttk.Label(sensor_info_frame, text="1", font=("Arial", 9))
        self.current_binning_label.grid(row=1, column=1, sticky=tk.W)
        
        # 操作按钮
        button_frame = ttk.Frame(roi_info_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(button_frame, text="刷新预览", command=self.update_roi_canvas).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="应用ROI", command=self.apply_roi_from_canvas).pack(fill=tk.X)
        
        # 第三行：快速设置
        ttk.Label(roi_frame, text="快速设置:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        
        preset_frame = ttk.Frame(roi_frame)
        preset_frame.grid(row=3, column=1, columnspan=6, sticky=tk.W, pady=(10, 0))
        
        ttk.Button(preset_frame, text="全分辨率", command=lambda: self.set_preset_resolution("full")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_frame, text="1280×960", command=lambda: self.set_preset_resolution("1280x960")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_frame, text="640×480", command=lambda: self.set_preset_resolution("640x480")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_frame, text="320×240", command=lambda: self.set_preset_resolution("320x240")).pack(side=tk.LEFT, padx=(0, 5))
        
        # 视频控制面板
        video_frame = ttk.LabelFrame(left_frame, text="视频控制", padding="10")
        video_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(video_frame, text="开始预览", command=self.start_capture)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(video_frame, text="停止预览", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.save_button = ttk.Button(video_frame, text="保存当前帧", command=self.save_frame, state=tk.DISABLED)
        self.save_button.grid(row=0, column=2, padx=(0, 10))
        
        # 文件索引控制
        index_frame = ttk.Frame(video_frame)
        index_frame.grid(row=0, column=3, padx=(10, 0))
        
        ttk.Label(index_frame, text="索引:").pack(side=tk.LEFT)
        self.index_var = tk.StringVar(value="1")
        index_entry = ttk.Entry(index_frame, textvariable=self.index_var, width=6)
        index_entry.pack(side=tk.LEFT, padx=(2, 5))
        index_entry.bind('<Return>', self.update_file_index)
        
        ttk.Button(index_frame, text="重置", command=self.reset_file_index).pack(side=tk.LEFT)
        
        # 统计功能开关
        self.enable_stats_var = tk.BooleanVar(value=False)
        self.stats_checkbox = ttk.Checkbutton(video_frame, text="启用灰度直方图", variable=self.enable_stats_var)
        self.stats_checkbox.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        ttk.Button(video_frame, text="选择保存路径", command=self.select_save_path).grid(row=1, column=2, pady=(10, 0))
        
        # 第二行右侧按钮
        path_buttons_frame = ttk.Frame(video_frame)
        path_buttons_frame.grid(row=1, column=3, pady=(10, 0))
        
        ttk.Button(path_buttons_frame, text="智能路径", command=self.auto_select_smart_path).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(path_buttons_frame, text="打开文件夹", command=self.open_zwodata_folder).pack(side=tk.LEFT)
        
        # === 串口控制面板 ===
        if SERIAL_AVAILABLE:
            serial_control_frame = ttk.LabelFrame(left_frame, text="串口控制", padding="10")
            serial_control_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # 第一行：端口和波特率选择
            ttk.Label(serial_control_frame, text="端口:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
            self.port_var = tk.StringVar()
            self.port_combo = ttk.Combobox(serial_control_frame, textvariable=self.port_var, width=10)
            self.port_combo.grid(row=0, column=1, padx=(0, 10))
            
            ttk.Button(serial_control_frame, text="刷新", command=self.refresh_serial_ports).grid(row=0, column=2, padx=(0, 10))
            
            ttk.Label(serial_control_frame, text="波特率:").grid(row=0, column=3, sticky=tk.W, padx=(0, 5))
            self.baudrate_var = tk.StringVar(value="460800")
            baudrate_combo = ttk.Combobox(serial_control_frame, textvariable=self.baudrate_var, width=8,
                                        values=["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"])
            baudrate_combo.grid(row=0, column=4, padx=(0, 10))
            baudrate_combo.state(['readonly'])
            
            # 第二行：控制按钮
            self.serial_open_button = ttk.Button(serial_control_frame, text="打开串口", command=self.open_serial_port)
            self.serial_open_button.grid(row=1, column=0, pady=(10, 0), padx=(0, 10))
            
            self.serial_close_button = ttk.Button(serial_control_frame, text="关闭串口", command=self.close_serial_port, state=tk.DISABLED)
            self.serial_close_button.grid(row=1, column=1, pady=(10, 0), padx=(0, 10))
            
            ttk.Button(serial_control_frame, text="清空接收区", command=self.clear_serial_data).grid(row=1, column=2, pady=(10, 0))
            
            # 新增：清空硬件缓冲区按钮
            ttk.Button(serial_control_frame, text="清空硬件缓冲区", command=self.clear_hardware_buffers).grid(row=1, column=3, pady=(10, 0))
            
            # 串口状态显示
            self.serial_status_label = ttk.Label(serial_control_frame, text="串口状态: 未连接", foreground="red")
            self.serial_status_label.grid(row=1, column=4, sticky=tk.W, pady=(10, 0), padx=(10, 0))
            
            # 新增：串口优化选项
            serial_opt_frame = ttk.LabelFrame(left_frame, text="串口优化选项", padding="10")
            serial_opt_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
            
            ttk.Checkbutton(serial_opt_frame, text="发送前清空缓冲区", variable=self.clear_before_send).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
            ttk.Checkbutton(serial_opt_frame, text="自动清空缓冲区", variable=self.auto_clear_buffer).grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
            ttk.Checkbutton(serial_opt_frame, text="性能模式(减少显示)", variable=self.performance_mode).grid(row=0, column=2, sticky=tk.W)
            
            # 缓冲区状态显示
            self.buffer_status_label = ttk.Label(serial_opt_frame, text="缓冲区: 0/1000")
            self.buffer_status_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # === 串口发送控制面板 ===
        if SERIAL_AVAILABLE:
            send_control_frame = ttk.LabelFrame(left_frame, text="串口发送控制", padding="10")
            send_control_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # 控制模式选择
            mode_frame = ttk.Frame(send_control_frame)
            mode_frame.grid(row=0, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
            
            ttk.Label(mode_frame, text="控制模式:").pack(side=tk.LEFT, padx=(0, 5))
            self.control_mode_var = tk.StringVar(value="全通道模式")
            mode_combo = ttk.Combobox(mode_frame, textvariable=self.control_mode_var, width=12,
                                    values=["全通道模式", "白光模式"])
            mode_combo.pack(side=tk.LEFT, padx=(0, 10))
            mode_combo.state(['readonly'])
            mode_combo.bind('<<ComboboxSelected>>', self.on_control_mode_changed)
            
            # PWM数据设置区域（全通道模式）
            self.pwm_frame = ttk.LabelFrame(send_control_frame, text="PWM数据设置 (全通道模式)", padding="5")
            self.pwm_frame.grid(row=1, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # 创建16个PWM通道输入框
            self.pwm_vars = []
            for i in range(16):
                row = i // 8
                col = i % 8
                
                ttk.Label(self.pwm_frame, text=f"CH{i+1}:", font=("Arial", 8)).grid(row=row*2, column=col, sticky=tk.W, padx=2)
                var = tk.StringVar(value="0")
                self.pwm_vars.append(var)
                entry = ttk.Entry(self.pwm_frame, textvariable=var, width=5, font=("Arial", 8))
                entry.grid(row=row*2+1, column=col, padx=2, pady=(0, 5))
                entry.bind('<Return>', self.validate_pwm_values)
            
            # 白光模式控制区域（默认隐藏）
            self.white_light_frame = ttk.LabelFrame(send_control_frame, text="白光控制 (白光模式)", padding="10")
            self.white_light_frame.grid(row=1, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
            self.white_light_frame.grid_remove()
            
            # 创建4个白光通道控制
            self.white_light_vars = {}
            white_light_channels = [
                ("CH7 (色温1)", 6),   # 对应CH7，数组索引6
                ("CH8 (色温2)", 7),   # 对应CH8，数组索引7
                ("CH11 (色温3)", 10), # 对应CH11，数组索引10
                ("CH12 (色温4)", 11)  # 对应CH12，数组索引11
            ]
            
            for i, (label, ch_index) in enumerate(white_light_channels):
                row = i // 2
                col = (i % 2) * 3
                
                ttk.Label(self.white_light_frame, text=label, font=("Arial", 10)).grid(row=row, column=col, sticky=tk.W, padx=(0, 5))
                var = tk.StringVar(value="0")
                self.white_light_vars[ch_index] = var
                entry = ttk.Entry(self.white_light_frame, textvariable=var, width=8, font=("Arial", 10))
                entry.grid(row=row, column=col+1, padx=(0, 20))
                entry.bind('<Return>', self.validate_white_light_values)
                
                # 添加滑块控制
                scale = ttk.Scale(self.white_light_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                command=lambda val, idx=ch_index: self.on_white_light_scale_changed(idx, val))
                scale.grid(row=row, column=col+2, padx=(0, 20), sticky=(tk.W, tk.E))
                scale.bind('<ButtonRelease-1>', lambda e, idx=ch_index: self.sync_white_light_entry_from_scale(idx, e))
                
                # 绑定输入框变化到滑块
                var.trace('w', lambda *args, idx=ch_index: self.sync_white_light_scale_from_entry(idx))
            
            # 白光模式快速设置按钮
            white_quick_frame = ttk.Frame(self.white_light_frame)
            white_quick_frame.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(10, 0))
            
            ttk.Label(white_quick_frame, text="快速设置:").pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(white_quick_frame, text="全部清零", command=self.clear_all_white_light).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(white_quick_frame, text="暖白光", command=lambda: self.set_white_light_preset("warm")).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(white_quick_frame, text="冷白光", command=lambda: self.set_white_light_preset("cool")).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(white_quick_frame, text="混合光", command=lambda: self.set_white_light_preset("mixed")).pack(side=tk.LEFT, padx=(0, 5))
            
            # PWM快速设置按钮（全通道模式）
            quick_set_frame = ttk.Frame(send_control_frame)
            quick_set_frame.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
            
            ttk.Label(quick_set_frame, text="快速设置:").pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_set_frame, text="全部清零", command=lambda: self.set_all_pwm(0)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_set_frame, text="全部25", command=lambda: self.set_all_pwm(25)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_set_frame, text="全部50", command=lambda: self.set_all_pwm(50)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_set_frame, text="全部100", command=lambda: self.set_all_pwm(100)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_set_frame, text="示例数据", command=self.set_example_pwm).pack(side=tk.LEFT, padx=(0, 5))
            
            # 发送控制区域
            send_cmd_frame = ttk.Frame(send_control_frame)
            send_cmd_frame.grid(row=3, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 5))
            
            # 发送格式选择
            ttk.Label(send_cmd_frame, text="发送格式:").pack(side=tk.LEFT, padx=(0, 5))
            self.send_format_var = tk.StringVar(value="PWM数据")
            format_combo = ttk.Combobox(send_cmd_frame, textvariable=self.send_format_var, width=10,
                                        values=["PWM数据", "自定义十六进制", "自定义文本"])
            format_combo.pack(side=tk.LEFT, padx=(0, 10))
            format_combo.state(['readonly'])
            format_combo.bind('<<ComboboxSelected>>', self.on_send_format_changed)
            
            # 发送按钮和选项
            self.send_button = ttk.Button(send_cmd_frame, text="发送数据", command=self.send_serial_data, state=tk.DISABLED)
            self.send_button.pack(side=tk.LEFT, padx=(0, 10))
            
            self.verify_echo_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(send_cmd_frame, text="验证回显", variable=self.verify_echo_var).pack(side=tk.LEFT, padx=(0, 10))
            
            self.show_timing_var = tk.BooleanVar(value=True)  # 默认显示时间
            ttk.Checkbutton(send_cmd_frame, text="显示耗时", variable=self.show_timing_var).pack(side=tk.LEFT)
            
            # 自定义数据输入区域（默认隐藏）
            self.custom_data_frame = ttk.Frame(send_control_frame)
            self.custom_data_frame.grid(row=4, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(5, 0))
            self.custom_data_frame.grid_remove()
            
            ttk.Label(self.custom_data_frame, text="自定义数据:").pack(side=tk.LEFT, padx=(0, 5))
            self.custom_data_var = tk.StringVar()
            self.custom_data_entry = ttk.Entry(self.custom_data_frame, textvariable=self.custom_data_var, width=50)
            self.custom_data_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
            
            # 发送统计显示
            stats_frame = ttk.Frame(send_control_frame)
            stats_frame.grid(row=5, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(10, 0))
            
            self.send_stats_label = ttk.Label(stats_frame, text="发送统计: 总计0次 | 错误0次 | 最后发送耗时: 0ms")
            self.send_stats_label.pack(side=tk.LEFT)
            
            ttk.Button(stats_frame, text="重置统计", command=self.reset_send_stats).pack(side=tk.RIGHT, padx=(10, 0))
            ttk.Button(stats_frame, text="查看时间历史", command=self.show_timing_history).pack(side=tk.RIGHT)
        
        # 状态信息
        status_frame = ttk.LabelFrame(left_frame, text="状态信息", padding="10")
        status_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="等待连接相机...")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0")
        self.fps_label.grid(row=1, column=0, sticky=tk.W)
        
        self.save_path_label = ttk.Label(status_frame, text=f"保存路径: {self.save_path}", wraplength=300)
        self.save_path_label.grid(row=2, column=0, sticky=tk.W)
        
        self.roi_info_label = ttk.Label(status_frame, text="ROI: 未设置", wraplength=300)
        self.roi_info_label.grid(row=3, column=0, sticky=tk.W)
        
        # === 中间视频显示区域 ===
        video_display_frame = ttk.LabelFrame(middle_frame, text="视频预览", padding="10")
        video_display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = ttk.Label(video_display_frame, text="视频预览区域\n(640x480)", 
                                     background="black", foreground="white",
                                     anchor="center", font=("Arial", 16))
        self.video_label.grid(row=0, column=0, padx=5, pady=5)
        
        # === 串口数据接收区域 ===
        if SERIAL_AVAILABLE:
            serial_data_frame = ttk.LabelFrame(middle_frame, text="串口数据接收区", padding="10")
            serial_data_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
            
            # 创建滚动文本框
            self.serial_text = scrolledtext.ScrolledText(serial_data_frame, height=10, width=80, 
                                                         font=("Consolas", 9))
            self.serial_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # 数据接收控制
            serial_control_row = ttk.Frame(serial_data_frame)
            serial_control_row.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
            
            # 显示格式选择
            ttk.Label(serial_control_row, text="显示格式:").pack(side=tk.LEFT, padx=(0, 5))
            self.display_format_var = tk.StringVar(value="文本")
            format_combo = ttk.Combobox(serial_control_row, textvariable=self.display_format_var, width=8,
                                        values=["文本", "十六进制", "十进制"])
            format_combo.pack(side=tk.LEFT, padx=(0, 10))
            format_combo.state(['readonly'])
            
            # 自动滚动选项
            self.auto_scroll_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(serial_control_row, text="自动滚动", variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=(0, 10))
            
            # 时间戳选项
            self.timestamp_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(serial_control_row, text="显示时间戳", variable=self.timestamp_var).pack(side=tk.LEFT, padx=(0, 10))
            
            # 数据统计
            self.data_count_label = ttk.Label(serial_control_row, text="接收: 0 字节")
            self.data_count_label.pack(side=tk.RIGHT)
        
        # === 右侧统计面板内容 ===
        # 灰度统计信息
        stats_frame = ttk.LabelFrame(right_frame, text="实时灰度统计", padding="10")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 统计数值显示
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(stats_grid, text="最小值:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.min_value_label = ttk.Label(stats_grid, text="0", font=("Arial", 10))
        self.min_value_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(stats_grid, text="最大值:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.max_value_label = ttk.Label(stats_grid, text="0", font=("Arial", 10))
        self.max_value_label.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(stats_grid, text="平均值:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        self.mean_value_label = ttk.Label(stats_grid, text="0", font=("Arial", 10))
        self.mean_value_label.grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(stats_grid, text="动态范围:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky=tk.W, padx=(0, 10))
        self.range_value_label = ttk.Label(stats_grid, text="0", font=("Arial", 10))
        self.range_value_label.grid(row=3, column=1, sticky=tk.W)
        
        # 添加说明文字
        note_label = ttk.Label(stats_frame, text="(实时显示当前帧的灰度统计信息)", font=("Arial", 8), foreground="gray")
        note_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # 灰度分布柱形图
        histogram_frame = ttk.LabelFrame(right_frame, text="灰度分布直方图 (可选)", padding="10")
        histogram_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建matplotlib图形
        self.fig = Figure(figsize=(5, 4), dpi=80, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('灰度值')
        self.ax.set_ylabel('像素数量')
        self.ax.set_title('实时灰度分布')
        self.ax.grid(True, alpha=0.3)
        
        # 创建canvas
        self.canvas = FigureCanvasTkAgg(self.fig, histogram_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 初始化空的直方图
        self.hist_bars = None
        self.update_empty_histogram()
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0)  # 左侧面板固定宽度
        main_frame.columnconfigure(1, weight=1)  # 中间视频区域可伸缩
        main_frame.columnconfigure(2, weight=0)  # 右侧统计面板固定宽度
        main_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(6, weight=1)  # 更新：状态信息行号
        middle_frame.columnconfigure(0, weight=1)
        middle_frame.rowconfigure(0, weight=2)  # 视频区域占更多空间
        if SERIAL_AVAILABLE:
            middle_frame.rowconfigure(1, weight=1)  # 串口接收区域
            serial_data_frame.columnconfigure(0, weight=1)
            serial_data_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        video_display_frame.columnconfigure(0, weight=1)
        video_display_frame.rowconfigure(0, weight=1)
        histogram_frame.columnconfigure(0, weight=1)
        histogram_frame.rowconfigure(0, weight=1)
    
    # === ROI可视化拖拽控制相关方法 ===
    def update_roi_canvas(self):
        """更新ROI预览Canvas"""
        try:
            # 清空Canvas
            self.roi_canvas.delete("all")
            
            # 获取传感器尺寸和当前ROI信息
            if not self.camera:
                # 如果相机未连接，使用默认值
                sensor_width = 1920
                sensor_height = 1080
                roi_x = int(self.roi_x_var.get()) if self.roi_x_var.get().isdigit() else 0
                roi_y = int(self.roi_y_var.get()) if self.roi_y_var.get().isdigit() else 0
                roi_width = int(self.width_var.get()) if self.width_var.get().isdigit() else 640
                roi_height = int(self.height_var.get()) if self.height_var.get().isdigit() else 480
                binning = int(self.binning_var.get()) if self.binning_var.get().isdigit() else 1
            else:
                # 获取实际传感器信息
                sensor_width = self.max_width
                sensor_height = self.max_height
                roi_x = int(self.roi_x_var.get()) if self.roi_x_var.get().isdigit() else 0
                roi_y = int(self.roi_y_var.get()) if self.roi_y_var.get().isdigit() else 0
                roi_width = int(self.width_var.get()) if self.width_var.get().isdigit() else 640
                roi_height = int(self.height_var.get()) if self.height_var.get().isdigit() else 480
                binning = int(self.binning_var.get()) if self.binning_var.get().isdigit() else 1
            
            # 考虑binning的影响
            effective_sensor_width = sensor_width // binning
            effective_sensor_height = sensor_height // binning
            
            # 计算缩放比例，保持宽高比
            scale_x = (self.roi_canvas_width - 20) / effective_sensor_width
            scale_y = (self.roi_canvas_height - 20) / effective_sensor_height
            scale = min(scale_x, scale_y)
            
            self.canvas_scale_x = effective_sensor_width / (self.roi_canvas_width - 20)
            self.canvas_scale_y = effective_sensor_height / (self.roi_canvas_height - 20)
            
            # 计算Canvas中心偏移
            canvas_center_x = self.roi_canvas_width // 2
            canvas_center_y = self.roi_canvas_height // 2
            scaled_sensor_width = effective_sensor_width * scale
            scaled_sensor_height = effective_sensor_height * scale
            
            # 绘制传感器边界（灰色矩形）
            sensor_left = canvas_center_x - scaled_sensor_width // 2
            sensor_top = canvas_center_y - scaled_sensor_height // 2
            sensor_right = sensor_left + scaled_sensor_width
            sensor_bottom = sensor_top + scaled_sensor_height
            
            self.sensor_rect_id = self.roi_canvas.create_rectangle(
                sensor_left, sensor_top, sensor_right, sensor_bottom,
                outline="black", fill="white", width=2
            )
            
            # 添加传感器标签
            self.roi_canvas.create_text(
                canvas_center_x, sensor_top - 10,
                text=f"传感器 {effective_sensor_width}×{effective_sensor_height} (Binning {binning})",
                font=("Arial", 8), anchor="center"
            )
            
            # 计算ROI在Canvas中的位置
            roi_canvas_x = sensor_left + roi_x * scale
            roi_canvas_y = sensor_top + roi_y * scale
            roi_canvas_width = roi_width * scale
            roi_canvas_height = roi_height * scale
            
            # 绘制ROI矩形（蓝色边框）
            self.roi_rect_id = self.roi_canvas.create_rectangle(
                roi_canvas_x, roi_canvas_y,
                roi_canvas_x + roi_canvas_width, roi_canvas_y + roi_canvas_height,
                outline="blue", fill="lightblue", width=2, stipple="gray25"
            )
            
            # 绘制可拖拽的圆点（红色，在ROI中心）
            circle_x = roi_canvas_x + roi_canvas_width // 2
            circle_y = roi_canvas_y + roi_canvas_height // 2
            circle_radius = 6
            
            self.roi_circle_id = self.roi_canvas.create_oval(
                circle_x - circle_radius, circle_y - circle_radius,
                circle_x + circle_radius, circle_y + circle_radius,
                outline="darkred", fill="red", width=2
            )
            
            # 添加ROI标签
            self.roi_canvas.create_text(
                circle_x, circle_y - 20,
                text=f"ROI {roi_width}×{roi_height}",
                font=("Arial", 8), anchor="center", fill="darkblue"
            )
            
            # 更新位置信息显示
            self.update_roi_position_display()
            
        except Exception as e:
            print(f"更新ROI Canvas失败: {e}")
    
    def update_roi_position_display(self):
        """更新ROI位置信息显示"""
        try:
            roi_x = self.roi_x_var.get()
            roi_y = self.roi_y_var.get()
            roi_width = self.width_var.get()
            roi_height = self.height_var.get()
            binning = self.binning_var.get()
            
            self.roi_pos_x_label.config(text=str(roi_x))
            self.roi_pos_y_label.config(text=str(roi_y))
            self.roi_width_label.config(text=str(roi_width))
            self.roi_height_label.config(text=str(roi_height))
            self.current_binning_label.config(text=str(binning))
            
            if hasattr(self, 'max_width') and hasattr(self, 'max_height'):
                self.sensor_size_label.config(text=f"{self.max_width}×{self.max_height}")
        except Exception as e:
            print(f"更新ROI位置显示失败: {e}")
    
    def canvas_to_sensor_coords(self, canvas_x, canvas_y):
        """将Canvas坐标转换为传感器坐标"""
        try:
            # 获取传感器信息
            if not self.camera:
                sensor_width = 1920
                sensor_height = 1080
            else:
                sensor_width = self.max_width
                sensor_height = self.max_height
            
            binning = int(self.binning_var.get()) if self.binning_var.get().isdigit() else 1
            effective_sensor_width = sensor_width // binning
            effective_sensor_height = sensor_height // binning
            
            # 计算Canvas中传感器区域的位置
            scale_x = (self.roi_canvas_width - 20) / effective_sensor_width
            scale_y = (self.roi_canvas_height - 20) / effective_sensor_height
            scale = min(scale_x, scale_y)
            
            canvas_center_x = self.roi_canvas_width // 2
            canvas_center_y = self.roi_canvas_height // 2
            scaled_sensor_width = effective_sensor_width * scale
            scaled_sensor_height = effective_sensor_height * scale
            
            sensor_left = canvas_center_x - scaled_sensor_width // 2
            sensor_top = canvas_center_y - scaled_sensor_height // 2
            
            # 转换坐标
            sensor_x = (canvas_x - sensor_left) / scale
            sensor_y = (canvas_y - sensor_top) / scale
            
            return int(sensor_x), int(sensor_y)
        except Exception as e:
            print(f"坐标转换失败: {e}")
            return 0, 0
    
    def on_roi_canvas_click(self, event):
        """处理Canvas鼠标点击事件"""
        try:
            # 检查是否点击在圆点上
            clicked_item = self.roi_canvas.find_closest(event.x, event.y)[0]
            
            if clicked_item == self.roi_circle_id:
                # 开始拖拽
                self.roi_dragging = True
                self.roi_drag_start_x = event.x
                self.roi_drag_start_y = event.y
                self.roi_canvas.config(cursor="hand2")
            else:
                # 点击在其他地方，移动ROI中心到点击位置
                self.move_roi_to_canvas_position(event.x, event.y)
        except Exception as e:
            print(f"Canvas点击处理失败: {e}")
    
    def on_roi_canvas_drag(self, event):
        """处理Canvas鼠标拖拽事件"""
        try:
            if self.roi_dragging:
                # 计算拖拽偏移
                dx = event.x - self.roi_drag_start_x
                dy = event.y - self.roi_drag_start_y
                
                # 更新圆点位置（临时显示）
                self.roi_canvas.move(self.roi_circle_id, dx, dy)
                self.roi_canvas.move(self.roi_rect_id, dx, dy)
                
                # 更新拖拽起始点
                self.roi_drag_start_x = event.x
                self.roi_drag_start_y = event.y
        except Exception as e:
            print(f"Canvas拖拽处理失败: {e}")
    
    def on_roi_canvas_release(self, event):
        """处理Canvas鼠标释放事件"""
        try:
            if self.roi_dragging:
                self.roi_dragging = False
                self.roi_canvas.config(cursor="")
                
                # 计算新的ROI位置
                self.move_roi_to_canvas_position(event.x, event.y)
        except Exception as e:
            print(f"Canvas释放处理失败: {e}")
    
    def on_roi_canvas_double_click(self, event):
        """处理Canvas双击事件（居中ROI）"""
        try:
            self.center_roi()
        except Exception as e:
            print(f"Canvas双击处理失败: {e}")
    
    def move_roi_to_canvas_position(self, canvas_x, canvas_y):
        """移动ROI中心到指定的Canvas位置"""
        try:
            # 将Canvas坐标转换为传感器坐标
            sensor_x, sensor_y = self.canvas_to_sensor_coords(canvas_x, canvas_y)
            
            # 获取当前ROI尺寸
            roi_width = int(self.width_var.get()) if self.width_var.get().isdigit() else 640
            roi_height = int(self.height_var.get()) if self.height_var.get().isdigit() else 480
            
            # 计算ROI左上角位置（使点击位置成为ROI中心）
            new_roi_x = sensor_x - roi_width // 2
            new_roi_y = sensor_y - roi_height // 2
            
            # 获取边界限制
            binning = int(self.binning_var.get()) if self.binning_var.get().isdigit() else 1
            max_x = (self.max_width // binning) - roi_width
            max_y = (self.max_height // binning) - roi_height
            
            # 限制在有效范围内
            new_roi_x = max(0, min(new_roi_x, max_x))
            new_roi_y = max(0, min(new_roi_y, max_y))
            
            # 更新UI变量
            self.roi_x_var.set(str(new_roi_x))
            self.roi_y_var.set(str(new_roi_y))
            
            # 应用ROI位置（如果相机已连接）
            if self.camera:
                self.apply_roi_move(new_roi_x, new_roi_y)
            
            # 更新Canvas显示
            self.update_roi_canvas()
            
            print(f"ROI移动到: ({new_roi_x}, {new_roi_y})")
            
        except Exception as e:
            print(f"移动ROI到Canvas位置失败: {e}")
    
    def apply_roi_from_canvas(self):
        """从Canvas应用ROI设置"""
        try:
            self.apply_roi()
            self.update_roi_canvas()
        except Exception as e:
            print(f"从Canvas应用ROI失败: {e}")
    
    def apply_roi_move(self, new_x, new_y):
        """应用ROI移动（优化版本，减少中断）"""
        if not self.camera:
            return
        
        try:
            # 如果正在捕获，使用更快的移动方式
            was_capturing = self.is_capturing
            
            if was_capturing:
                # 尝试直接设置ROI位置而不停止捕获
                try:
                    self.camera.set_roi_start_position(new_x, new_y)
                except:
                    # 如果直接设置失败，才停止并重新开始捕获
                    self.camera.stop_video_capture()
                    time.sleep(0.01)  # 更短的等待时间
                    self.camera.set_roi_start_position(new_x, new_y)
                    self.camera.start_video_capture()
            else:
                # 未在捕获时，直接设置
                self.camera.set_roi_start_position(new_x, new_y)
            
            # 更新状态显示
            roi_info = self.get_current_roi_info()
            self.roi_info_label.config(text=roi_info)
            
        except Exception as e:
            print(f"应用ROI移动失败: {e}")
            # 如果失败，尝试恢复捕获状态
            if was_capturing and not self.is_capturing:
                try:
                    self.camera.start_video_capture()
                except:
                    pass
    
    # === 文件保存和路径相关方法 ===
    def get_smart_save_path(self):
        """智能选择保存路径，避免U盘，选择大容量硬盘"""
        try:
            if not PSUTIL_AVAILABLE:
                # 如果psutil不可用，使用简化方法
                return self.get_simple_save_path()
            
            import psutil
            import platform
            
            suitable_drives = []
            
            if platform.system() == "Windows":
                # 获取所有磁盘分区
                partitions = psutil.disk_partitions()
                
                for partition in partitions:
                    try:
                        # 跳过C盘
                        if partition.mountpoint.upper().startswith('C:'):
                            continue
                        
                        # 跳过光驱等
                        if 'cdrom' in partition.opts or partition.fstype == '':
                            continue
                        
                        # 获取磁盘使用情况
                        disk_usage = psutil.disk_usage(partition.mountpoint)
                        total_gb = disk_usage.total / (1024**3)
                        
                        # 只选择大于128GB的磁盘（避免U盘）
                        if total_gb > 128:
                            suitable_drives.append({
                                'path': partition.mountpoint,
                                'size_gb': total_gb,
                                'free_gb': disk_usage.free / (1024**3)
                            })
                            print(f"发现合适的驱动器: {partition.mountpoint} ({total_gb:.1f}GB)")
                        else:
                            print(f"跳过小容量驱动器: {partition.mountpoint} ({total_gb:.1f}GB)")
                    
                    except (PermissionError, FileNotFoundError):
                        continue
            else:
                # Linux/Mac系统的处理
                return self.get_simple_save_path()
            
            # 选择最大的可用磁盘
            if suitable_drives:
                best_drive = max(suitable_drives, key=lambda x: x['free_gb'])
                base_path = best_drive['path']
                print(f"选择驱动器: {base_path} (可用空间: {best_drive['free_gb']:.1f}GB)")
            else:
                # 如果没有找到合适的驱动器，使用简化方法
                return self.get_simple_save_path()
            
            # 创建zwodata文件夹
            zwo_path = os.path.join(base_path, "zwodata")
            if not os.path.exists(zwo_path):
                os.makedirs(zwo_path)
                print(f"创建zwodata文件夹: {zwo_path}")
            
            return zwo_path
            
        except Exception as e:
            print(f"智能路径选择失败: {e}")
            return self.get_simple_save_path()
    
    def get_simple_save_path(self):
        """简化的保存路径选择（当psutil不可用时）"""
        try:
            import platform
            
            if platform.system() == "Windows":
                # 在Windows上尝试常见的数据盘
                for drive in ['D:', 'E:', 'F:']:
                    drive_path = drive + os.sep
                    if os.path.exists(drive_path):
                        zwo_path = os.path.join(drive_path, "zwodata")
                        if not os.path.exists(zwo_path):
                            os.makedirs(zwo_path)
                        print(f"使用简化模式选择路径: {zwo_path}")
                        return zwo_path
            
            # 如果都不可用，使用当前目录下的zwodata
            zwo_path = os.path.join(os.getcwd(), "zwodata")
            if not os.path.exists(zwo_path):
                os.makedirs(zwo_path)
            print(f"使用当前目录: {zwo_path}")
            return zwo_path
            
        except Exception as e:
            print(f"简化路径选择失败: {e}")
            return os.getcwd()
    
    def create_session_folder(self):
        """创建新的会话文件夹（带时间戳）"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_folder = os.path.join(self.save_path, f"session_{timestamp}")
            
            if not os.path.exists(session_folder):
                os.makedirs(session_folder)
                print(f"创建会话文件夹: {session_folder}")
            
            self.current_session_folder = session_folder
            return session_folder
        except Exception as e:
            print(f"创建会话文件夹失败: {e}")
            return self.save_path
    
    def get_current_session_folder(self):
        """获取当前会话文件夹，如果不存在则创建"""
        if self.current_session_folder is None or not os.path.exists(self.current_session_folder):
            return self.create_session_folder()
        return self.current_session_folder
    
    def update_file_index(self, event=None):
        """更新文件索引"""
        try:
            new_index = int(self.index_var.get())
            if new_index <= 0:
                raise ValueError("索引必须大于0")
            self.file_index = new_index
            print(f"文件索引更新为: {self.file_index}")
        except ValueError as e:
            messagebox.showerror("错误", f"索引设置错误: {e}")
            self.index_var.set(str(self.file_index))
    
    def reset_file_index(self):
        """重置文件索引并创建新的会话文件夹"""
        self.file_index = 1
        self.index_var.set("1")
        self.create_session_folder()
        print(f"文件索引已重置为1，新会话文件夹: {self.current_session_folder}")
    
    def auto_select_smart_path(self):
        """重新智能选择保存路径"""
        new_path = self.get_smart_save_path()
        self.save_path = new_path
        self.current_session_folder = None  # 重置会话文件夹
        self.save_path_label.config(text=f"保存路径: {self.save_path}")
        messagebox.showinfo("成功", f"已重新选择智能保存路径:\n{new_path}")
    
    def open_zwodata_folder(self):
        """打开当前会话的时间戳子文件夹"""
        try:
            import subprocess
            import platform
            
            # 获取当前会话文件夹路径
            folder_path = self.get_current_session_folder()
            
            # 检查文件夹是否存在
            if not os.path.exists(folder_path):
                messagebox.showerror("错误", f"会话文件夹不存在:\n{folder_path}")
                return
            
            # 根据操作系统选择打开方式
            system = platform.system()
            if system == "Windows":
                # Windows系统
                os.startfile(folder_path)
            elif system == "Darwin":
                # macOS系统
                subprocess.run(["open", folder_path])
            else:
                # Linux系统
                subprocess.run(["xdg-open", folder_path])
            
            print(f"已打开当前会话文件夹: {folder_path}")
            
        except Exception as e:
            error_msg = f"打开会话文件夹失败: {str(e)}"
            messagebox.showerror("错误", error_msg)
            print(error_msg)
    
    def select_save_path(self):
        """选择保存路径"""
        path = filedialog.askdirectory(initialdir=self.save_path)
        if path:
            self.save_path = path
            self.current_session_folder = None  # 重置会话文件夹
            self.save_path_label.config(text=f"保存路径: {self.save_path}")
    
    # === 新增的串口优化方法 ===
    def clear_hardware_buffers(self):
        """清空串口硬件缓冲区"""
        if not self.is_serial_open or not self.serial_port:
            messagebox.showwarning("警告", "串口未打开")
            return
        
        try:
            # 清空输入缓冲区
            self.serial_port.reset_input_buffer()
            # 清空输出缓冲区
            self.serial_port.reset_output_buffer()
            
            # 清空软件缓冲区
            self.serial_buffer = []
            self.receive_byte_count = 0
            
            # 更新显示
            self.update_buffer_status()
            self.add_to_serial_display("[系统] 硬件缓冲区已清空", "info")
            
            print("硬件缓冲区已清空")
            
        except Exception as e:
            messagebox.showerror("错误", f"清空硬件缓冲区失败: {e}")
    
    def update_buffer_status(self):
        """更新缓冲区状态显示"""
        if SERIAL_AVAILABLE:
            buffer_size = len(self.serial_buffer)
            hw_buffer_size = 0
            
            if self.is_serial_open and self.serial_port:
                try:
                    hw_buffer_size = self.serial_port.in_waiting
                except:
                    pass
            
            status_text = f"缓冲区: {buffer_size}/{self.max_buffer_size} | 硬件: {hw_buffer_size}"
            self.buffer_status_label.config(text=status_text)
    
    def show_timing_history(self):
        """显示发送时间历史记录"""
        if not self.send_stats['send_history']:
            messagebox.showinfo("信息", "暂无时间历史记录")
            return
        
        # 创建新窗口显示时间历史
        history_window = tk.Toplevel(self.root)
        history_window.title("发送时间历史记录")
        history_window.geometry("600x400")
        
        # 创建文本框显示历史记录
        text_widget = scrolledtext.ScrolledText(history_window, font=("Consolas", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 添加统计信息
        history = self.send_stats['send_history']
        text_widget.insert(tk.END, f"发送时间历史记录 (最近{len(history)}次)\n")
        text_widget.insert(tk.END, "=" * 50 + "\n")
        
        if history:
            avg_time = sum(history) / len(history)
            min_time = min(history)
            max_time = max(history)
            
            text_widget.insert(tk.END, f"平均时间: {avg_time:.2f} ms\n")
            text_widget.insert(tk.END, f"最短时间: {min_time:.2f} ms\n")
            text_widget.insert(tk.END, f"最长时间: {max_time:.2f} ms\n")
            text_widget.insert(tk.END, "=" * 50 + "\n\n")
            
            # 显示详细历史
            for i, timing in enumerate(reversed(history[-50:]), 1):  # 最近50次
                text_widget.insert(tk.END, f"第{len(history)-i+1}次: {timing:.2f} ms\n")
        
        text_widget.config(state=tk.DISABLED)
    
    # === 串口发送相关方法（优化版本）===
    def on_control_mode_changed(self, event=None):
        """控制模式改变时的处理"""
        mode = self.control_mode_var.get()
        if mode == "全通道模式":
            self.pwm_frame.grid()
            self.white_light_frame.grid_remove()
        elif mode == "白光模式":
            self.pwm_frame.grid_remove()
            self.white_light_frame.grid()
            # 切换到白光模式时，自动清空其他通道
            self.sync_white_light_to_pwm()
    
    def validate_white_light_values(self, event=None):
        """验证白光模式PWM值的有效性"""
        try:
            for ch_index, var in self.white_light_vars.items():
                value = int(var.get())
                if value < 0 or value > 100:
                    var.set("0")
                    messagebox.showwarning("警告", f"CH{ch_index+1}的值必须在0-100之间")
                    return False
            return True
        except ValueError:
            messagebox.showerror("错误", "白光PWM值必须是整数")
            return False
    
    def on_white_light_scale_changed(self, ch_index, value):
        """白光滑块值改变时的处理"""
        try:
            # 更新对应的输入框（整数值）
            self.white_light_vars[ch_index].set(str(int(float(value))))
        except:
            pass
    
    def sync_white_light_entry_from_scale(self, ch_index, event):
        """从滑块同步到输入框（鼠标释放时）"""
        try:
            scale_widget = event.widget
            value = scale_widget.get()
            self.white_light_vars[ch_index].set(str(int(value)))
            # 同步到全通道PWM数组
            self.sync_white_light_to_pwm()
        except:
            pass
    
    def sync_white_light_scale_from_entry(self, ch_index):
        """从输入框同步到滑块"""
        try:
            value = int(self.white_light_vars[ch_index].get())
            if 0 <= value <= 100:
                # 找到对应的滑块并更新
                for widget in self.white_light_frame.winfo_children():
                    if isinstance(widget, ttk.Scale):
                        # 通过位置判断是否是对应的滑块
                        grid_info = widget.grid_info()
                        if grid_info:
                            row = int(grid_info.get('row', 0))
                            col = int(grid_info.get('column', 0))
                            # 根据布局计算是否匹配
                            ch_list = [6, 7, 10, 11]  # CH7, CH8, CH11, CH12对应的索引
                            ch_positions = {6: (0, 2), 7: (0, 5), 10: (1, 2), 11: (1, 5)}
                            if ch_index in ch_positions and ch_positions[ch_index] == (row, col):
                                widget.set(value)
                                break
                # 同步到全通道PWM数组
                self.sync_white_light_to_pwm()
        except:
            pass
    
    def sync_white_light_to_pwm(self):
        """将白光模式的值同步到全通道PWM数组"""
        try:
            # 先清空所有PWM通道
            for i, var in enumerate(self.pwm_vars):
                var.set("0")
            
            # 设置白光通道的值
            for ch_index, var in self.white_light_vars.items():
                try:
                    value = int(var.get())
                    if 0 <= value <= 100:
                        self.pwm_vars[ch_index].set(str(value))
                except:
                    pass
        except Exception as e:
            print(f"同步白光到PWM失败: {e}")
    
    def clear_all_white_light(self):
        """清空所有白光通道"""
        for var in self.white_light_vars.values():
            var.set("0")
        self.sync_white_light_to_pwm()
    
    def set_white_light_preset(self, preset_type):
        """设置白光预设"""
        presets = {
            "warm": {"6": 80, "7": 20, "10": 0, "11": 0},      # 暖白光：主要CH7
            "cool": {"6": 0, "7": 0, "10": 80, "11": 20},      # 冷白光：主要CH11
            "mixed": {"6": 50, "7": 30, "10": 30, "11": 50}    # 混合光：各通道均衡
        }
        
        if preset_type in presets:
            preset = presets[preset_type]
            for ch_index, var in self.white_light_vars.items():
                value = preset.get(str(ch_index), 0)
                var.set(str(value))
            self.sync_white_light_to_pwm()
            print(f"已设置{preset_type}白光预设")
    
    def on_send_format_changed(self, event=None):
        """发送格式改变时的处理"""
        format_type = self.send_format_var.get()
        if format_type == "PWM数据":
            self.custom_data_frame.grid_remove()
        else:
            self.custom_data_frame.grid()
            if format_type == "自定义十六进制":
                self.custom_data_var.set("00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F")
            elif format_type == "自定义文本":
                self.custom_data_var.set("Hello STM32!")
    
    def validate_pwm_values(self, event=None):
        """验证PWM值的有效性"""
        try:
            for i, var in enumerate(self.pwm_vars):
                value = int(var.get())
                if value < 0 or value > 100:
                    var.set("0")
                    messagebox.showwarning("警告", f"通道{i+1}的值必须在0-100之间")
                    return False
            return True
        except ValueError:
            messagebox.showerror("错误", "PWM值必须是整数")
            return False
    
    def set_all_pwm(self, value):
        """设置所有PWM通道为指定值"""
        for var in self.pwm_vars:
            var.set(str(value))
    
    def set_example_pwm(self):
        """设置示例PWM数据"""
        example_values = [25, 50, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i, var in enumerate(self.pwm_vars):
            var.set(str(example_values[i]))
    
    def prepare_send_data(self):
        """准备要发送的数据"""
        format_type = self.send_format_var.get()
        
        if format_type == "PWM数据":
            # 验证PWM值
            if not self.validate_pwm_values():
                return None
            
            # 如果是白光模式，先同步数据
            if hasattr(self, 'control_mode_var') and self.control_mode_var.get() == "白光模式":
                if not self.validate_white_light_values():
                    return None
                self.sync_white_light_to_pwm()
            
            # 获取PWM值并转换为字节数据
            pwm_values = []
            for var in self.pwm_vars:
                try:
                    value = int(var.get())
                    pwm_values.append(max(0, min(100, value)))  # 确保在0-100范围内
                except ValueError:
                    pwm_values.append(0)
            
            return bytes(pwm_values)
        
        elif format_type == "自定义十六进制":
            try:
                hex_str = self.custom_data_var.get().replace(" ", "").replace(",", "")
                if len(hex_str) % 2 != 0:
                    raise ValueError("十六进制字符串长度必须是偶数")
                return bytes.fromhex(hex_str)
            except ValueError as e:
                messagebox.showerror("错误", f"十六进制格式错误: {e}")
                return None
        
        elif format_type == "自定义文本":
            try:
                return self.custom_data_var.get().encode('utf-8')
            except Exception as e:
                messagebox.showerror("错误", f"文本编码错误: {e}")
                return None
        
        return None
    
    def send_serial_data(self):
        """发送串口数据（优化版本）"""
        if not self.is_serial_open or not self.serial_port:
            messagebox.showerror("错误", "串口未打开")
            return
        
        # 准备数据
        data_to_send = self.prepare_send_data()
        if data_to_send is None:
            return
        
        try:
            # 发送前清空缓冲区（如果启用）
            if self.clear_before_send.get():
                self.serial_port.reset_input_buffer()
                self.serial_port.reset_output_buffer()
                if not self.performance_mode.get():
                    self.add_to_serial_display("[系统] 发送前已清空缓冲区", "info")
            
            # 记录发送时间
            start_time = time.perf_counter()
            
            # 发送数据
            bytes_sent = self.serial_port.write(data_to_send)
            
            # 强制刷新输出缓冲区
            self.serial_port.flush()
            
            # 记录发送完成时间
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000
            
            # 更新发送统计
            self.send_stats['total_sent'] += 1
            self.send_stats['last_send_time'] = elapsed_ms
            
            # 记录时间历史（保持最近100次）
            self.send_stats['send_history'].append(elapsed_ms)
            if len(self.send_stats['send_history']) > 100:
                self.send_stats['send_history'] = self.send_stats['send_history'][-100:]
            
            # 在接收区显示发送的数据（如果不是性能模式）
            if not self.performance_mode.get():
                self.display_sent_data(data_to_send)
            
            # 如果需要验证回显
            if self.verify_echo_var.get():
                self.handle_echo_verification(data_to_send, start_time)
            else:
                # 不验证回显时显示发送时间
                if self.show_timing_var.get() and not self.performance_mode.get():
                    self.add_to_serial_display(f"[时间] 发送耗时: {elapsed_ms:.2f} 毫秒", "info")
            
            # 更新统计显示
            self.update_send_stats_display()
            
            print(f"数据发送成功: {len(data_to_send)} 字节, 耗时: {elapsed_ms:.2f}ms")
            
        except serial.SerialException as e:
            self.send_stats['send_errors'] += 1
            self.update_send_stats_display()
            messagebox.showerror("错误", f"串口发送失败: {e}")
        except Exception as e:
            self.send_stats['send_errors'] += 1
            self.update_send_stats_display()
            messagebox.showerror("错误", f"发送数据失败: {e}")
    
    def handle_echo_verification(self, sent_data, start_time):
        """处理回显验证"""
        try:
            # 期望接收的数据长度
            expected_length = len(sent_data)
            if self.send_format_var.get() == "PWM数据":
                expected_length += 2  # 额外的'\r\n'
            
            # 设置较短的超时时间
            original_timeout = self.serial_port.timeout
            self.serial_port.timeout = 0.1  # 100ms超时
            
            # 读取回显数据
            response = self.serial_port.read(expected_length)
            end_time = time.perf_counter()
            
            # 恢复原始超时
            self.serial_port.timeout = original_timeout
            
            if response:
                self.process_echo_response(sent_data, response, start_time, end_time)
            else:
                elapsed_ms = (end_time - start_time) * 1000
                self.send_stats['send_errors'] += 1
                if not self.performance_mode.get():
                    self.add_to_serial_display(f"[发送] 警告: 在超时时间内未收到回显数据", "warning")
                    
                    if self.show_timing_var.get():
                        self.add_to_serial_display(f"[时间] 发送耗时: {elapsed_ms:.2f} 毫秒", "info")
        
        except Exception as e:
            print(f"回显验证失败: {e}")
    
    def display_sent_data(self, data):
        """在接收区显示发送的数据"""
        format_type = self.send_format_var.get()
        
        if format_type == "PWM数据":
            # 检查是否为白光模式
            if hasattr(self, 'control_mode_var') and self.control_mode_var.get() == "白光模式":
                # 白光模式：只显示激活的通道
                white_channels = {6: "CH7", 7: "CH8", 10: "CH11", 11: "CH12"}
                active_channels = []
                for i, value in enumerate(data[:16]):
                    if value > 0 and i in white_channels:
                        active_channels.append(f'{white_channels[i]}:{value}')
                
                if active_channels:
                    white_str = ', '.join(active_channels)
                    self.add_to_serial_display(f"[发送] 白光数据: {white_str}", "send")
                else:
                    self.add_to_serial_display(f"[发送] 白光数据: 全部关闭", "send")
            else:
                # 全通道模式：显示所有通道
                pwm_str = ', '.join([f'CH{i+1}:{data[i]}' for i in range(min(16, len(data)))])
                self.add_to_serial_display(f"[发送] PWM数据 ({len(data)} 字节): {pwm_str}", "send")
            
            # 始终显示十六进制数据
            self.add_to_serial_display(f"[发送] 十六进制: {data.hex(' ').upper()}", "send")
            
        elif format_type == "自定义十六进制":
            self.add_to_serial_display(f"[发送] 十六进制 ({len(data)} 字节): {data.hex(' ').upper()}", "send")
        elif format_type == "自定义文本":
            try:
                text_data = data.decode('utf-8')
                self.add_to_serial_display(f"[发送] 文本 ({len(data)} 字节): {text_data}", "send")
            except:
                self.add_to_serial_display(f"[发送] 数据 ({len(data)} 字节): {data.hex(' ').upper()}", "send")
    
    def process_echo_response(self, sent_data, response, start_time, end_time):
        """处理回显响应"""
        elapsed_ms = (end_time - start_time) * 1000
        self.send_stats['last_send_time'] = elapsed_ms
        
        if not self.performance_mode.get():
            # 显示接收到的原始数据
            self.add_to_serial_display(f"[接收] 回显数据 ({len(response)} 字节): {response.hex(' ').upper()}", "receive")
            
            # 如果是PWM数据，分离回显和结束符
            if self.send_format_var.get() == "PWM数据" and len(response) >= len(sent_data):
                echo_data = response[:len(sent_data)]
                if len(response) > len(sent_data):
                    end_chars = response[len(sent_data):]
                    self.add_to_serial_display(f"[接收] 结束符: {repr(end_chars.decode('utf-8', errors='replace'))}", "receive")
                
                # 验证回显数据
                if echo_data == sent_data:
                    self.add_to_serial_display("[验证] 成功: 回显数据与发送数据完全一致!", "success")
                else:
                    self.add_to_serial_display("[验证] 失败: 回显数据与发送数据不匹配!", "error")
                    self.send_stats['send_errors'] += 1
            else:
                # 其他格式的简单验证
                if len(response) >= len(sent_data) and response[:len(sent_data)] == sent_data:
                    self.add_to_serial_display("[验证] 成功: 回显数据匹配!", "success")
                else:
                    self.add_to_serial_display("[验证] 警告: 回显数据可能不完整或不匹配", "warning")
            
            # 显示时间统计
            if self.show_timing_var.get():
                self.add_to_serial_display(f"[时间] 发送到接收完整回显耗时: {elapsed_ms:.2f} 毫秒", "info")
    
    def add_to_serial_display(self, text, msg_type="normal"):
        """添加格式化文本到串口显示区"""
        # 性能模式下减少显示更新
        if self.performance_mode.get() and msg_type in ["send", "receive"]:
            return
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3] if self.timestamp_var.get() else ""
        
        # 根据消息类型设置颜色标记
        if msg_type == "send":
            prefix = "→ "
        elif msg_type == "receive":
            prefix = "← "
        elif msg_type == "success":
            prefix = "✓ "
        elif msg_type == "error":
            prefix = "✗ "
        elif msg_type == "warning":
            prefix = "⚠ "
        elif msg_type == "info":
            prefix = "ℹ "
        else:
            prefix = ""
        
        # 构建显示文本
        if timestamp:
            display_text = f"[{timestamp}] {prefix}{text}\n"
        else:
            display_text = f"{prefix}{text}\n"
        
        # 插入到文本框
        self.serial_text.insert(tk.END, display_text)
        
        # 自动滚动
        if self.auto_scroll_var.get():
            self.serial_text.see(tk.END)
        
        # 限制文本框内容长度
        if int(self.serial_text.index('end-1c').split('.')[0]) > 1000:
            self.serial_text.delete(1.0, "100.0")
    
    def update_send_stats_display(self):
        """更新发送统计显示"""
        stats_text = (f"发送统计: 总计{self.send_stats['total_sent']}次 | "
                      f"错误{self.send_stats['send_errors']}次 | "
                      f"最后发送耗时: {self.send_stats['last_send_time']:.2f}ms")
        self.send_stats_label.config(text=stats_text)
    
    def reset_send_stats(self):
        """重置发送统计"""
        self.send_stats = {
            'total_sent': 0,
            'last_send_time': 0,
            'send_errors': 0,
            'send_history': []
        }
        self.update_send_stats_display()
        self.add_to_serial_display("[系统] 发送统计已重置", "info")
    
    # === 串口接收相关方法（优化版本）===
    def refresh_serial_ports(self):
        """刷新串口端口列表"""
        if not SERIAL_AVAILABLE:
            return
        
        try:
            ports = serial.tools.list_ports.comports()
            
            # 获取最新插入的端口（通常是列表中的最后一个）
            latest_port = None
            port_names = []
            
            if ports:
                # 按设备描述信息排序，尝试找到最新的设备
                sorted_ports = self.sort_ports_by_insertion_time(ports)
                port_names = [port.device for port in sorted_ports]
                latest_port = port_names[0] if port_names else None
                
                print(f"发现串口: {port_names}")
                if latest_port:
                    print(f"检测到最新插入的串口: {latest_port}")
            
            self.port_combo['values'] = port_names
            
            if port_names:
                if not self.port_var.get() or self.port_var.get() not in port_names:
                    # 默认选择最新插入的串口
                    self.port_var.set(latest_port)
            else:
                self.port_var.set("")
                
        except Exception as e:
            print(f"刷新串口列表失败: {e}")
    
    def sort_ports_by_insertion_time(self, ports):
        """按设备插入时间排序，最新的在前"""
        try:
            # 方法1：根据设备路径和硬件信息判断
            # USB设备通常有更详细的信息，认为是较新的
            def get_port_priority(port):
                priority = 0
                
                # USB设备优先级更高（通常是较新插入的）
                if hasattr(port, 'vid') and port.vid is not None:
                    priority += 1000
                
                # 有产品描述的设备优先级较高
                if hasattr(port, 'product') and port.product:
                    priority += 100
                
                # 有制造商信息的设备优先级较高  
                if hasattr(port, 'manufacturer') and port.manufacturer:
                    priority += 50
                
                # 有序列号的设备优先级较高
                if hasattr(port, 'serial_number') and port.serial_number:
                    priority += 25
                
                # 提取COM端口号作为次要排序条件
                import re
                match = re.search(r'COM(\d+)', port.device.upper())
                com_num = int(match.group(1)) if match else 0
                priority += com_num  # COM号大的通常是较新的
                
                return priority
            
            # 按优先级降序排序
            return sorted(ports, key=get_port_priority, reverse=True)
            
        except Exception as e:
            print(f"端口排序失败: {e}")
            # 如果排序失败，返回原始列表的逆序（假设最后枚举的是最新的）
            return list(reversed(ports))
    
    def auto_open_latest_serial(self):
        """自动打开最新插入的串口"""
        if not SERIAL_AVAILABLE:
            return
        
        try:
            # 如果有可用串口，自动尝试打开最新插入的串口
            if self.port_var.get():
                print(f"尝试自动打开最新插入的串口: {self.port_var.get()}")
                self.open_serial_port(auto_open=True)
        except Exception as e:
            print(f"自动打开串口失败: {e}")
            # 自动打开失败不显示错误框，只在控制台打印信息
    
    def open_serial_port(self, auto_open=False):
        """打开串口"""
        if not SERIAL_AVAILABLE:
            if not auto_open:
                messagebox.showerror("错误", "pyserial库未安装，无法使用串口功能")
            return
        
        try:
            port = self.port_var.get()
            baudrate = int(self.baudrate_var.get())
            
            if not port:
                if not auto_open:
                    messagebox.showerror("错误", "请选择串口")
                return
            
            # 关闭已存在的连接
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            
            # 打开新连接（优化参数）
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,  # 较短的超时时间
                write_timeout=0.1,  # 写入超时
                # 禁用软件流控制
                xonxoff=False,
                # 禁用硬件流控制  
                rtscts=False,
                dsrdtr=False
            )
            
            self.is_serial_open = True
            
            # 清空缓冲区
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            # 启动接收线程
            self.serial_thread = threading.Thread(target=self.serial_receive_loop, daemon=True)
            self.serial_thread.start()
            
            # 更新UI状态
            self.serial_open_button.config(state=tk.DISABLED)
            self.serial_close_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self.serial_status_label.config(text=f"串口状态: 已连接 ({port}, {baudrate})", foreground="green")
            
            # 清空接收区
            self.serial_text.delete(1.0, tk.END)
            self.serial_buffer = []
            self.receive_byte_count = 0
            self.data_count_label.config(text="接收: 0 字节")
            self.update_buffer_status()
            
            # 重置发送统计
            self.reset_send_stats()
            
            print(f"串口已打开: {port}, 波特率: {baudrate}")
            
        except serial.SerialException as e:
            if not auto_open:
                messagebox.showerror("错误", f"打开串口失败: {str(e)}")
            else:
                print(f"自动打开串口失败: {str(e)}")
        except ValueError as e:
            if not auto_open:
                messagebox.showerror("错误", f"波特率设置错误: {str(e)}")
            else:
                print(f"波特率设置错误: {str(e)}")
        except Exception as e:
            if not auto_open:
                messagebox.showerror("错误", f"串口连接失败: {str(e)}")
            else:
                print(f"串口连接失败: {str(e)}")
    
    def close_serial_port(self):
        """关闭串口"""
        try:
            self.is_serial_open = False
            
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            
            # 更新UI状态
            self.serial_open_button.config(state=tk.NORMAL)
            self.serial_close_button.config(state=tk.DISABLED)
            self.send_button.config(state=tk.DISABLED)
            self.serial_status_label.config(text="串口状态: 未连接", foreground="red")
            
            # 更新缓冲区状态
            self.update_buffer_status()
            
            print("串口已关闭")
            
        except Exception as e:
            messagebox.showerror("错误", f"关闭串口失败: {str(e)}")
    
    def serial_receive_loop(self):
        """串口数据接收循环（优化版本）"""
        while self.is_serial_open and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    # 读取数据
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    if data:
                        # 添加到缓冲区
                        self.serial_buffer.extend(data)
                        self.receive_byte_count += len(data)
                        
                        # 限制缓冲区大小
                        if len(self.serial_buffer) > self.max_buffer_size:
                            self.serial_buffer = self.serial_buffer[-self.max_buffer_size:]
                        
                        # 自动清空缓冲区（如果启用且缓冲区满）
                        if self.auto_clear_buffer.get() and len(self.serial_buffer) >= self.max_buffer_size * 0.9:
                            self.serial_buffer = self.serial_buffer[-100:]  # 只保留最后100字节
                            if not self.performance_mode.get():
                                self.root.after(0, lambda: self.add_to_serial_display("[系统] 自动清空缓冲区", "info"))
                        
                        # 在主线程中更新显示
                        if not self.performance_mode.get():
                            self.root.after(0, lambda d=data: self.update_serial_display(d))
                        
                        # 更新缓冲区状态
                        self.root.after(0, self.update_buffer_status)
                
                time.sleep(0.005)  # 较短的延迟
                
            except Exception as e:
                print(f"串口接收错误: {e}")
                time.sleep(0.1)
    
    def update_serial_display(self, data):
        """更新串口数据显示（优化版本）"""
        try:
            display_format = self.display_format_var.get()
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3] if self.timestamp_var.get() else ""
            
            # 根据显示格式转换数据
            if display_format == "文本":
                try:
                    text_data = data.decode('utf-8', errors='replace')
                except:
                    text_data = data.decode('latin-1', errors='replace')
                display_data = text_data
            elif display_format == "十六进制":
                display_data = ' '.join([f'{b:02X}' for b in data])
            elif display_format == "十进制":
                display_data = ' '.join([str(b) for b in data])
            
            # 添加时间戳
            if timestamp:
                display_line = f"[{timestamp}] {display_data}"
            else:
                display_line = display_data
            
            # 如果数据不以换行符结尾，添加换行符
            if not display_line.endswith('\n'):
                display_line += '\n'
            
            # 插入到文本框
            self.serial_text.insert(tk.END, display_line)
            
            # 自动滚动到底部
            if self.auto_scroll_var.get():
                self.serial_text.see(tk.END)
            
            # 更新数据计数
            self.data_count_label.config(text=f"接收: {self.receive_byte_count} 字节")
            
            # 限制文本框内容长度
            if int(self.serial_text.index('end-1c').split('.')[0]) > 1000:
                self.serial_text.delete(1.0, "100.0")
            
        except Exception as e:
            print(f"串口显示更新错误: {e}")
    
    def clear_serial_data(self):
        """清空串口接收数据（增强版本）"""
        try:
            # 清空显示区
            self.serial_text.delete(1.0, tk.END)
            
            # 清空软件缓冲区
            self.serial_buffer = []
            self.receive_byte_count = 0
            
            # 清空硬件缓冲区（如果串口已打开）
            if self.is_serial_open and self.serial_port:
                self.serial_port.reset_input_buffer()
                self.serial_port.reset_output_buffer()
            
            # 更新显示
            self.data_count_label.config(text="接收: 0 字节")
            self.update_buffer_status()
            
            self.add_to_serial_display("[系统] 接收区和缓冲区已清空", "info")
            print("串口接收区和缓冲区已清空")
            
        except Exception as e:
            print(f"清空串口数据失败: {e}")
            messagebox.showerror("错误", f"清空串口数据失败: {e}")
    
    # === 相机相关方法 ===
    def initialize_camera(self):
        try:
            # 检查是否有连接的相机
            num_cameras = asi.get_num_cameras()
            if num_cameras == 0:
                self.status_label.config(text="错误：未检测到相机")
                messagebox.showerror("错误", "未检测到ZWO相机，请检查连接")
                return
            
            # 获取第一个相机
            camera_info = asi.list_cameras()
            self.status_label.config(text=f"检测到相机: {camera_info[0]}")
            
            # 打开相机
            self.camera = asi.Camera(0)
            
            # 获取相机属性
            camera_props = self.camera.get_camera_property()
            self.max_width = camera_props['MaxWidth']
            self.max_height = camera_props['MaxHeight']
            
            # 获取支持的binning值
            if 'SupportedBins' in camera_props:
                self.supported_bins = camera_props['SupportedBins']
            
            # 更新UI中的最大分辨率
            self.width_var.set(str(min(640, self.max_width)))
            self.height_var.set(str(min(480, self.max_height)))
            
            # 设置图像格式为RAW16
            self.camera.set_image_type(asi.ASI_IMG_RAW16)
            
            # 应用初始分辨率
            self.apply_resolution()
            
            # 优化相机设置以获得更高帧率
            try:
                # 设置高速模式（如果支持）
                controls = self.camera.get_controls()
                if 'HighSpeedMode' in controls:
                    self.camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, 1)
                
                # 关闭自动增益和自动曝光以获得稳定帧率
                if 'Gain' in controls:
                    self.camera.set_control_value(asi.ASI_GAIN, 0, auto=False)
                if 'Exposure' in controls:
                    self.camera.set_control_value(asi.ASI_EXPOSURE, 10000, auto=False)
                    
            except Exception as e:
                print(f"设置相机优化参数时出现警告: {e}")
            
            # 获取控制参数范围
            controls = self.camera.get_controls()
            
            if 'Exposure' in controls:
                exp_min = controls['Exposure']['MinValue']
                exp_max = controls['Exposure']['MaxValue']
                self.status_label.config(text=f"相机就绪 - 曝光范围: {exp_min}-{exp_max}μs")
            
            # 设置初始参数
            self.set_exposure()
            self.set_gain()
            
            # 更新ROI Canvas显示
            self.update_roi_canvas()
            
        except Exception as e:
            error_msg = f"相机初始化失败: {str(e)}"
            self.status_label.config(text=error_msg)
            messagebox.showerror("错误", error_msg)

    def format_exposure_time(self, microseconds):
        """将微秒转换为人类可读的字符串 (us, ms, s)"""
        try:
            microseconds = float(microseconds)
            if microseconds < 1000:
                return f"{int(microseconds)} µs"
            elif microseconds < 1_000_000:
                ms = microseconds / 1000
                if ms == int(ms):
                    return f"{int(ms)} ms"
                else:
                    return f"{ms:.2f} ms"
            else:
                s = microseconds / 1_000_000
                if s == int(s):
                    return f"{int(s)} s"
                else:
                    return f"{s:.2f} s"
        except (ValueError, TypeError):
            return ""

    def parse_exposure_input(self, input_str):
        """解析带单位（s, ms, us）的用户输入，并返回微秒"""
        input_str = input_str.lower().strip()
        try:
            if input_str.endswith('s') and not input_str.endswith('ms') and not input_str.endswith('us'):
                value = float(input_str[:-1].strip())
                return int(value * 1_000_000)
            elif input_str.endswith('ms'):
                value = float(input_str[:-2].strip())
                return int(value * 1_000)
            elif input_str.endswith('us') or input_str.endswith('µs'):
                value = float(input_str[:-2].strip())
                return int(value)
            else:
                # 如果没有单位，则假定为微秒
                return int(float(input_str))
        except (ValueError, TypeError):
            raise ValueError("无效的曝光时间格式。请使用数字或带单位（s, ms, us）的格式。")

    def set_exposure(self, event=None):
        if not self.camera:
            return
        
        try:
            # 解析用户输入
            exposure_value_us = self.parse_exposure_input(self.exposure_var.get())
            
            # 更新输入框以显示规范的微秒值
            self.exposure_var.set(str(exposure_value_us))
            
            # 设置相机值
            self.camera.set_control_value(asi.ASI_EXPOSURE, exposure_value_us)
            
            # 更新显示标签
            formatted_time = self.format_exposure_time(exposure_value_us)
            self.exposure_display_label.config(text=f"({formatted_time})")
            
            print(f"曝光设置为: {exposure_value_us}μs ({formatted_time})")
        
        except ValueError as e:
            messagebox.showerror("输入错误", f"{e}")
            self.exposure_display_label.config(text="")  # 出错时清空
        except Exception as e:
            messagebox.showerror("错误", f"设置曝光失败: {str(e)}")
            self.exposure_display_label.config(text="")  # 出错时清空
    
    def set_gain(self, event=None):
        if not self.camera:
            return
        
        try:
            gain_value = int(self.gain_var.get())
            self.camera.set_control_value(asi.ASI_GAIN, gain_value)
            print(f"增益设置为: {gain_value}")
        except ValueError:
            messagebox.showerror("错误", "增益必须是整数")
        except Exception as e:
            messagebox.showerror("错误", f"设置增益失败: {str(e)}")
    
    def toggle_auto_exposure(self):
        """切换自动曝光状态"""
        self.auto_exposure_enabled = self.auto_exposure_var.get()
        if self.auto_exposure_enabled:
            print("自动曝光已启用")
            # 启用自动曝光时，重置计数器和历史记录
            self.auto_exposure_counter = 0
            self.last_auto_exposure_time = time.time()
            self.auto_exposure_history = []
        else:
            print("自动曝光已禁用")
    
    def update_target_gray(self, event=None):
        """更新目标灰度值"""
        try:
            target_value = int(self.target_gray_var.get())
            if target_value < 1000 or target_value > 65000:
                raise ValueError("目标灰度值应在1000-65000之间")
            self.target_gray_value = target_value
            print(f"目标灰度值设置为: {target_value}")
        except ValueError as e:
            messagebox.showerror("错误", f"目标灰度值设置错误: {e}")
            self.target_gray_var.set(str(self.target_gray_value))
    
    def update_safe_exposure(self, event=None):
        """更新安全曝光时间"""
        try:
            safe_time_ms = float(self.safe_exposure_var.get())
            if safe_time_ms < 0.1 or safe_time_ms > 1000:
                raise ValueError("安全曝光时间应在0.1-1000ms之间")
            self.safe_exposure_time = int(safe_time_ms * 1000)  # 转换为微秒
            print(f"安全曝光时间设置为: {safe_time_ms}ms ({self.safe_exposure_time}μs)")
        except ValueError as e:
            messagebox.showerror("错误", f"安全曝光时间设置错误: {e}")
            self.safe_exposure_var.set(str(self.safe_exposure_time / 1000))
    
    def auto_adjust_exposure(self, current_max_gray):
        """根据当前最大灰度值自动调整曝光时间（优化版本）"""
        if not self.auto_exposure_enabled or not self.camera:
            return
        
        try:
            # 减少调整频率限制，提高响应速度
            current_time = time.time()
            if current_time - self.last_auto_exposure_time < 0.2:
                return
            
            # 获取当前曝光时间
            current_exposure = self.parse_exposure_input(self.exposure_var.get())
            
            # 定义过曝阈值（95%满量程认为是过曝）
            overexposure_threshold = 62000  # 约95%的65536
            
            if current_max_gray >= overexposure_threshold:
                # 检测到过曝，直接退回到安全曝光时间
                safe_exposure = self.safe_exposure_time
                
                # 确保不低于相机最小曝光时间
                try:
                    controls = self.camera.get_controls()
                    if 'Exposure' in controls:
                        exp_min = controls['Exposure']['MinValue']
                        safe_exposure = max(exp_min, safe_exposure)
                except:
                    pass
                
                self.exposure_var.set(str(safe_exposure))
                self.set_exposure()
                self.last_auto_exposure_time = current_time
                
                print(f"检测到过曝! 退回安全曝光: {current_exposure}μs -> {safe_exposure}μs "
                      f"(灰度: {current_max_gray}, 安全时间: {self.safe_exposure_time/1000}ms)")
                return
            
            # 正常情况下的比例调整
            if current_max_gray > 0:
                ratio = self.target_gray_value / current_max_gray
                error_percentage = abs(current_max_gray - self.target_gray_value) / self.target_gray_value
                
                # 根据误差大小决定调整策略
                if error_percentage > 0.5:  # 误差超过50%，使用大步长快速调整
                    if ratio > 4.0:
                        ratio = 4.0
                    elif ratio < 0.25:
                        ratio = 0.25
                elif error_percentage > 0.2:  # 误差20%-50%，使用中等步长
                    if ratio > 2.0:
                        ratio = 2.0
                    elif ratio < 0.5:
                        ratio = 0.5
                else:  # 误差小于20%，使用小步长精细调整
                    if ratio > 1.5:
                        ratio = 1.5
                    elif ratio < 0.7:
                        ratio = 0.7
                
                # 计算新的曝光时间
                new_exposure = int(current_exposure * ratio)
                
                # 获取相机曝光范围限制
                try:
                    controls = self.camera.get_controls()
                    if 'Exposure' in controls:
                        exp_min = controls['Exposure']['MinValue']
                        exp_max = controls['Exposure']['MaxValue']
                        new_exposure = max(exp_min, min(exp_max, new_exposure))
                except:
                    # 如果无法获取范围，使用默认限制
                    new_exposure = max(1, min(30000000, new_exposure))  # 1μs到30s
                
                # 动态调整门槛：误差越大，门槛越低
                if error_percentage > 0.3:
                    adjustment_threshold = 0.01  # 1%门槛，快速调整
                elif error_percentage > 0.1:
                    adjustment_threshold = 0.02  # 2%门槛
                else:
                    adjustment_threshold = 0.05  # 5%门槛，精细调整
                
                if abs(new_exposure - current_exposure) / current_exposure > adjustment_threshold:
                    # 记录调整历史
                    self.auto_exposure_history.append({
                        'time': current_time,
                        'old_exposure': current_exposure,
                        'new_exposure': new_exposure,
                        'max_gray': current_max_gray,
                        'error': error_percentage
                    })
                    
                    # 只保留最近10次记录
                    if len(self.auto_exposure_history) > 10:
                        self.auto_exposure_history = self.auto_exposure_history[-10:]
                    
                    self.exposure_var.set(str(new_exposure))
                    self.set_exposure()
                    self.last_auto_exposure_time = current_time
                    
                    # 显示调整信息，包括误差百分比
                    print(f"自动曝光调整: {current_exposure}μs -> {new_exposure}μs "
                          f"(灰度: {current_max_gray}/{self.target_gray_value}, 误差: {error_percentage:.1%})")
                
        except Exception as e:
            print(f"自动曝光调整失败: {e}")
    
    def calculate_and_update_stats(self, frame):
        """在单独线程中计算统计信息并更新显示"""
        try:
            # 计算基本统计信息（总是计算）
            self.calculate_basic_statistics(frame)
            # 在主线程中更新基本统计显示
            self.root.after(0, self.update_basic_statistics_display)
            
            # 只有启用直方图时才计算直方图
            if self.enable_stats_var.get():
                self.calculate_histogram_data(frame)
                self.root.after(0, self.update_histogram)
        except Exception as e:
            print(f"统计计算错误: {str(e)}")
    
    def calculate_basic_statistics(self, frame):
        """计算基本灰度统计信息（快速版本）"""
        try:
            if frame is None:
                return
            
            # 使用numpy的快速计算
            flat_frame = frame.flatten()
            
            # 计算基本统计信息
            self.gray_min = int(np.min(flat_frame))
            self.gray_max = int(np.max(flat_frame))
            self.gray_mean = float(np.mean(flat_frame))
            
        except Exception as e:
            print(f"基本统计计算错误: {str(e)}")
    
    def calculate_histogram_data(self, frame):
        """计算直方图数据（仅在需要时调用）"""
        try:
            if frame is None:
                return
            
            # 使用numpy的快速计算
            flat_frame = frame.flatten()
            
            # 优化的直方图计算 - 使用更少的bins和采样
            if frame.dtype == np.uint16:
                # 为了提高性能，只使用128个bins而不是256个
                # 并且只对图像的一部分进行采样
                sample_size = min(len(flat_frame), 50000)  # 最多采样50000个像素
                if len(flat_frame) > sample_size:
                    indices = np.random.choice(len(flat_frame), sample_size, replace=False)
                    sample_data = flat_frame[indices]
                else:
                    sample_data = flat_frame
                
                hist, bin_edges = np.histogram(sample_data, bins=128, range=(0, 65536))
            else:
                sample_size = min(len(flat_frame), 30000)
                if len(flat_frame) > sample_size:
                    indices = np.random.choice(len(flat_frame), sample_size, replace=False)
                    sample_data = flat_frame[indices]
                else:
                    sample_data = flat_frame
                hist, bin_edges = np.histogram(sample_data, bins=128, range=(0, 256))
            
            self.histogram_data = hist
            self.bin_edges = bin_edges
            
        except Exception as e:
            print(f"直方图计算错误: {str(e)}")
    
    def update_basic_statistics_display(self):
        """更新基本统计信息显示（总是更新）"""
        try:
            self.min_value_label.config(text=str(self.gray_min))
            self.max_value_label.config(text=str(self.gray_max))
            self.mean_value_label.config(text=f"{self.gray_mean:.0f}")
            self.range_value_label.config(text=str(self.gray_max - self.gray_min))
        except Exception as e:
            print(f"基本统计显示更新错误: {str(e)}")
    
    def calculate_gray_statistics(self, frame):
        """计算灰度统计信息（优化版本） - 保持向后兼容"""
        # 这个方法保持向后兼容，实际调用新的分离方法
        self.calculate_basic_statistics(frame)
        if self.enable_stats_var.get():
            self.calculate_histogram_data(frame)
    
    def update_statistics_display(self):
        """更新统计信息显示 - 保持向后兼容"""
        # 这个方法保持向后兼容，实际调用新的方法
        self.update_basic_statistics_display()
    
    def update_histogram(self):
        """更新直方图显示（优化版本）"""
        try:
            if self.histogram_data is None or not self.enable_stats_var.get():
                return
            
            # 清除并重绘 - 使用更快的绘图方法
            self.ax.clear()
            
            # 绘制直方图 - 简化版本
            bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            self.ax.bar(bin_centers, self.histogram_data, 
                        width=(self.bin_edges[1] - self.bin_edges[0]) * 0.9,
                        alpha=0.7, color='steelblue')
            
            # 简化标签和格式
            self.ax.set_xlabel('灰度值')
            self.ax.set_ylabel('像素数')
            self.ax.set_title(f'灰度分布 (最小:{self.gray_min} 最大:{self.gray_max} 均值:{self.gray_mean:.0f})')
            
            # 设置范围
            if hasattr(self, 'bin_edges'):
                self.ax.set_xlim(0, self.bin_edges[-1])
            
            # 减少网格线以提高性能
            self.ax.grid(True, alpha=0.2)
            
            # 使用更快的绘图更新
            self.canvas.draw_idle()  # 使用idle更新而不是立即更新
            
        except Exception as e:
            print(f"直方图更新错误: {str(e)}")
    
    def update_empty_histogram(self):
        """显示空的直方图"""
        self.ax.clear()
        self.ax.set_xlabel('灰度值')
        self.ax.set_ylabel('像素数量')
        
        if hasattr(self, 'enable_stats_var') and not self.enable_stats_var.get():
            self.ax.set_title('灰度分布 (请勾选"启用灰度直方图")')
        else:
            self.ax.set_title('灰度分布 (等待数据...)')
            
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(0, 65536)
        self.canvas.draw()
    
    def start_capture(self):
        if not self.camera:
            messagebox.showerror("错误", "相机未连接")
            return
        
        if self.is_capturing:
            return
        
        try:
            self.camera.start_video_capture()
            self.is_capturing = True
            self.frame_count = 0
            self.fps_start_time = time.time()
            self.stats_update_counter = 0
            self.auto_exposure_counter = 0  # 重置自动曝光计数器
            self.last_auto_exposure_time = time.time()
            
            # 启动捕获线程
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            
            # 更新按钮状态
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            
            # 根据统计功能状态更新显示
            if not self.enable_stats_var.get():
                self.update_empty_histogram()
            
            print("开始视频捕获")
            if self.auto_exposure_enabled:
                print(f"自动曝光已启用，目标灰度值: {self.target_gray_value}")
            
        except Exception as e:
            messagebox.showerror("错误", f"启动视频捕获失败: {str(e)}")
    
    def stop_capture(self):
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        
        try:
            self.camera.stop_video_capture()
            
            # 更新按钮状态
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            
            # 重置显示
            self.root.after(0, self.update_empty_histogram)
            
            # 清空统计信息显示（但保留最后的值，不重置为0）
            # 用户可以在停止后仍然看到最后一帧的统计信息
            
            print("停止视频捕获")
            
        except Exception as e:
            messagebox.showerror("错误", f"停止视频捕获失败: {str(e)}")
    
    def capture_loop(self):
        """视频捕获循环，运行在单独线程中"""
        consecutive_timeouts = 0
        max_consecutive_timeouts = 10
        
        while self.is_capturing:
            try:
                # 捕获帧，使用更长的超时时间
                frame = self.camera.capture_video_frame(timeout=200)
                
                if frame is not None:
                    consecutive_timeouts = 0  # 重置超时计数
                    self.current_frame = frame
                    self.frame_count += 1
                    self.stats_update_counter += 1
                    self.auto_exposure_counter += 1
                    
                    # 计算FPS
                    if self.frame_count % 30 == 0:  # 每30帧计算一次FPS
                        current_time = time.time()
                        elapsed = current_time - self.fps_start_time
                        if elapsed > 0:
                            self.current_fps = 30 / elapsed
                            self.fps_start_time = current_time
                            self.frame_count = 0
                            
                            # 更新FPS显示（在主线程中）
                            self.root.after(0, lambda: self.fps_label.config(text=f"FPS: {self.current_fps:.1f}"))
                    
                    # 自动曝光检查（每10帧检查一次，更快响应）
                    if self.auto_exposure_enabled and self.auto_exposure_counter >= self.auto_exposure_adjustment_interval:
                        try:
                            # 计算当前帧的最大灰度值
                            current_max_gray = int(np.max(frame))
                            # 在主线程中进行曝光调整
                            self.root.after(0, lambda: self.auto_adjust_exposure(current_max_gray))
                            self.auto_exposure_counter = 0
                        except Exception as e:
                            print(f"自动曝光检查失败: {e}")
                    
                    # 每帧都计算基本统计信息（用于实时显示和自动曝光）
                    if self.stats_update_counter % 30 == 0:  # 每30帧更新一次基本统计
                        # 在单独线程中计算基本统计信息
                        stats_thread = threading.Thread(target=self.calculate_and_update_stats, args=(frame.copy(),), daemon=True)
                        stats_thread.start()
                    
                    # 只有启用直方图功能时才计算详细的直方图数据
                    elif self.enable_stats_var.get() and self.stats_update_counter % 90 == 0:  # 每90帧更新一次直方图
                        # 在单独线程中计算统计信息以避免阻塞
                        stats_thread = threading.Thread(target=self.calculate_and_update_stats, args=(frame.copy(),), daemon=True)
                        stats_thread.start()
                    
                    # 转换图像用于显示
                    display_image = self.convert_for_display(frame)
                    if display_image:
                        # 在主线程中更新显示
                        self.root.after(0, lambda img=display_image: self.update_display(img))
                
                else:
                    # 处理超时情况
                    consecutive_timeouts += 1
                    if consecutive_timeouts <= 3:  # 前几次超时只记录，不报错
                        pass
                    elif consecutive_timeouts <= max_consecutive_timeouts:
                        if consecutive_timeouts % 5 == 0:  # 每5次超时报告一次
                            print(f"连续超时 {consecutive_timeouts} 次，检查相机连接...")
                    else:
                        print("连续超时过多，尝试重新初始化相机...")
                        self.root.after(0, self.handle_capture_timeout)
                        break
                
                # 减少sleep时间以提高帧率
                time.sleep(0.001)  # 更短的延迟
                
            except Exception as e:
                error_msg = str(e)
                if "Timeout" not in error_msg:  # 只打印非超时错误
                    print(f"捕获帧错误: {error_msg}")
                consecutive_timeouts += 1
                
                if consecutive_timeouts > max_consecutive_timeouts:
                    print("错误过多，停止捕获")
                    self.root.after(0, self.handle_capture_error)
                    break
                
                time.sleep(0.05)
    
    def handle_capture_timeout(self):
        """处理捕获超时的恢复机制"""
        try:
            print("尝试恢复相机连接...")
            if self.camera:
                # 停止当前捕获
                try:
                    self.camera.stop_video_capture()
                except:
                    pass
                
                # 等待一下再重新开始
                time.sleep(0.5)
                
                # 重新开始捕获
                if self.is_capturing:
                    self.camera.start_video_capture()
                    print("相机连接已恢复")
        except Exception as e:
            print(f"相机恢复失败: {e}")
            self.stop_capture()
    
    def handle_capture_error(self):
        """处理捕获错误"""
        print("由于连续错误，停止视频捕获")
        self.stop_capture()
    
    def convert_for_display(self, frame):
        """将RAW16图像转换为可显示的格式"""
        try:
            # RAW16图像是16位灰度图像
            if frame.dtype == np.uint16:
                # 转换为8位用于显示
                frame_8bit = (frame / 256).astype(np.uint8)
            else:
                frame_8bit = frame
            
            # 调整图像大小以适应显示区域
            height, width = frame_8bit.shape[:2]
            max_display_width = 640
            max_display_height = 480
            
            if width > max_display_width or height > max_display_height:
                scale = min(max_display_width / width, max_display_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                pil_image = Image.fromarray(frame_8bit)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                pil_image = Image.fromarray(frame_8bit)
            
            return ImageTk.PhotoImage(pil_image)
            
        except Exception as e:
            print(f"图像转换错误: {str(e)}")
            return None
    
    def update_display(self, image):
        """更新视频显示"""
        self.video_label.config(image=image, text="")
        self.video_label.image = image  # 保持引用
    
    def save_frame(self):
        """保存当前帧为PNG文件"""
        if self.current_frame is None:
            messagebox.showwarning("警告", "没有可保存的图像")
            return
        
        try:
            # 获取当前曝光时间
            exposure_time = self.exposure_var.get()
            try:
                # 解析曝光时间为微秒值
                exposure_us = self.parse_exposure_input(exposure_time)
            except:
                exposure_us = 10000  # 默认值
            
            # 获取当前会话文件夹
            session_folder = self.get_current_session_folder()
            
            # 生成文件名：Violet{索引}_{曝光时间us}.png
            filename = f"Violet{self.file_index}_{exposure_us}.png"
            filepath = os.path.join(session_folder, filename)
            
            # 保存RAW16图像
            if self.current_frame.dtype == np.uint16:
                # PIL可以处理16位灰度图像
                pil_image = Image.fromarray(self.current_frame, mode='I;16')
                pil_image.save(filepath, 'PNG')
            else:
                # 如果不是16位，直接保存
                pil_image = Image.fromarray(self.current_frame)
                pil_image.save(filepath, 'PNG')
            
            print(f"图像已保存: {filepath}")
            
            # 更新文件索引并在UI中显示
            self.file_index += 1
            self.index_var.set(str(self.file_index))
            
            # 更新状态显示（简化版，不弹窗）
            self.status_label.config(text=f"已保存: {filename}")
            
        except Exception as e:
            error_msg = f"保存图像失败: {str(e)}"
            messagebox.showerror("错误", error_msg)
            print(error_msg)
    
    def apply_resolution(self):
        """应用分辨率设置"""
        if not self.camera:
            messagebox.showerror("错误", "相机未连接")
            return
        
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            bins = int(self.binning_var.get())
            
            # 验证参数
            if width <= 0 or height <= 0:
                raise ValueError("宽度和高度必须大于0")
            
            if bins not in self.supported_bins:
                raise ValueError(f"不支持的binning值: {bins}")
            
            # 检查分辨率限制
            max_width_binned = self.max_width // bins
            max_height_binned = self.max_height // bins
            
            if width > max_width_binned:
                raise ValueError(f"宽度超过最大值: {max_width_binned} (binning {bins})")
            
            if height > max_height_binned:
                raise ValueError(f"高度超过最大值: {max_height_binned} (binning {bins})")
            
            # 确保宽度是8的倍数，高度是2的倍数
            width = (width // 8) * 8
            height = (height // 2) * 2
            
            # 更新显示值
            self.width_var.set(str(width))
            self.height_var.set(str(height))
            
            # 如果正在捕获，需要暂停并重新设置
            was_capturing = self.is_capturing
            if was_capturing:
                # 暂时停止捕获但不更新UI按钮状态
                try:
                    self.camera.stop_video_capture()
                except:
                    pass
                time.sleep(0.1)  # 等待停止完成
            
            # 应用设置
            self.camera.set_roi_format(width, height, bins, asi.ASI_IMG_RAW16)
            
            # 更新状态显示
            roi_info = self.get_current_roi_info()
            self.status_label.config(text=f"分辨率设置成功: {width}×{height}, Binning: {bins}")
            self.roi_info_label.config(text=roi_info)
            
            print(f"分辨率已设置: {width}×{height}, Binning: {bins}")
            
            # 如果之前在捕获，重新开始（不改变UI状态）
            if was_capturing:
                time.sleep(0.1)
                try:
                    self.camera.start_video_capture()
                    print("视频捕获已恢复")
                except Exception as e:
                    print(f"重新开始捕获失败: {e}")
                    self.stop_capture()
            
            # 更新ROI Canvas显示
            self.update_roi_canvas()
                
        except ValueError as e:
            messagebox.showerror("错误", f"参数错误: {str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"设置分辨率失败: {str(e)}")
    
    def apply_roi(self):
        """应用ROI位置设置"""
        if not self.camera:
            messagebox.showerror("错误", "相机未连接")
            return
        
        try:
            start_x = int(self.roi_x_var.get())
            start_y = int(self.roi_y_var.get())
            
            # 验证参数
            if start_x < 0 or start_y < 0:
                raise ValueError("ROI起始坐标不能为负数")
            
            # 获取当前ROI设置
            current_roi = self.camera.get_roi_format()
            width, height, bins = current_roi[0], current_roi[1], current_roi[2]
            
            # 检查ROI是否超出边界
            max_x = (self.max_width // bins) - width
            max_y = (self.max_height // bins) - height
            
            if start_x > max_x:
                raise ValueError(f"X坐标超出范围，最大值: {max_x}")
            
            if start_y > max_y:
                raise ValueError(f"Y坐标超出范围，最大值: {max_y}")
            
            # 如果正在捕获，需要暂停并重新设置
            was_capturing = self.is_capturing
            if was_capturing:
                # 暂时停止捕获但不更新UI按钮状态
                try:
                    self.camera.stop_video_capture()
                except:
                    pass
                time.sleep(0.1)
            
            # 应用ROI位置
            self.camera.set_roi_start_position(start_x, start_y)
            
            # 更新状态显示
            roi_info = self.get_current_roi_info()
            self.status_label.config(text=f"ROI位置设置成功: 起始({start_x}, {start_y}), 大小({width}×{height})")
            self.roi_info_label.config(text=roi_info)

            print(f"ROI位置已设置: 起始({start_x}, {start_y})")
            
            # 如果之前在捕获，重新开始（不改变UI状态）
            if was_capturing:
                time.sleep(0.1)
                try:
                    self.camera.start_video_capture()
                    print("视频捕获已恢复")
                except Exception as e:
                    print(f"重新开始捕获失败: {e}")
                    self.stop_capture()
            
            # 更新ROI Canvas显示
            self.update_roi_canvas()
                
        except ValueError as e:
            messagebox.showerror("错误", f"参数错误: {str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"设置ROI失败: {str(e)}")
    
    def center_roi(self):
        """将ROI居中"""
        if not self.camera:
            messagebox.showerror("错误", "相机未连接")
            return
        
        try:
            # 获取当前ROI设置
            current_roi = self.camera.get_roi_format()
            width, height, bins = current_roi[0], current_roi[1], current_roi[2]
            
            # 计算居中位置
            max_width_binned = self.max_width // bins
            max_height_binned = self.max_height // bins
            
            center_x = (max_width_binned - width) // 2
            center_y = (max_height_binned - height) // 2
            
            # 更新UI
            self.roi_x_var.set(str(center_x))
            self.roi_y_var.set(str(center_y))
            
            # 应用设置
            self.apply_roi()
            
        except Exception as e:
            messagebox.showerror("错误", f"居中ROI失败: {str(e)}")
    
    def set_preset_resolution(self, preset):
        """设置预设分辨率"""
        if not self.camera:
            messagebox.showerror("错误", "相机未连接")
            return
        
        try:
            if preset == "full":
                width = self.max_width
                height = self.max_height
                bins = 1
            elif preset == "1280x960":
                width = min(1280, self.max_width)
                height = min(960, self.max_height)
                bins = 1
            elif preset == "640x480":
                width = min(640, self.max_width)
                height = min(480, self.max_height)
                bins = 1
            elif preset == "320x240":
                width = min(320, self.max_width)
                height = min(240, self.max_height)
                bins = 1
            else:
                return
            
            # 更新UI
            self.width_var.set(str(width))
            self.height_var.set(str(height))
            self.binning_var.set(str(bins))
            
            # 应用设置
            self.apply_resolution()
            
            # 居中ROI
            self.center_roi()
            
        except Exception as e:
            messagebox.showerror("错误", f"设置预设分辨率失败: {str(e)}")

    def get_current_roi_info(self):
        """获取当前ROI信息用于显示"""
        try:
            if self.camera:
                roi = self.camera.get_roi()
                roi_format = self.camera.get_roi_format()
                return f"ROI: 起始({roi[0]}, {roi[1]}) 大小({roi[2]}×{roi[3]}) Binning:{roi_format[2]}"
            return "ROI: 未设置"
        except:
            return "ROI: 获取失败"
    
    def on_closing(self):
        """程序关闭时的清理工作"""
        # 停止相机捕获
        self.stop_capture()
        if self.camera:
            try:
                self.camera.close()
            except:
                pass
        
        # 关闭串口
        if SERIAL_AVAILABLE:
            self.close_serial_port()
        
        self.root.destroy()


def main():
    # 全局禁用字体相关警告
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
    
    # 检查zwoasi库
    try:
        asi.init()
    except Exception as e:
        print(f"ZWO ASI库初始化失败: {e}")
        print("请确保已安装ZWO ASI SDK")
        return
    
    # 检查串口库
    if not SERIAL_AVAILABLE:
        print("警告：pyserial库未安装，可使用以下命令安装：")
        print("pip install pyserial")
        print("串口功能将不可用，但相机功能正常")
    
    # 检查系统信息库
    if not PSUTIL_AVAILABLE:
        print("提示：psutil库未安装，智能路径选择将使用简化模式")
        print("可使用以下命令安装以获得更好的智能路径选择功能：")
        print("pip install psutil")
    
    root = tk.Tk()
    app = ZWOCameraController(root)
    
    # 设置关闭事件处理
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # 启动GUI
    root.mainloop()


if __name__ == "__main__":
    main()