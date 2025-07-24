#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口控制工具 - 优化版
功能：串口通信（收发），PWM数据控制，白光模式控制，四通道模式控制
优化：解决STM32逐字节发送导致的接收问题，增强回显验证
新增：自定义数值快速设置功能
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import datetime
import os

# 串口通信
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("警告：未安装pyserial库，串口功能将不可用")
    print("请使用 pip install pyserial 安装")


class SerialController:
    def __init__(self, root):
        self.root = root
        self.root.title("串口控制工具 - 优化版")
        self.root.geometry("1400x900")  # 增大窗口尺寸
        self.root.minsize(1200, 800)    # 设置最小尺寸
        
        # 配置tkinter字体
        try:
            import platform
            import tkinter.font as tkFont
            
            system = platform.system()
            if system == "Windows":
                try:
                    test_font = tkFont.Font(family='Microsoft YaHei', size=9)
                    default_font_family = 'Microsoft YaHei'
                except:
                    try:
                        test_font = tkFont.Font(family='SimHei', size=9)
                        default_font_family = 'SimHei'
                    except:
                        default_font_family = 'Arial'
                
                for font_name in ['TkDefaultFont', 'TkTextFont', 'TkFixedFont', 'TkMenuFont', 'TkHeadingFont']:
                    try:
                        font = tkFont.nametofont(font_name)
                        font.configure(family=default_font_family, size=9)
                    except Exception as e:
                        print(f"配置字体 {font_name} 失败: {e}")
                        
        except Exception as e:
            print(f"字体设置失败，使用系统默认字体: {e}")
        
        # 串口相关变量
        self.serial_port = None
        self.is_serial_open = False
        self.serial_thread = None
        self.serial_buffer = []
        self.max_buffer_size = 1000
        self.receive_byte_count = 0
        
        # 回显验证相关
        self.echo_buffer = bytearray()
        self.expected_echo_data = None
        self.echo_timeout = 0.2  # 减少回显超时时间到200ms
        self.echo_start_time = None
        self.echo_verification_lock = threading.Lock()
        self.fast_echo_mode = tk.BooleanVar(value=True)  # 快速回显模式
        
        # 串口发送相关变量
        self.send_stats = {
            'total_sent': 0,
            'last_send_time': 0,
            'send_errors': 0,
            'send_history': []
        }
        
        # 串口优化选项
        self.auto_clear_buffer = tk.BooleanVar(value=True)
        self.clear_before_send = tk.BooleanVar(value=True)
        self.performance_mode = tk.BooleanVar(value=False)
        
        # 自定义数值变量 - 新增
        self.custom_value_var = tk.StringVar(value="5")  # 默认值为5
        
        # 创建GUI界面
        self.create_widgets()
        
        # 初始化串口端口列表
        if SERIAL_AVAILABLE:
            self.refresh_serial_ports()
            self.auto_open_latest_serial()
    
    def create_widgets(self):
        # 创建主要的PanedWindow来分割左右区域
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板框架（使用Canvas和Scrollbar实现滚动）
        left_canvas_frame = ttk.Frame(main_paned)
        
        # 创建Canvas和滚动条
        left_canvas = tk.Canvas(left_canvas_frame, width=450, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_canvas_frame, orient="vertical", command=left_canvas.yview)
        self.left_scrollable_frame = ttk.Frame(left_canvas)
        
        self.left_scrollable_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_canvas.create_window((0, 0), window=self.left_scrollable_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")
        
        # 绑定鼠标滚轮事件 - 兼容Windows和Linux
        def on_mousewheel(event):
            try:
                # Windows
                if event.delta:
                    left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                # Linux
                elif event.num == 4:
                    left_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    left_canvas.yview_scroll(1, "units")
            except:
                pass
        
        # Windows系统滚轮绑定
        left_canvas.bind("<MouseWheel>", on_mousewheel)
        # Linux系统滚轮绑定
        left_canvas.bind("<Button-4>", on_mousewheel)
        left_canvas.bind("<Button-5>", on_mousewheel)
        
        # 右侧数据接收区域
        right_frame = ttk.Frame(main_paned)
        
        # 将左右框架添加到PanedWindow
        main_paned.add(left_canvas_frame, weight=0)
        main_paned.add(right_frame, weight=1)
        
        # 设置初始分割位置
        self.root.after(10, lambda: main_paned.sashpos(0, 450))
        
        # === 串口控制面板 ===
        if SERIAL_AVAILABLE:
            serial_control_frame = ttk.LabelFrame(self.left_scrollable_frame, text="串口控制", padding="10")
            serial_control_frame.pack(fill=tk.X, pady=(0, 10))
            
            # 第一行：端口和波特率选择
            port_frame = ttk.Frame(serial_control_frame)
            port_frame.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Label(port_frame, text="端口:").pack(side=tk.LEFT, padx=(0, 5))
            self.port_var = tk.StringVar()
            self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, width=10)
            self.port_combo.pack(side=tk.LEFT, padx=(0, 10))
            
            ttk.Button(port_frame, text="刷新", command=self.refresh_serial_ports).pack(side=tk.LEFT, padx=(0, 10))
            
            ttk.Label(port_frame, text="波特率:").pack(side=tk.LEFT, padx=(0, 5))
            self.baudrate_var = tk.StringVar(value="460800")
            baudrate_combo = ttk.Combobox(port_frame, textvariable=self.baudrate_var, width=8,
                                        values=["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"])
            baudrate_combo.pack(side=tk.LEFT, padx=(0, 10))
            baudrate_combo.state(['readonly'])
            
            # 第二行：控制按钮
            button_frame = ttk.Frame(serial_control_frame)
            button_frame.pack(fill=tk.X, pady=(5, 0))
            
            self.serial_open_button = ttk.Button(button_frame, text="打开串口", command=self.open_serial_port)
            self.serial_open_button.pack(side=tk.LEFT, padx=(0, 5))
            
            self.serial_close_button = ttk.Button(button_frame, text="关闭串口", command=self.close_serial_port, state=tk.DISABLED)
            self.serial_close_button.pack(side=tk.LEFT, padx=(0, 5))
            
            ttk.Button(button_frame, text="清空接收区", command=self.clear_serial_data).pack(side=tk.LEFT, padx=(0, 5))
            
            ttk.Button(button_frame, text="清空硬件缓冲区", command=self.clear_hardware_buffers).pack(side=tk.LEFT, padx=(0, 5))
            
            # 串口状态显示
            self.serial_status_label = ttk.Label(serial_control_frame, text="串口状态: 未连接", foreground="red")
            self.serial_status_label.pack(pady=(5, 0))
            
            # 串口优化选项
            serial_opt_frame = ttk.LabelFrame(self.left_scrollable_frame, text="串口优化选项", padding="10")
            serial_opt_frame.pack(fill=tk.X, pady=(0, 10))
            
            opt_frame1 = ttk.Frame(serial_opt_frame)
            opt_frame1.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Checkbutton(opt_frame1, text="发送前清空缓冲区", variable=self.clear_before_send).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Checkbutton(opt_frame1, text="自动清空缓冲区", variable=self.auto_clear_buffer).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Checkbutton(opt_frame1, text="性能模式", variable=self.performance_mode).pack(side=tk.LEFT)
            
            # 回显超时设置
            opt_frame2 = ttk.Frame(serial_opt_frame)
            opt_frame2.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Label(opt_frame2, text="回显超时(秒):").pack(side=tk.LEFT, padx=(0, 5))
            self.echo_timeout_var = tk.StringVar(value="0.5")
            echo_timeout_entry = ttk.Entry(opt_frame2, textvariable=self.echo_timeout_var, width=8)
            echo_timeout_entry.pack(side=tk.LEFT, padx=(0, 10))
            echo_timeout_entry.bind('<Return>', self.update_echo_timeout)
            
            # 缓冲区状态显示
            self.buffer_status_label = ttk.Label(opt_frame2, text="缓冲区: 0/1000")
            self.buffer_status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # === 串口发送控制面板 ===
        if SERIAL_AVAILABLE:
            send_control_frame = ttk.LabelFrame(self.left_scrollable_frame, text="串口发送控制", padding="10")
            send_control_frame.pack(fill=tk.X, pady=(0, 10))
            
            # 控制模式选择
            mode_frame = ttk.Frame(send_control_frame)
            mode_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(mode_frame, text="控制模式:").pack(side=tk.LEFT, padx=(0, 5))
            # 修改默认值为四通道模式
            self.control_mode_var = tk.StringVar(value="四通道模式")
            mode_combo = ttk.Combobox(mode_frame, textvariable=self.control_mode_var, width=15,
                                    values=["四通道模式", "全通道模式", "白光模式", "定时器频率控制"])
            mode_combo.pack(side=tk.LEFT, padx=(0, 10))
            mode_combo.state(['readonly'])
            mode_combo.bind('<<ComboboxSelected>>', self.on_control_mode_changed)
            
            # 四通道模式控制区域（新增，默认显示）
            self.four_channel_frame = ttk.LabelFrame(send_control_frame, text="四通道控制 (CH2、CH6、CH12、CH14)", padding="10")
            self.four_channel_frame.pack(fill=tk.X, pady=(0, 10))
            
            # 创建4个通道控制
            self.four_channel_vars = {}
            four_channels = [
                ("CH2", 1),   # 对应CH2，数组索引1
                ("CH6", 5),   # 对应CH6，数组索引5
                ("CH12", 11), # 对应CH12，数组索引11
                ("CH14", 13)  # 对应CH14，数组索引13
            ]
            
            for i, (label, ch_index) in enumerate(four_channels):
                ch_frame = ttk.Frame(self.four_channel_frame)
                ch_frame.pack(fill=tk.X, pady=2)
                
                ttk.Label(ch_frame, text=label, font=("Arial", 10), width=8).pack(side=tk.LEFT, padx=(0, 5))
                var = tk.StringVar(value="0")
                self.four_channel_vars[ch_index] = var
                entry = ttk.Entry(ch_frame, textvariable=var, width=8, font=("Arial", 10))
                entry.pack(side=tk.LEFT, padx=(0, 10))
                entry.bind('<Return>', self.validate_four_channel_values)
                
                # 添加滑块控制
                scale = ttk.Scale(ch_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                command=lambda val, idx=ch_index: self.on_four_channel_scale_changed(idx, val))
                scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
                scale.bind('<ButtonRelease-1>', lambda e, idx=ch_index: self.sync_four_channel_entry_from_scale(idx, e))
                
                # 绑定输入框变化到滑块
                var.trace('w', lambda *args, idx=ch_index: self.sync_four_channel_scale_from_entry(idx))
            
            # 四通道模式快速设置按钮
            four_quick_frame = ttk.Frame(self.four_channel_frame)
            four_quick_frame.pack(fill=tk.X, pady=(10, 0))
            
            ttk.Label(four_quick_frame, text="快速设置:").pack(side=tk.LEFT, padx=(0, 5))
            
            button_row1 = ttk.Frame(four_quick_frame)
            button_row1.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Button(button_row1, text="全部清零", command=self.clear_all_four_channel).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(button_row1, text="全部25", command=lambda: self.set_all_four_channel(25)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(button_row1, text="全部50", command=lambda: self.set_all_four_channel(50)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(button_row1, text="全部100", command=lambda: self.set_all_four_channel(100)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(button_row1, text="示例数据", command=self.set_four_channel_example).pack(side=tk.LEFT, padx=(0, 5))
            
            # 自定义数值设置
            custom_four_frame = ttk.Frame(self.four_channel_frame)
            custom_four_frame.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Label(custom_four_frame, text="自定义数值:").pack(side=tk.LEFT, padx=(0, 5))
            custom_four_entry = ttk.Entry(custom_four_frame, textvariable=self.custom_value_var, width=8)
            custom_four_entry.pack(side=tk.LEFT, padx=(0, 5))
            custom_four_entry.bind('<Return>', self.apply_custom_four_channel)
            ttk.Button(custom_four_frame, text="应用到全部", command=self.apply_custom_four_channel).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Label(custom_four_frame, text="(0-100)", font=("Arial", 8), foreground="gray").pack(side=tk.LEFT)
            
            # PWM数据设置区域（全通道模式）- 默认隐藏
            self.pwm_frame = ttk.LabelFrame(send_control_frame, text="PWM数据设置 (全通道模式)", padding="5")
            self.pwm_frame.pack(fill=tk.X, pady=(0, 10))
            self.pwm_frame.pack_forget()
            
            # 创建16个PWM通道输入框
            self.pwm_vars = []
            pwm_grid_frame = ttk.Frame(self.pwm_frame)
            pwm_grid_frame.pack(fill=tk.X)
            
            for i in range(16):
                row = i // 8
                col = i % 8
                
                ch_mini_frame = ttk.Frame(pwm_grid_frame)
                ch_mini_frame.grid(row=row*2, column=col, padx=2, pady=2, sticky="ew")
                
                ttk.Label(ch_mini_frame, text=f"CH{i+1}:", font=("Arial", 8)).pack()
                var = tk.StringVar(value="0")
                self.pwm_vars.append(var)
                entry = ttk.Entry(ch_mini_frame, textvariable=var, width=5, font=("Arial", 8))
                entry.pack()
                entry.bind('<Return>', self.validate_pwm_values)
            
            # 白光模式控制区域（默认隐藏）
            self.white_light_frame = ttk.LabelFrame(send_control_frame, text="白光控制 (白光模式)", padding="10")
            self.white_light_frame.pack(fill=tk.X, pady=(0, 10))
            self.white_light_frame.pack_forget()
            
            # 创建4个白光通道控制
            self.white_light_vars = {}
            white_light_channels = [
                ("CH7 (色温1)", 6),   # 对应CH7，数组索引6
                ("CH8 (色温2)", 7),   # 对应CH8，数组索引7
                ("CH11 (色温3)", 10), # 对应CH11，数组索引10
                ("CH12 (色温4)", 11)  # 对应CH12，数组索引11
            ]
            
            for i, (label, ch_index) in enumerate(white_light_channels):
                wl_frame = ttk.Frame(self.white_light_frame)
                wl_frame.pack(fill=tk.X, pady=2)
                
                ttk.Label(wl_frame, text=label, font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=(0, 5))
                var = tk.StringVar(value="0")
                self.white_light_vars[ch_index] = var
                entry = ttk.Entry(wl_frame, textvariable=var, width=8, font=("Arial", 10))
                entry.pack(side=tk.LEFT, padx=(0, 10))
                entry.bind('<Return>', self.validate_white_light_values)
                
                # 添加滑块控制
                scale = ttk.Scale(wl_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                command=lambda val, idx=ch_index: self.on_white_light_scale_changed(idx, val))
                scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
                scale.bind('<ButtonRelease-1>', lambda e, idx=ch_index: self.sync_white_light_entry_from_scale(idx, e))
                
                # 绑定输入框变化到滑块
                var.trace('w', lambda *args, idx=ch_index: self.sync_white_light_scale_from_entry(idx))
            
            # 白光模式快速设置按钮
            white_quick_frame = ttk.Frame(self.white_light_frame)
            white_quick_frame.pack(fill=tk.X, pady=(10, 0))
            
            ttk.Label(white_quick_frame, text="快速设置:").pack()
            
            white_button_row1 = ttk.Frame(white_quick_frame)
            white_button_row1.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Button(white_button_row1, text="全部清零", command=self.clear_all_white_light).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(white_button_row1, text="全部25", command=lambda: self.set_all_white_light_custom(25)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(white_button_row1, text="全部50", command=lambda: self.set_all_white_light_custom(50)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(white_button_row1, text="暖白光", command=lambda: self.set_white_light_preset("warm")).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(white_button_row1, text="冷白光", command=lambda: self.set_white_light_preset("cool")).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(white_button_row1, text="混合光", command=lambda: self.set_white_light_preset("mixed")).pack(side=tk.LEFT, padx=(0, 5))
            
            # 白光模式自定义数值设置
            custom_white_frame = ttk.Frame(self.white_light_frame)
            custom_white_frame.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Label(custom_white_frame, text="自定义数值:").pack(side=tk.LEFT, padx=(0, 5))
            custom_white_entry = ttk.Entry(custom_white_frame, textvariable=self.custom_value_var, width=8)
            custom_white_entry.pack(side=tk.LEFT, padx=(0, 5))
            custom_white_entry.bind('<Return>', self.apply_custom_white_light)
            ttk.Button(custom_white_frame, text="应用到全部", command=self.apply_custom_white_light).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Label(custom_white_frame, text="(0-100)", font=("Arial", 8), foreground="gray").pack(side=tk.LEFT)
            
            # 定时器频率控制区域（新增，默认隐藏）
            self.timer_frame = ttk.LabelFrame(send_control_frame, text="定时器频率控制", padding="10")
            self.timer_frame.pack(fill=tk.X, pady=(0, 10))
            self.timer_frame.pack_forget()
            
            # 创建4个定时器控制
            self.timer_vars = {}
            timers = [
                ("TIM1", 1),
                ("TIM2", 2),
                ("TIM3", 3),
                ("TIM4", 4)
            ]
            
            for i, (label, tim_num) in enumerate(timers):
                timer_frame = ttk.LabelFrame(self.timer_frame, text=label, padding="5")
                timer_frame.pack(fill=tk.X, pady=2)
                
                # 频率输入
                freq_frame = ttk.Frame(timer_frame)
                freq_frame.pack(fill=tk.X, pady=2)
                ttk.Label(freq_frame, text="频率(Hz):", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))
                freq_var = tk.StringVar(value="1000")
                self.timer_vars[f"tim{tim_num}_freq"] = freq_var
                freq_entry = ttk.Entry(freq_frame, textvariable=freq_var, width=10)
                freq_entry.pack(side=tk.LEFT, padx=(0, 15))
                freq_entry.bind('<Return>', self.validate_timer_values)
                
                # 占空比输入
                duty_frame = ttk.Frame(timer_frame)
                duty_frame.pack(fill=tk.X, pady=2)
                ttk.Label(duty_frame, text="占空比(%):", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))
                duty_var = tk.StringVar(value="50")
                self.timer_vars[f"tim{tim_num}_duty"] = duty_var
                duty_entry = ttk.Entry(duty_frame, textvariable=duty_var, width=10)
                duty_entry.pack(side=tk.LEFT, padx=(0, 15))
                duty_entry.bind('<Return>', self.validate_timer_values)
                
                # 启用/禁用复选框和状态
                control_frame = ttk.Frame(timer_frame)
                control_frame.pack(fill=tk.X, pady=2)
                
                enable_var = tk.BooleanVar(value=False)
                self.timer_vars[f"tim{tim_num}_enable"] = enable_var
                enable_check = ttk.Checkbutton(control_frame, text="启用", variable=enable_var)
                enable_check.pack(side=tk.LEFT, padx=(0, 10))
                
                # 状态显示
                status_var = tk.StringVar(value="禁用")
                self.timer_vars[f"tim{tim_num}_status"] = status_var
                status_label = ttk.Label(control_frame, textvariable=status_var, font=("Arial", 8), foreground="gray")
                status_label.pack(side=tk.LEFT)
                
                # 绑定启用状态变化
                enable_var.trace('w', lambda *args, tim=tim_num: self.on_timer_enable_changed(tim))
            
            # 定时器快速设置按钮
            timer_quick_frame = ttk.Frame(self.timer_frame)
            timer_quick_frame.pack(fill=tk.X, pady=(10, 0))
            
            ttk.Label(timer_quick_frame, text="快速设置:").pack()
            
            timer_button_row1 = ttk.Frame(timer_quick_frame)
            timer_button_row1.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Button(timer_button_row1, text="全部禁用", command=self.disable_all_timers).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(timer_button_row1, text="启用所有", command=self.enable_all_timers).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(timer_button_row1, text="1KHz 50%", command=lambda: self.set_all_timers_preset(1000, 50)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(timer_button_row1, text="10KHz 25%", command=lambda: self.set_all_timers_preset(10000, 25)).pack(side=tk.LEFT, padx=(0, 5))
            
            # 定时器自定义预设
            timer_custom_frame = ttk.Frame(self.timer_frame)
            timer_custom_frame.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Label(timer_custom_frame, text="自定义预设:").pack()
            
            custom_input_frame = ttk.Frame(timer_custom_frame)
            custom_input_frame.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Label(custom_input_frame, text="频率:").pack(side=tk.LEFT, padx=(0, 2))
            self.custom_timer_freq_var = tk.StringVar(value="5000")
            freq_custom_entry = ttk.Entry(custom_input_frame, textvariable=self.custom_timer_freq_var, width=8)
            freq_custom_entry.pack(side=tk.LEFT, padx=(0, 5))
            
            ttk.Label(custom_input_frame, text="Hz 占空比:").pack(side=tk.LEFT, padx=(5, 2))
            self.custom_timer_duty_var = tk.StringVar(value="30")
            duty_custom_entry = ttk.Entry(custom_input_frame, textvariable=self.custom_timer_duty_var, width=6)
            duty_custom_entry.pack(side=tk.LEFT, padx=(0, 2))
            
            ttk.Label(custom_input_frame, text="%").pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(custom_input_frame, text="应用自定义", command=self.apply_custom_timer_preset).pack(side=tk.LEFT, padx=(5, 0))
            
            # 定时器发送控制
            timer_send_frame = ttk.Frame(self.timer_frame)
            timer_send_frame.pack(fill=tk.X, pady=(10, 0))
            
            ttk.Button(timer_send_frame, text="发送定时器配置", command=self.send_timer_config).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(timer_send_frame, text="停止所有定时器", command=self.stop_all_timers).pack(side=tk.LEFT, padx=(0, 10))
            
            self.timer_send_status_var = tk.StringVar(value="等待发送...")
            ttk.Label(timer_send_frame, textvariable=self.timer_send_status_var, font=("Arial", 9), foreground="blue").pack(pady=(5, 0))
            
            # PWM快速设置按钮（全通道模式）
            quick_set_frame = ttk.Frame(send_control_frame)
            quick_set_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(quick_set_frame, text="快速设置:").pack()
            
            quick_button_row = ttk.Frame(quick_set_frame)
            quick_button_row.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Button(quick_button_row, text="全部清零", command=lambda: self.set_all_pwm(0)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_button_row, text="全部25", command=lambda: self.set_all_pwm(25)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_button_row, text="全部50", command=lambda: self.set_all_pwm(50)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_button_row, text="全部100", command=lambda: self.set_all_pwm(100)).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_button_row, text="示例数据", command=self.set_example_pwm).pack(side=tk.LEFT, padx=(0, 5))
            
            # 全通道模式自定义数值设置
            custom_pwm_frame = ttk.Frame(send_control_frame)
            custom_pwm_frame.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Label(custom_pwm_frame, text="自定义数值:").pack(side=tk.LEFT, padx=(0, 5))
            custom_pwm_entry = ttk.Entry(custom_pwm_frame, textvariable=self.custom_value_var, width=8)
            custom_pwm_entry.pack(side=tk.LEFT, padx=(0, 5))
            custom_pwm_entry.bind('<Return>', self.apply_custom_pwm)
            ttk.Button(custom_pwm_frame, text="应用到全部PWM", command=self.apply_custom_pwm).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Label(custom_pwm_frame, text="(0-100)", font=("Arial", 8), foreground="gray").pack(side=tk.LEFT)
            
            # 发送控制区域
            send_cmd_frame = ttk.LabelFrame(self.left_scrollable_frame, text="发送控制", padding="10")
            send_cmd_frame.pack(fill=tk.X, pady=(0, 10))
            
            # 发送格式选择
            format_frame = ttk.Frame(send_cmd_frame)
            format_frame.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Label(format_frame, text="发送格式:").pack(side=tk.LEFT, padx=(0, 5))
            self.send_format_var = tk.StringVar(value="PWM数据")
            format_combo = ttk.Combobox(format_frame, textvariable=self.send_format_var, width=15,
                                        values=["PWM数据", "自定义十六进制", "自定义文本"])
            format_combo.pack(side=tk.LEFT, padx=(0, 10))
            format_combo.state(['readonly'])
            format_combo.bind('<<ComboboxSelected>>', self.on_send_format_changed)
            
            # 发送按钮和选项
            send_button_frame = ttk.Frame(send_cmd_frame)
            send_button_frame.pack(fill=tk.X, pady=(5, 0))
            
            self.send_button = ttk.Button(send_button_frame, text="发送数据", command=self.send_serial_data, state=tk.DISABLED)
            self.send_button.pack(side=tk.LEFT, padx=(0, 10))
            
            self.verify_echo_var = tk.BooleanVar(value=True)  # 默认开启回显验证
            ttk.Checkbutton(send_button_frame, text="验证回显", variable=self.verify_echo_var).pack(side=tk.LEFT, padx=(0, 10))
            
            self.show_timing_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(send_button_frame, text="显示耗时", variable=self.show_timing_var).pack(side=tk.LEFT)
            
            # 自定义数据输入区域（默认隐藏）
            self.custom_data_frame = ttk.Frame(send_cmd_frame)
            self.custom_data_frame.pack(fill=tk.X, pady=(5, 0))
            self.custom_data_frame.pack_forget()
            
            ttk.Label(self.custom_data_frame, text="自定义数据:").pack()
            self.custom_data_var = tk.StringVar()
            self.custom_data_entry = ttk.Entry(self.custom_data_frame, textvariable=self.custom_data_var)
            self.custom_data_entry.pack(fill=tk.X, pady=(5, 0))
            
            # 发送统计显示
            stats_frame = ttk.Frame(send_cmd_frame)
            stats_frame.pack(fill=tk.X, pady=(10, 0))
            
            self.send_stats_label = ttk.Label(stats_frame, text="发送统计: 总计0次 | 错误0次 | 最后发送耗时: 0ms")
            self.send_stats_label.pack()
            
            stats_button_frame = ttk.Frame(stats_frame)
            stats_button_frame.pack(fill=tk.X, pady=(5, 0))
            
            ttk.Button(stats_button_frame, text="重置统计", command=self.reset_send_stats).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(stats_button_frame, text="查看时间历史", command=self.show_timing_history).pack(side=tk.LEFT)
        
        # 状态信息
        status_frame = ttk.LabelFrame(self.left_scrollable_frame, text="状态信息", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="串口控制工具就绪")
        self.status_label.pack()
        
        # === 串口数据接收区域 ===
        if SERIAL_AVAILABLE:
            serial_data_frame = ttk.LabelFrame(right_frame, text="串口数据接收区", padding="10")
            serial_data_frame.pack(fill=tk.BOTH, expand=True)
            
            # 创建滚动文本框
            self.serial_text = scrolledtext.ScrolledText(serial_data_frame, height=30, width=80, 
                                                         font=("Consolas", 9))
            self.serial_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
            # 数据接收控制
            serial_control_row = ttk.Frame(serial_data_frame)
            serial_control_row.pack(fill=tk.X)
            
            # 第一行控制
            control_row1 = ttk.Frame(serial_control_row)
            control_row1.pack(fill=tk.X, pady=(0, 5))
            
            # 显示格式选择
            ttk.Label(control_row1, text="显示格式:").pack(side=tk.LEFT, padx=(0, 5))
            self.display_format_var = tk.StringVar(value="文本")
            format_combo = ttk.Combobox(control_row1, textvariable=self.display_format_var, width=8,
                                        values=["文本", "十六进制", "十进制"])
            format_combo.pack(side=tk.LEFT, padx=(0, 10))
            format_combo.state(['readonly'])
            
            # 自动滚动选项
            self.auto_scroll_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(control_row1, text="自动滚动", variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=(0, 10))
            
            # 时间戳选项
            self.timestamp_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(control_row1, text="显示时间戳", variable=self.timestamp_var).pack(side=tk.LEFT, padx=(0, 10))
            
            # 数据统计
            self.data_count_label = ttk.Label(control_row1, text="接收: 0 字节")
            self.data_count_label.pack(side=tk.RIGHT)
    
    # === 新增自定义数值方法 ===
    def apply_custom_four_channel(self, event=None):
        """应用自定义数值到四通道模式"""
        try:
            value = int(self.custom_value_var.get())
            if value < 0 or value > 100:
                messagebox.showwarning("警告", "自定义数值必须在0-100之间")
                return
            
            for var in self.four_channel_vars.values():
                var.set(str(value))
            self.sync_four_channel_to_pwm()
            print(f"四通道模式：已应用自定义数值 {value} 到所有通道")
            
        except ValueError:
            messagebox.showerror("错误", "自定义数值必须是整数")
    
    def apply_custom_white_light(self, event=None):
        """应用自定义数值到白光模式"""
        try:
            value = int(self.custom_value_var.get())
            if value < 0 or value > 100:
                messagebox.showwarning("警告", "自定义数值必须在0-100之间")
                return
            
            for var in self.white_light_vars.values():
                var.set(str(value))
            self.sync_white_light_to_pwm()
            print(f"白光模式：已应用自定义数值 {value} 到所有通道")
            
        except ValueError:
            messagebox.showerror("错误", "自定义数值必须是整数")
    
    def apply_custom_pwm(self, event=None):
        """应用自定义数值到全通道PWM模式"""
        try:
            value = int(self.custom_value_var.get())
            if value < 0 or value > 100:
                messagebox.showwarning("警告", "自定义数值必须在0-100之间")
                return
            
            for var in self.pwm_vars:
                var.set(str(value))
            print(f"全通道模式：已应用自定义数值 {value} 到所有PWM通道")
            
        except ValueError:
            messagebox.showerror("错误", "自定义数值必须是整数")
    
    # === 新增定时器控制方法 ===
    def validate_timer_values(self, event=None):
        """验证定时器参数的有效性"""
        try:
            for tim_num in [1, 2, 3, 4]:
                # 验证频率
                freq = float(self.timer_vars[f"tim{tim_num}_freq"].get())
                if freq <= 0 or freq > 1000000:  # 最大1MHz
                    self.timer_vars[f"tim{tim_num}_freq"].set("1000")
                    messagebox.showwarning("警告", f"TIM{tim_num}频率必须在0-1000000Hz之间")
                    return False
                
                # 验证占空比
                duty = float(self.timer_vars[f"tim{tim_num}_duty"].get())
                if duty < 0 or duty > 100:
                    self.timer_vars[f"tim{tim_num}_duty"].set("50")
                    messagebox.showwarning("警告", f"TIM{tim_num}占空比必须在0-100%之间")
                    return False
            return True
        except ValueError:
            messagebox.showerror("错误", "定时器参数必须是数字")
            return False
    
    def on_timer_enable_changed(self, tim_num):
        """定时器启用状态改变时的处理"""
        enable_var = self.timer_vars[f"tim{tim_num}_enable"]
        status_var = self.timer_vars[f"tim{tim_num}_status"]
        
        if enable_var.get():
            status_var.set("启用")
        else:
            status_var.set("禁用")
    
    def disable_all_timers(self):
        """禁用所有定时器"""
        for tim_num in [1, 2, 3, 4]:
            self.timer_vars[f"tim{tim_num}_enable"].set(False)
        print("已禁用所有定时器")
    
    def enable_all_timers(self):
        """启用所有定时器"""
        if self.validate_timer_values():
            for tim_num in [1, 2, 3, 4]:
                self.timer_vars[f"tim{tim_num}_enable"].set(True)
            print("已启用所有定时器")
    
    def set_all_timers_preset(self, frequency, duty_cycle):
        """设置所有定时器为预设值"""
        for tim_num in [1, 2, 3, 4]:
            self.timer_vars[f"tim{tim_num}_freq"].set(str(frequency))
            self.timer_vars[f"tim{tim_num}_duty"].set(str(duty_cycle))
            self.timer_vars[f"tim{tim_num}_enable"].set(True)
        print(f"已设置所有定时器为: {frequency}Hz, {duty_cycle}%")
    
    def apply_custom_timer_preset(self):
        """应用自定义定时器预设"""
        try:
            frequency = float(self.custom_timer_freq_var.get())
            duty_cycle = float(self.custom_timer_duty_var.get())
            
            if frequency <= 0 or frequency > 1000000:
                messagebox.showwarning("警告", "频率必须在0-1000000Hz之间")
                return
            
            if duty_cycle < 0 or duty_cycle > 100:
                messagebox.showwarning("警告", "占空比必须在0-100%之间")
                return
            
            self.set_all_timers_preset(int(frequency), int(duty_cycle))
            
        except ValueError:
            messagebox.showerror("错误", "自定义预设参数必须是数字")
    
    def send_timer_config(self):
        """发送定时器配置到STM32"""
        if not self.is_serial_open or not self.serial_port:
            messagebox.showerror("错误", "串口未打开")
            return
        
        if not self.validate_timer_values():
            return
        
        try:
            # 构建定时器配置数据包
            data_packet = bytearray()
            data_packet.append(0xA0)  # 命令头
            
            for tim_num in [1, 2, 3, 4]:
                enable = 1 if self.timer_vars[f"tim{tim_num}_enable"].get() else 0
                frequency = int(float(self.timer_vars[f"tim{tim_num}_freq"].get()))
                duty = int(float(self.timer_vars[f"tim{tim_num}_duty"].get()))
                
                data_packet.append(enable)                    # 启用标志
                data_packet.append((frequency >> 16) & 0xFF) # 频率高16位的高8位
                data_packet.append((frequency >> 8) & 0xFF)  # 频率高16位的低8位
                data_packet.append(frequency & 0xFF)         # 频率低8位
                data_packet.append(duty)                     # 占空比
            
            # 计算校验和
            checksum = sum(data_packet) & 0xFF
            data_packet.append(checksum)
            
            # 发送数据
            bytes_sent = self.serial_port.write(data_packet)
            self.serial_port.flush()
            
            # 显示发送的数据
            if not self.performance_mode.get():
                self.display_timer_sent_data(data_packet)
            
            # 更新状态
            enabled_timers = [f"TIM{i}" for i in [1,2,3,4] if self.timer_vars[f"tim{i}_enable"].get()]
            if enabled_timers:
                status_text = f"已发送配置: {', '.join(enabled_timers)}"
            else:
                status_text = "已发送配置: 所有定时器禁用"
            
            self.timer_send_status_var.set(status_text)
            print(f"定时器配置发送成功: {len(data_packet)} 字节")
            
        except Exception as e:
            self.timer_send_status_var.set("发送失败!")
            messagebox.showerror("错误", f"发送定时器配置失败: {e}")
    
    def stop_all_timers(self):
        """停止所有定时器"""
        if not self.is_serial_open or not self.serial_port:
            messagebox.showerror("错误", "串口未打开")
            return
        
        try:
            # 发送停止命令: 0xA1
            stop_cmd = bytearray([0xA1, 0xA1])  # 停止命令 + 校验
            bytes_sent = self.serial_port.write(stop_cmd)
            self.serial_port.flush()
            
            # 更新界面状态
            for tim_num in [1, 2, 3, 4]:
                self.timer_vars[f"tim{tim_num}_enable"].set(False)
            
            self.timer_send_status_var.set("已停止所有定时器")
            self.add_to_serial_display("[发送] 停止所有定时器命令: A1 A1", "send")
            print("已发送停止所有定时器命令")
            
        except Exception as e:
            messagebox.showerror("错误", f"停止定时器失败: {e}")
    
    def display_timer_sent_data(self, data):
        """显示发送的定时器配置数据"""
        # 解析并显示定时器数据
        enabled_timers = []
        for i, tim_num in enumerate([1, 2, 3, 4]):
            offset = 1 + i * 5  # 每个定时器占5字节
            if data[offset] == 1:  # 启用状态
                frequency = (data[offset+1] << 16) | (data[offset+2] << 8) | data[offset+3]
                duty = data[offset+4]
                enabled_timers.append(f"TIM{tim_num}({frequency}Hz,{duty}%)")
        
        if enabled_timers:
            timer_info = ', '.join(enabled_timers)
            self.add_to_serial_display(f"[发送] 定时器配置: {timer_info}", "send")
        else:
            self.add_to_serial_display(f"[发送] 定时器配置: 所有定时器禁用", "send")
        
        self.add_to_serial_display(f"[发送] 十六进制: {data.hex(' ').upper()}", "send")

    def set_all_white_light_custom(self, value):
        """设置所有白光通道为指定值（新增）"""
        for var in self.white_light_vars.values():
            var.set(str(value))
        self.sync_white_light_to_pwm()
    
    # === 新增回显验证优化方法 ===
    def update_echo_timeout(self, event=None):
        """更新回显超时时间"""
        try:
            timeout = float(self.echo_timeout_var.get())
            if 0.1 <= timeout <= 5.0:
                self.echo_timeout = timeout
                print(f"回显超时设置为: {timeout} 秒")
            else:
                self.echo_timeout_var.set(str(self.echo_timeout))
                messagebox.showwarning("警告", "回显超时时间必须在0.1-5.0秒之间")
        except ValueError:
            self.echo_timeout_var.set(str(self.echo_timeout))
            messagebox.showerror("错误", "回显超时时间必须是数字")
    
    def start_echo_verification(self, expected_data):
        """开始回显验证"""
        with self.echo_verification_lock:
            self.expected_echo_data = expected_data
            self.echo_buffer.clear()
            self.echo_start_time = time.perf_counter()
    
    def process_echo_data(self, data):
        """处理接收到的数据，用于回显验证"""
        with self.echo_verification_lock:
            if self.expected_echo_data is None:
                return False
            
            # 将新数据添加到回显缓冲区
            self.echo_buffer.extend(data)
            
            # 检查是否收到了完整的回显数据
            expected_len = len(self.expected_echo_data)
            
            if len(self.echo_buffer) >= expected_len:
                # 检查前expected_len字节是否匹配
                received_echo = bytes(self.echo_buffer[:expected_len])
                
                if received_echo == self.expected_echo_data:
                    # 回显验证成功
                    elapsed_time = (time.perf_counter() - self.echo_start_time) * 1000
                    
                    # 检查是否还有结束符
                    if len(self.echo_buffer) > expected_len:
                        end_chars = self.echo_buffer[expected_len:]
                        if not self.performance_mode.get():
                            try:
                                end_str = bytes(end_chars).decode('utf-8', errors='replace')
                                self.root.after(0, lambda: self.add_to_serial_display(f"[接收] 结束符: {repr(end_str)}", "receive"))
                            except:
                                pass
                    
                    # 显示验证结果
                    if not self.performance_mode.get():
                        self.root.after(0, lambda: self.add_to_serial_display(
                            f"[接收] 回显数据 ({len(received_echo)} 字节): {received_echo.hex(' ').upper()}", "receive"))
                        self.root.after(0, lambda: self.add_to_serial_display("[验证] 成功: 回显数据与发送数据完全一致!", "success"))
                        
                        if self.show_timing_var.get():
                            self.root.after(0, lambda: self.add_to_serial_display(
                                f"[时间] 发送到接收完整回显耗时: {elapsed_time:.2f} 毫秒", "info"))
                    
                    # 更新发送统计
                    self.send_stats['last_send_time'] = elapsed_time
                    self.root.after(0, self.update_send_stats_display)
                    
                    # 清除验证状态
                    self.expected_echo_data = None
                    self.echo_buffer.clear()
                    return True
                else:
                    # 回显不匹配
                    elapsed_time = (time.perf_counter() - self.echo_start_time) * 1000
                    
                    if not self.performance_mode.get():
                        self.root.after(0, lambda: self.add_to_serial_display(
                            f"[接收] 回显数据 ({len(received_echo)} 字节): {received_echo.hex(' ').upper()}", "receive"))
                        self.root.after(0, lambda: self.add_to_serial_display("[验证] 失败: 回显数据与发送数据不匹配!", "error"))
                    
                    self.send_stats['send_errors'] += 1
                    self.root.after(0, self.update_send_stats_display)
                    
                    # 清除验证状态
                    self.expected_echo_data = None
                    self.echo_buffer.clear()
                    return True
            
            # 检查超时
            elif time.perf_counter() - self.echo_start_time > self.echo_timeout:
                elapsed_time = (time.perf_counter() - self.echo_start_time) * 1000
                
                if not self.performance_mode.get():
                    if len(self.echo_buffer) > 0:
                        partial_data = bytes(self.echo_buffer)
                        self.root.after(0, lambda: self.add_to_serial_display(
                            f"[接收] 部分回显 ({len(partial_data)} 字节): {partial_data.hex(' ').upper()}", "receive"))
                    
                    self.root.after(0, lambda: self.add_to_serial_display(
                        f"[验证] 超时: 在{self.echo_timeout}秒内未收到完整回显数据", "warning"))
                    
                    if self.show_timing_var.get():
                        self.root.after(0, lambda: self.add_to_serial_display(
                            f"[时间] 发送耗时: {elapsed_time:.2f} 毫秒", "info"))
                
                self.send_stats['send_errors'] += 1
                self.root.after(0, self.update_send_stats_display)
                
                # 清除验证状态
                self.expected_echo_data = None
                self.echo_buffer.clear()
                return True
            
            return False
    
    # === 串口相关方法 ===
    def clear_hardware_buffers(self):
        """清空串口硬件缓冲区"""
        if not self.is_serial_open or not self.serial_port:
            messagebox.showwarning("警告", "串口未打开")
            return
        
        try:
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            self.serial_buffer = []
            self.receive_byte_count = 0
            
            # 清除回显验证状态
            with self.echo_verification_lock:
                self.expected_echo_data = None
                self.echo_buffer.clear()
            
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
            echo_buffer_size = len(self.echo_buffer)
            
            if self.is_serial_open and self.serial_port:
                try:
                    hw_buffer_size = self.serial_port.in_waiting
                except:
                    pass
            
            status_text = f"缓冲区: {buffer_size}/{self.max_buffer_size} | 硬件: {hw_buffer_size} | 回显: {echo_buffer_size}"
            self.buffer_status_label.config(text=status_text)
    
    def show_timing_history(self):
        """显示发送时间历史记录"""
        if not self.send_stats['send_history']:
            messagebox.showinfo("信息", "暂无时间历史记录")
            return
        
        history_window = tk.Toplevel(self.root)
        history_window.title("发送时间历史记录")
        history_window.geometry("600x400")
        
        text_widget = scrolledtext.ScrolledText(history_window, font=("Consolas", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
            
            for i, timing in enumerate(reversed(history[-50:]), 1):
                text_widget.insert(tk.END, f"第{len(history)-i+1}次: {timing:.2f} ms\n")
        
        text_widget.config(state=tk.DISABLED)
    
    # === 四通道模式相关方法 ===
    def validate_four_channel_values(self, event=None):
        """验证四通道模式PWM值的有效性"""
        try:
            for ch_index, var in self.four_channel_vars.items():
                value = int(var.get())
                if value < 0 or value > 100:
                    var.set("0")
                    messagebox.showwarning("警告", f"CH{ch_index+1}的值必须在0-100之间")
                    return False
            return True
        except ValueError:
            messagebox.showerror("错误", "四通道PWM值必须是整数")
            return False
    
    def on_four_channel_scale_changed(self, ch_index, value):
        """四通道滑块值改变时的处理"""
        try:
            self.four_channel_vars[ch_index].set(str(int(float(value))))
        except:
            pass
    
    def sync_four_channel_entry_from_scale(self, ch_index, event):
        """从滑块同步到输入框（鼠标释放时）"""
        try:
            scale_widget = event.widget
            value = scale_widget.get()
            self.four_channel_vars[ch_index].set(str(int(value)))
            self.sync_four_channel_to_pwm()
        except:
            pass
    
    def sync_four_channel_scale_from_entry(self, ch_index):
        """从输入框同步到滑块"""
        try:
            value = int(self.four_channel_vars[ch_index].get())
            if 0 <= value <= 100:
                # 找到对应的滑块并设置值
                for widget in self.four_channel_frame.winfo_children():
                    if isinstance(widget, ttk.Frame):
                        for child in widget.winfo_children():
                            if isinstance(child, ttk.Scale):
                                child.set(value)
                                break
                self.sync_four_channel_to_pwm()
        except:
            pass
    
    def sync_four_channel_to_pwm(self):
        """将四通道模式的值同步到全通道PWM数组"""
        try:
            # 首先清零所有通道
            for i, var in enumerate(self.pwm_vars):
                var.set("0")
            
            # 设置四个指定通道的值
            for ch_index, var in self.four_channel_vars.items():
                try:
                    value = int(var.get())
                    if 0 <= value <= 100:
                        self.pwm_vars[ch_index].set(str(value))
                except:
                    pass
        except Exception as e:
            print(f"同步四通道到PWM失败: {e}")
    
    def clear_all_four_channel(self):
        """清空所有四通道"""
        for var in self.four_channel_vars.values():
            var.set("0")
        self.sync_four_channel_to_pwm()
    
    def set_all_four_channel(self, value):
        """设置所有四通道为指定值"""
        for var in self.four_channel_vars.values():
            var.set(str(value))
        self.sync_four_channel_to_pwm()
    
    def set_four_channel_example(self):
        """设置四通道示例数据"""
        example_values = {1: 25, 5: 50, 11: 75, 13: 100}  # CH2=25, CH6=50, CH12=75, CH14=100
        for ch_index, var in self.four_channel_vars.items():
            value = example_values.get(ch_index, 0)
            var.set(str(value))
        self.sync_four_channel_to_pwm()
    
    # === 串口发送相关方法 ===
    def on_control_mode_changed(self, event=None):
        """控制模式改变时的处理"""
        mode = self.control_mode_var.get()
        if mode == "四通道模式":
            self.four_channel_frame.pack(fill=tk.X, pady=(0, 10))
            self.pwm_frame.pack_forget()
            self.white_light_frame.pack_forget()
            self.timer_frame.pack_forget()
            self.sync_four_channel_to_pwm()
        elif mode == "全通道模式":
            self.four_channel_frame.pack_forget()
            self.pwm_frame.pack(fill=tk.X, pady=(0, 10))
            self.white_light_frame.pack_forget()
            self.timer_frame.pack_forget()
        elif mode == "白光模式":
            self.four_channel_frame.pack_forget()
            self.pwm_frame.pack_forget()
            self.white_light_frame.pack(fill=tk.X, pady=(0, 10))
            self.timer_frame.pack_forget()
            self.sync_white_light_to_pwm()
        elif mode == "定时器频率控制":
            self.four_channel_frame.pack_forget()
            self.pwm_frame.pack_forget()
            self.white_light_frame.pack_forget()
            self.timer_frame.pack(fill=tk.X, pady=(0, 10))
    
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
            self.white_light_vars[ch_index].set(str(int(float(value))))
        except:
            pass
    
    def sync_white_light_entry_from_scale(self, ch_index, event):
        """从滑块同步到输入框（鼠标释放时）"""
        try:
            scale_widget = event.widget
            value = scale_widget.get()
            self.white_light_vars[ch_index].set(str(int(value)))
            self.sync_white_light_to_pwm()
        except:
            pass
    
    def sync_white_light_scale_from_entry(self, ch_index):
        """从输入框同步到滑块"""
        try:
            value = int(self.white_light_vars[ch_index].get())
            if 0 <= value <= 100:
                # 找到对应的滑块并设置值
                for widget in self.white_light_frame.winfo_children():
                    if isinstance(widget, ttk.Frame):
                        for child in widget.winfo_children():
                            if isinstance(child, ttk.Scale):
                                child.set(value)
                                break
                self.sync_white_light_to_pwm()
        except:
            pass
    
    def sync_white_light_to_pwm(self):
        """将白光模式的值同步到全通道PWM数组"""
        try:
            for i, var in enumerate(self.pwm_vars):
                var.set("0")
            
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
            "warm": {"6": 80, "7": 20, "10": 0, "11": 0},
            "cool": {"6": 0, "7": 0, "10": 80, "11": 20},
            "mixed": {"6": 50, "7": 30, "10": 30, "11": 50}
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
            self.custom_data_frame.pack_forget()
        else:
            self.custom_data_frame.pack(fill=tk.X, pady=(5, 0))
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
            current_mode = self.control_mode_var.get()
            
            if current_mode == "四通道模式":
                if not self.validate_four_channel_values():
                    return None
                self.sync_four_channel_to_pwm()
            elif current_mode == "白光模式":
                if not self.validate_white_light_values():
                    return None
                self.sync_white_light_to_pwm()
            elif current_mode == "全通道模式":
                if not self.validate_pwm_values():
                    return None
            elif current_mode == "定时器频率控制":
                # 定时器模式使用专门的发送方法
                messagebox.showinfo("提示", "定时器模式请使用'发送定时器配置'按钮")
                return None
            
            pwm_values = []
            for var in self.pwm_vars:
                try:
                    value = int(var.get())
                    pwm_values.append(max(0, min(100, value)))
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
        """优化的串口数据发送方法"""
        if not self.is_serial_open or not self.serial_port:
            messagebox.showerror("错误", "串口未打开")
            return
        
        data_to_send = self.prepare_send_data()
        if data_to_send is None:
            return
        
        try:
            # 发送前清空缓冲区
            if self.clear_before_send.get():
                self.serial_port.reset_input_buffer()
                self.serial_port.reset_output_buffer()
                if not self.performance_mode.get():
                    self.add_to_serial_display("[系统] 发送前已清空缓冲区", "info")
            
            # 如果启用回显验证，先设置验证状态
            if self.verify_echo_var.get():
                self.start_echo_verification(data_to_send)
            
            # 发送数据并记录时间
            start_time = time.perf_counter()
            bytes_sent = self.serial_port.write(data_to_send)
            self.serial_port.flush()
            
            # 更新发送统计
            self.send_stats['total_sent'] += 1
            
            # 显示发送的数据
            if not self.performance_mode.get():
                self.display_sent_data(data_to_send)
            
            # 如果不启用回显验证，直接计算发送时间
            if not self.verify_echo_var.get():
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                self.send_stats['last_send_time'] = elapsed_ms
                self.send_stats['send_history'].append(elapsed_ms)
                if len(self.send_stats['send_history']) > 100:
                    self.send_stats['send_history'] = self.send_stats['send_history'][-100:]
                
                if self.show_timing_var.get() and not self.performance_mode.get():
                    self.add_to_serial_display(f"[时间] 发送耗时: {elapsed_ms:.2f} 毫秒", "info")
                
                self.update_send_stats_display()
            
            print(f"数据发送成功: {len(data_to_send)} 字节")
            
        except serial.SerialException as e:
            self.send_stats['send_errors'] += 1
            self.update_send_stats_display()
            messagebox.showerror("错误", f"串口发送失败: {e}")
        except Exception as e:
            self.send_stats['send_errors'] += 1
            self.update_send_stats_display()
            messagebox.showerror("错误", f"发送数据失败: {e}")
    
    def display_sent_data(self, data):
        """在接收区显示发送的数据"""
        format_type = self.send_format_var.get()
        
        if format_type == "PWM数据":
            current_mode = self.control_mode_var.get()
            
            if current_mode == "四通道模式":
                # 显示四通道数据
                four_channels = {1: "CH2", 5: "CH6", 11: "CH12", 13: "CH14"}
                active_channels = []
                for i, value in enumerate(data[:16]):
                    if value > 0 and i in four_channels:
                        active_channels.append(f'{four_channels[i]}:{value}')
                
                if active_channels:
                    four_str = ', '.join(active_channels)
                    self.add_to_serial_display(f"[发送] 四通道数据: {four_str}", "send")
                else:
                    self.add_to_serial_display(f"[发送] 四通道数据: 全部关闭", "send")
                    
            elif current_mode == "白光模式":
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
                # 全通道模式
                pwm_str = ', '.join([f'CH{i+1}:{data[i]}' for i in range(min(16, len(data)))])
                self.add_to_serial_display(f"[发送] PWM数据 ({len(data)} 字节): {pwm_str}", "send")
            
            self.add_to_serial_display(f"[发送] 十六进制: {data.hex(' ').upper()}", "send")
            
        elif format_type == "自定义十六进制":
            self.add_to_serial_display(f"[发送] 十六进制 ({len(data)} 字节): {data.hex(' ').upper()}", "send")
        elif format_type == "自定义文本":
            try:
                text_data = data.decode('utf-8')
                self.add_to_serial_display(f"[发送] 文本 ({len(data)} 字节): {text_data}", "send")
            except:
                self.add_to_serial_display(f"[发送] 数据 ({len(data)} 字节): {data.hex(' ').upper()}", "send")
    
    def add_to_serial_display(self, text, msg_type="normal"):
        """添加格式化文本到串口显示区"""
        if self.performance_mode.get() and msg_type in ["send", "receive"]:
            return
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3] if self.timestamp_var.get() else ""
        
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
        
        if timestamp:
            display_text = f"[{timestamp}] {prefix}{text}\n"
        else:
            display_text = f"{prefix}{text}\n"
        
        self.serial_text.insert(tk.END, display_text)
        
        if self.auto_scroll_var.get():
            self.serial_text.see(tk.END)
        
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
    
    # === 串口接收相关方法（优化版） ===
    def refresh_serial_ports(self):
        """刷新串口端口列表"""
        if not SERIAL_AVAILABLE:
            return
        
        try:
            ports = serial.tools.list_ports.comports()
            latest_port = None
            port_names = []
            
            if ports:
                sorted_ports = self.sort_ports_by_insertion_time(ports)
                port_names = [port.device for port in sorted_ports]
                latest_port = port_names[0] if port_names else None
                print(f"发现串口: {port_names}")
                if latest_port:
                    print(f"检测到最新插入的串口: {latest_port}")
            
            self.port_combo['values'] = port_names
            
            if port_names:
                if not self.port_var.get() or self.port_var.get() not in port_names:
                    self.port_var.set(latest_port)
            else:
                self.port_var.set("")
                
        except Exception as e:
            print(f"刷新串口列表失败: {e}")
    
    def sort_ports_by_insertion_time(self, ports):
        """按设备插入时间排序，最新的在前"""
        try:
            def get_port_priority(port):
                priority = 0
                
                if hasattr(port, 'vid') and port.vid is not None:
                    priority += 1000
                
                if hasattr(port, 'product') and port.product:
                    priority += 100
                
                if hasattr(port, 'manufacturer') and port.manufacturer:
                    priority += 50
                
                if hasattr(port, 'serial_number') and port.serial_number:
                    priority += 25
                
                import re
                match = re.search(r'COM(\d+)', port.device.upper())
                com_num = int(match.group(1)) if match else 0
                priority += com_num
                
                return priority
            
            return sorted(ports, key=get_port_priority, reverse=True)
            
        except Exception as e:
            print(f"端口排序失败: {e}")
            return list(reversed(ports))
    
    def auto_open_latest_serial(self):
        """自动打开最新插入的串口"""
        if not SERIAL_AVAILABLE:
            return
        
        try:
            if self.port_var.get():
                print(f"尝试自动打开最新插入的串口: {self.port_var.get()}")
                self.open_serial_port(auto_open=True)
        except Exception as e:
            print(f"自动打开串口失败: {e}")
    
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
            
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.001,  # 进一步减少超时时间
                write_timeout=0.05,  # 减少写入超时
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
            
            self.is_serial_open = True
            
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            self.serial_thread = threading.Thread(target=self.serial_receive_loop, daemon=True)
            self.serial_thread.start()
            
            self.serial_open_button.config(state=tk.DISABLED)
            self.serial_close_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self.serial_status_label.config(text=f"串口状态: 已连接 ({port}, {baudrate})", foreground="green")
            
            self.serial_text.delete(1.0, tk.END)
            self.serial_buffer = []
            self.receive_byte_count = 0
            self.data_count_label.config(text="接收: 0 字节")
            
            # 清除回显验证状态
            with self.echo_verification_lock:
                self.expected_echo_data = None
                self.echo_buffer.clear()
            
            self.update_buffer_status()
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
            
            self.serial_open_button.config(state=tk.NORMAL)
            self.serial_close_button.config(state=tk.DISABLED)
            self.send_button.config(state=tk.DISABLED)
            self.serial_status_label.config(text="串口状态: 未连接", foreground="red")
            
            # 清除回显验证状态
            with self.echo_verification_lock:
                self.expected_echo_data = None
                self.echo_buffer.clear()
            
            self.update_buffer_status()
            
            print("串口已关闭")
            
        except Exception as e:
            messagebox.showerror("错误", f"关闭串口失败: {str(e)}")
    
    def serial_receive_loop(self):
        """优化的串口数据接收循环"""
        while self.is_serial_open and self.serial_port and self.serial_port.is_open:
            try:
                # 检查是否有数据可读
                if self.serial_port.in_waiting > 0:
                    # 一次性读取所有可用数据
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    if data:
                        self.serial_buffer.extend(data)
                        self.receive_byte_count += len(data)
                        
                        # 处理回显验证
                        echo_processed = False
                        if self.verify_echo_var.get():
                            echo_processed = self.process_echo_data(data)
                        
                        # 如果不是回显验证模式或回显验证已完成，正常显示数据
                        if not self.verify_echo_var.get() or echo_processed:
                            if not self.performance_mode.get():
                                self.root.after(0, lambda d=data: self.update_serial_display(d))
                        
                        # 管理缓冲区大小
                        if len(self.serial_buffer) > self.max_buffer_size:
                            self.serial_buffer = self.serial_buffer[-self.max_buffer_size:]
                        
                        if self.auto_clear_buffer.get() and len(self.serial_buffer) >= self.max_buffer_size * 0.9:
                            self.serial_buffer = self.serial_buffer[-100:]
                            if not self.performance_mode.get():
                                self.root.after(0, lambda: self.add_to_serial_display("[系统] 自动清空缓冲区", "info"))
                        
                        self.root.after(0, self.update_buffer_status)
                
                # 减少CPU占用，但保持响应性
                time.sleep(0.001)
                
            except Exception as e:
                print(f"串口接收错误: {e}")
                time.sleep(0.1)
    
    def update_serial_display(self, data):
        """更新串口数据显示"""
        try:
            display_format = self.display_format_var.get()
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3] if self.timestamp_var.get() else ""
            
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
            
            if timestamp:
                display_line = f"[{timestamp}] {display_data}"
            else:
                display_line = display_data
            
            if not display_line.endswith('\n'):
                display_line += '\n'
            
            self.serial_text.insert(tk.END, display_line)
            
            if self.auto_scroll_var.get():
                self.serial_text.see(tk.END)
            
            self.data_count_label.config(text=f"接收: {self.receive_byte_count} 字节")
            
            if int(self.serial_text.index('end-1c').split('.')[0]) > 1000:
                self.serial_text.delete(1.0, "100.0")
            
        except Exception as e:
            print(f"串口显示更新错误: {e}")
    
    def clear_serial_data(self):
        """清空串口接收数据"""
        try:
            self.serial_text.delete(1.0, tk.END)
            self.serial_buffer = []
            self.receive_byte_count = 0
            
            if self.is_serial_open and self.serial_port:
                self.serial_port.reset_input_buffer()
                self.serial_port.reset_output_buffer()
            
            # 清除回显验证状态
            with self.echo_verification_lock:
                self.expected_echo_data = None
                self.echo_buffer.clear()
            
            self.data_count_label.config(text="接收: 0 字节")
            self.update_buffer_status()
            
            self.add_to_serial_display("[系统] 接收区和缓冲区已清空", "info")
            print("串口接收区和缓冲区已清空")
            
        except Exception as e:
            print(f"清空串口数据失败: {e}")
            messagebox.showerror("错误", f"清空串口数据失败: {e}")
    
    def on_closing(self):
        """程序关闭时的清理工作"""
        if SERIAL_AVAILABLE:
            self.close_serial_port()
        self.root.destroy()


def main():
    # 检查串口库
    if not SERIAL_AVAILABLE:
        print("错误：pyserial库未安装")
        print("请使用以下命令安装：pip install pyserial")
        return
    
    root = tk.Tk()
    app = SerialController(root)
    
    # 设置关闭事件处理
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # 启动GUI
    root.mainloop()


if __name__ == "__main__":
    main()