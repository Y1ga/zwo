#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM32 PWM控制工具
功能：控制一个STM32的PWM参数（TIM1/TIM2/TIM3/TIM4）
支持两种模式：
1. 强度模式：只改变强度，保持原有频率
2. 完整模式：可以改变频率和强度
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


class PWMController:
    def __init__(self, root):
        self.root = root
        self.root.title("STM32 PWM控制工具")
        self.root.geometry("1400x900")
        
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
        self.receive_byte_count = 0
        
        # 发送统计
        self.send_stats = {'total_sent': 0, 'send_errors': 0}
        
        # PWM变量存储
        self.pwm_vars = {}
        
        # 控制模式：intensity(强度模式) 或 full(完整模式)
        self.control_mode = tk.StringVar(value="intensity")
        
        # 创建GUI界面
        self.create_widgets()
        
        # 初始化串口端口列表
        if SERIAL_AVAILABLE:
            self.refresh_serial_ports()
    
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 左侧控制面板
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 右侧数据接收区域
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === 模式选择 ===
        self.create_mode_selection(left_frame)
        
        # === 串口控制面板 ===
        if SERIAL_AVAILABLE:
            self.create_serial_controls(left_frame)
        
        # === PWM控制 ===
        if SERIAL_AVAILABLE:
            self.create_pwm_controls(left_frame)
        
        # === 快速控制 ===
        if SERIAL_AVAILABLE:
            self.create_quick_controls(left_frame)
        
        # 状态信息
        self.create_status_section(left_frame)
        
        # === 串口数据接收区域 ===
        if SERIAL_AVAILABLE:
            self.create_data_display(right_frame)
        
        # 配置网格权重
        self.configure_grid_weights()
        
        # 初始化界面状态（在所有控件创建完成后）
        if SERIAL_AVAILABLE:
            self.update_interface_for_mode()
    
    def create_mode_selection(self, parent):
        """创建模式选择"""
        mode_frame = ttk.LabelFrame(parent, text="控制模式", padding="10")
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Radiobutton(mode_frame, text="强度模式 (只改变强度，保持原频率)", 
                       variable=self.control_mode, value="intensity", 
                       command=self.on_mode_changed).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(mode_frame, text="完整模式 (可改变频率和强度)", 
                       variable=self.control_mode, value="full", 
                       command=self.on_mode_changed).pack(anchor=tk.W, pady=2)
        
        # 模式说明
        info_frame = ttk.Frame(mode_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.mode_info_label = ttk.Label(info_frame, text="当前：强度模式 - 只发送强度值，STM32保持原有频率设置", 
                                        font=("Arial", 9), foreground="blue")
        self.mode_info_label.pack(anchor=tk.W)
    
    def create_serial_controls(self, parent):
        """创建串口控制面板"""
        serial_frame = ttk.LabelFrame(parent, text="STM32 串口控制", padding="10")
        serial_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 端口选择
        ttk.Label(serial_frame, text="端口:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(serial_frame, textvariable=self.port_var, width=12)
        self.port_combo.grid(row=0, column=1, padx=(0, 10))
        
        # 波特率
        ttk.Label(serial_frame, text="波特率:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.baudrate_var = tk.StringVar(value="115200")
        baudrate_combo = ttk.Combobox(serial_frame, textvariable=self.baudrate_var, width=8,
                                    values=["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"])
        baudrate_combo.grid(row=0, column=3, padx=(0, 10))
        baudrate_combo.state(['readonly'])
        
        # 控制按钮
        button_frame = ttk.Frame(serial_frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=(10, 0))
        
        self.serial_open_button = ttk.Button(button_frame, text="打开串口", command=self.open_serial_port)
        self.serial_open_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.serial_close_button = ttk.Button(button_frame, text="关闭串口", command=self.close_serial_port, state=tk.DISABLED)
        self.serial_close_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="刷新端口", command=self.refresh_serial_ports).pack(side=tk.LEFT)
        
        # 状态显示
        self.serial_status_label = ttk.Label(serial_frame, text="串口状态: 未连接", foreground="red")
        self.serial_status_label.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(10, 0))
    
    def create_pwm_controls(self, parent):
        """创建PWM控制界面"""
        pwm_frame = ttk.LabelFrame(parent, text="PWM控制 (TIM1/TIM2/TIM3/TIM4)", padding="15")
        pwm_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 创建PWM控制
        self.create_timer_controls(pwm_frame)
        
        # 发送控制
        send_frame = ttk.Frame(pwm_frame)
        send_frame.grid(row=20, column=0, columnspan=8, sticky=(tk.W, tk.E), pady=(15, 0))
        
        ttk.Button(send_frame, text="发送配置", command=self.send_pwm_config, 
                  style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 15))
        ttk.Button(send_frame, text="停止所有PWM", command=self.stop_all_pwm).pack(side=tk.LEFT, padx=(0, 15))
        
        self.pwm_send_status_var = tk.StringVar(value="等待发送...")
        ttk.Label(send_frame, textvariable=self.pwm_send_status_var, 
                 font=("Arial", 9), foreground="blue").pack(side=tk.LEFT, padx=(15, 0))
    
    def create_timer_controls(self, parent_frame):
        """创建定时器控制界面"""
        pwm_vars = {}
        
        timers = [("TIM1", 1), ("TIM2", 2), ("TIM3", 3), ("TIM4", 4)]
        
        # 标题行
        ttk.Label(parent_frame, text="定时器", font=("Arial", 11, "bold")).grid(row=0, column=0, padx=(0, 15), pady=(0, 10))
        
        # 根据模式显示不同的标题
        self.freq_label = ttk.Label(parent_frame, text="频率(Hz)", font=("Arial", 11, "bold"))
        self.freq_label.grid(row=0, column=1, padx=(0, 15), pady=(0, 10))
        
        ttk.Label(parent_frame, text="强度(0-100)", font=("Arial", 11, "bold")).grid(row=0, column=2, padx=(0, 15), pady=(0, 10))
        ttk.Label(parent_frame, text="启用", font=("Arial", 11, "bold")).grid(row=0, column=3, padx=(0, 15), pady=(0, 10))
        ttk.Label(parent_frame, text="状态", font=("Arial", 11, "bold")).grid(row=0, column=4, padx=(0, 15), pady=(0, 10))
        ttk.Label(parent_frame, text="强度滑块", font=("Arial", 11, "bold")).grid(row=0, column=5, pady=(0, 10))
        
        for i, (label, tim_num) in enumerate(timers):
            row = i + 1
            
            # 定时器标签
            ttk.Label(parent_frame, text=label, font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, padx=(0, 15), pady=8)
            
            # 频率输入框（完整模式才显示）
            freq_var = tk.StringVar(value="1000")
            pwm_vars[f"tim{tim_num}_freq"] = freq_var
            freq_entry = ttk.Entry(parent_frame, textvariable=freq_var, width=10, font=("Arial", 10))
            freq_entry.grid(row=row, column=1, padx=(0, 15), pady=8)
            setattr(self, f'freq_entry_{tim_num}', freq_entry)
            
            # 强度输入框
            intensity_var = tk.StringVar(value="0")
            pwm_vars[f"tim{tim_num}_intensity"] = intensity_var
            intensity_entry = ttk.Entry(parent_frame, textvariable=intensity_var, width=10, font=("Arial", 10))
            intensity_entry.grid(row=row, column=2, padx=(0, 15), pady=8)
            
            # 启用/禁用复选框
            enable_var = tk.BooleanVar(value=False)
            pwm_vars[f"tim{tim_num}_enable"] = enable_var
            enable_check = ttk.Checkbutton(parent_frame, text="启用", variable=enable_var)
            enable_check.grid(row=row, column=3, padx=(0, 15), pady=8)
            
            # 状态显示
            status_var = tk.StringVar(value="禁用")
            pwm_vars[f"tim{tim_num}_status"] = status_var
            status_label = ttk.Label(parent_frame, textvariable=status_var, font=("Arial", 9), foreground="gray")
            status_label.grid(row=row, column=4, padx=(0, 15), pady=8, sticky=tk.W)
            
            # 滑块控制
            intensity_scale = tk.Scale(parent_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                     variable=intensity_var, length=180, resolution=1)
            intensity_scale.grid(row=row, column=5, padx=(0, 10), pady=8)
            
            # 绑定事件
            enable_var.trace('w', lambda *args, tim=tim_num: self.on_pwm_enable_changed(tim))
            intensity_var.trace('w', lambda *args, tim=tim_num: self.on_intensity_changed(tim))
            freq_var.trace('w', lambda *args, tim=tim_num: self.on_frequency_changed(tim))
        
        # 保存PWM变量
        self.pwm_vars = pwm_vars
    
    def create_quick_controls(self, parent):
        """创建快速控制"""
        quick_frame = ttk.LabelFrame(parent, text="快速控制", padding="10")
        quick_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 预设控制
        preset_frame = ttk.Frame(quick_frame)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(preset_frame, text="预设强度:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(preset_frame, text="全部0%", command=lambda: self.set_all_intensity(0)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_frame, text="全部25%", command=lambda: self.set_all_intensity(25)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_frame, text="全部50%", command=lambda: self.set_all_intensity(50)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_frame, text="全部75%", command=lambda: self.set_all_intensity(75)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_frame, text="全部100%", command=lambda: self.set_all_intensity(100)).pack(side=tk.LEFT, padx=(0, 5))
        
        # 开关控制
        switch_frame = ttk.Frame(quick_frame)
        switch_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Label(switch_frame, text="开关控制:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(switch_frame, text="全部启用", command=self.enable_all_pwm).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(switch_frame, text="全部禁用", command=self.disable_all_pwm).pack(side=tk.LEFT, padx=(0, 5))
        
        # 频率预设（完整模式）
        self.freq_preset_frame = ttk.Frame(quick_frame)
        self.freq_preset_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Label(self.freq_preset_frame, text="频率预设:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(self.freq_preset_frame, text="1KHz", command=lambda: self.set_all_frequency(1000)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.freq_preset_frame, text="10KHz", command=lambda: self.set_all_frequency(10000)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.freq_preset_frame, text="100KHz", command=lambda: self.set_all_frequency(100000)).pack(side=tk.LEFT, padx=(0, 5))
        
        # 自定义控制
        custom_frame = ttk.Frame(quick_frame)
        custom_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(custom_frame, text="自定义:").pack(side=tk.LEFT, padx=(0, 5))
        
        # 自定义频率（完整模式）
        self.custom_freq_frame = ttk.Frame(custom_frame)
        self.custom_freq_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(self.custom_freq_frame, text="频率:").pack(side=tk.LEFT, padx=(0, 2))
        self.custom_freq_var = tk.StringVar(value="5000")
        freq_entry = ttk.Entry(self.custom_freq_frame, textvariable=self.custom_freq_var, width=8)
        freq_entry.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(self.custom_freq_frame, text="Hz").pack(side=tk.LEFT, padx=(0, 5))
        
        # 自定义强度
        ttk.Label(custom_frame, text="强度:").pack(side=tk.LEFT, padx=(0, 2))
        self.custom_intensity_var = tk.StringVar(value="30")
        intensity_entry = ttk.Entry(custom_frame, textvariable=self.custom_intensity_var, width=6)
        intensity_entry.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(custom_frame, text="%").pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(custom_frame, text="应用", command=self.apply_custom_settings).pack(side=tk.LEFT, padx=(5, 0))
    
    def create_status_section(self, parent):
        """创建状态信息区域"""
        status_frame = ttk.LabelFrame(parent, text="状态信息", padding="10")
        status_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="PWM控制工具就绪")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # 发送统计
        self.send_stats_label = ttk.Label(status_frame, text="发送: 0次 | 错误: 0次")
        self.send_stats_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
    
    def create_data_display(self, parent):
        """创建数据接收显示区域"""
        display_frame = ttk.LabelFrame(parent, text="串口数据接收", padding="10")
        display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.serial_text = scrolledtext.ScrolledText(display_frame, height=35, width=70, font=("Consolas", 9))
        self.serial_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        control_frame = ttk.Frame(display_frame)
        control_frame.pack(fill=tk.X)
        
        self.data_count_label = ttk.Label(control_frame, text="接收: 0 字节")
        self.data_count_label.pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="清空", command=self.clear_serial_data).pack(side=tk.RIGHT)
    
    def configure_grid_weights(self):
        """配置网格权重"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 获取主框架
        main_frame = self.root.grid_slaves()[0]
        main_frame.columnconfigure(0, weight=0)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # 获取左右框架
        frames = main_frame.grid_slaves()
        for frame in frames:
            if frame.grid_info()['column'] == 0:  # 左侧面板
                frame.columnconfigure(0, weight=1)
            elif frame.grid_info()['column'] == 1:  # 右侧面板
                frame.columnconfigure(0, weight=1)
                frame.rowconfigure(0, weight=1)
    
    # === 模式控制方法 ===
    def on_mode_changed(self):
        """模式改变时的处理"""
        self.update_interface_for_mode()
        
        if self.control_mode.get() == "intensity":
            self.mode_info_label.config(text="当前：强度模式 - 只发送强度值，STM32保持原有频率设置")
        else:
            self.mode_info_label.config(text="当前：完整模式 - 发送频率和强度值，完全控制PWM参数")
    
    def update_interface_for_mode(self):
        """根据模式更新界面"""
        try:
            if self.control_mode.get() == "intensity":
                # 强度模式：隐藏频率相关控件
                if hasattr(self, 'freq_label'):
                    self.freq_label.config(text="频率(Hz) [只读]")
                
                for tim_num in [1, 2, 3, 4]:
                    if hasattr(self, f'freq_entry_{tim_num}'):
                        freq_entry = getattr(self, f'freq_entry_{tim_num}')
                        freq_entry.config(state='readonly')
                
                # 隐藏频率预设
                if hasattr(self, 'freq_preset_frame'):
                    self.freq_preset_frame.pack_forget()
                if hasattr(self, 'custom_freq_frame'):
                    self.custom_freq_frame.pack_forget()
            else:
                # 完整模式：显示所有控件
                if hasattr(self, 'freq_label'):
                    self.freq_label.config(text="频率(Hz)")
                
                for tim_num in [1, 2, 3, 4]:
                    if hasattr(self, f'freq_entry_{tim_num}'):
                        freq_entry = getattr(self, f'freq_entry_{tim_num}')
                        freq_entry.config(state='normal')
                
                # 显示频率预设
                if hasattr(self, 'freq_preset_frame'):
                    self.freq_preset_frame.pack(fill=tk.X, pady=(5, 10))
                if hasattr(self, 'custom_freq_frame'):
                    self.custom_freq_frame.pack(side=tk.LEFT, padx=(0, 10))
        except Exception as e:
            print(f"界面更新错误: {e}")
    
    # === PWM控制方法 ===
    def on_pwm_enable_changed(self, tim_num):
        """PWM启用状态改变时的处理"""
        enable_var = self.pwm_vars[f"tim{tim_num}_enable"]
        status_var = self.pwm_vars[f"tim{tim_num}_status"]
        
        if enable_var.get():
            self.update_timer_status(tim_num)
        else:
            status_var.set("禁用")
    
    def on_intensity_changed(self, tim_num):
        """强度值改变时的处理"""
        enable_var = self.pwm_vars[f"tim{tim_num}_enable"]
        if enable_var.get():
            self.update_timer_status(tim_num)
    
    def on_frequency_changed(self, tim_num):
        """频率值改变时的处理"""
        enable_var = self.pwm_vars[f"tim{tim_num}_enable"]
        if enable_var.get() and self.control_mode.get() == "full":
            self.update_timer_status(tim_num)
    
    def update_timer_status(self, tim_num):
        """更新定时器状态显示"""
        status_var = self.pwm_vars[f"tim{tim_num}_status"]
        intensity = self.pwm_vars[f"tim{tim_num}_intensity"].get()
        
        try:
            intensity_val = float(intensity)
            if 0 <= intensity_val <= 100:
                if self.control_mode.get() == "full":
                    freq = self.pwm_vars[f"tim{tim_num}_freq"].get()
                    try:
                        freq_val = float(freq)
                        status_var.set(f"启用 - {freq}Hz, {intensity}%")
                    except:
                        status_var.set(f"启用 - 无效频率, {intensity}%")
                else:
                    status_var.set(f"启用 - {intensity}%")
            else:
                status_var.set("启用 - 无效强度值")
        except:
            status_var.set("启用 - 无效强度值")
    
    def set_all_intensity(self, intensity):
        """设置所有PWM强度"""
        for tim_num in [1, 2, 3, 4]:
            self.pwm_vars[f"tim{tim_num}_intensity"].set(str(intensity))
            self.pwm_vars[f"tim{tim_num}_enable"].set(True)
        print(f"已设置所有PWM强度为: {intensity}%")
    
    def set_all_frequency(self, frequency):
        """设置所有PWM频率（仅完整模式）"""
        if self.control_mode.get() == "full":
            for tim_num in [1, 2, 3, 4]:
                self.pwm_vars[f"tim{tim_num}_freq"].set(str(frequency))
            print(f"已设置所有PWM频率为: {frequency}Hz")
    
    def enable_all_pwm(self):
        """启用所有PWM"""
        for tim_num in [1, 2, 3, 4]:
            self.pwm_vars[f"tim{tim_num}_enable"].set(True)
        print("已启用所有PWM")
    
    def disable_all_pwm(self):
        """禁用所有PWM"""
        for tim_num in [1, 2, 3, 4]:
            self.pwm_vars[f"tim{tim_num}_enable"].set(False)
        print("已禁用所有PWM")
    
    def apply_custom_settings(self):
        """应用自定义设置"""
        try:
            intensity = float(self.custom_intensity_var.get())
            if intensity < 0 or intensity > 100:
                messagebox.showwarning("警告", "强度值必须在0-100之间")
                return
            
            if self.control_mode.get() == "full":
                frequency = float(self.custom_freq_var.get())
                if frequency <= 0 or frequency > 1000000:
                    messagebox.showwarning("警告", "频率必须在0-1000000Hz之间")
                    return
                self.set_all_frequency(int(frequency))
            
            self.set_all_intensity(int(intensity))
            
        except ValueError:
            messagebox.showerror("错误", "参数必须是数字")
    
    def validate_values(self):
        """验证参数的有效性"""
        try:
            for tim_num in [1, 2, 3, 4]:
                # 验证强度
                intensity = float(self.pwm_vars[f"tim{tim_num}_intensity"].get())
                if intensity < 0 or intensity > 100:
                    self.pwm_vars[f"tim{tim_num}_intensity"].set("0")
                    messagebox.showwarning("警告", f"TIM{tim_num}强度值必须在0-100之间")
                    return False
                
                # 验证频率（完整模式）
                if self.control_mode.get() == "full":
                    freq = float(self.pwm_vars[f"tim{tim_num}_freq"].get())
                    if freq <= 0 or freq > 1000000:
                        self.pwm_vars[f"tim{tim_num}_freq"].set("1000")
                        messagebox.showwarning("警告", f"TIM{tim_num}频率必须在0-1000000Hz之间")
                        return False
            return True
        except ValueError:
            messagebox.showerror("错误", "参数必须是数字")
            return False
    
    # === 串口通信方法 ===
    def refresh_serial_ports(self):
        """刷新串口端口列表"""
        if not SERIAL_AVAILABLE:
            return
        
        try:
            ports = serial.tools.list_ports.comports()
            port_names = [port.device for port in ports]
            
            self.port_combo['values'] = port_names
            
            if port_names and not self.port_var.get():
                self.port_var.set(port_names[0])
            
            print(f"发现串口: {port_names}")
                
        except Exception as e:
            print(f"刷新串口列表失败: {e}")
    
    def open_serial_port(self):
        """打开串口"""
        if not SERIAL_AVAILABLE:
            messagebox.showerror("错误", "pyserial库未安装，无法使用串口功能")
            return
        
        try:
            port = self.port_var.get()
            baudrate = int(self.baudrate_var.get())
            
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            
            self.serial_port = serial.Serial(
                port=port, baudrate=baudrate, bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                timeout=0.001, write_timeout=0.05, xonxoff=False, rtscts=False, dsrdtr=False
            )
            
            self.is_serial_open = True
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            self.serial_thread = threading.Thread(target=self.serial_receive_loop, daemon=True)
            self.serial_thread.start()
            
            # 更新界面状态
            self.serial_open_button.config(state=tk.DISABLED)
            self.serial_close_button.config(state=tk.NORMAL)
            self.serial_status_label.config(text=f"串口状态: 已连接 ({port}, {baudrate})", foreground="green")
            
            self.receive_byte_count = 0
            self.data_count_label.config(text="接收: 0 字节")
            
            print(f"串口已打开: {port}, 波特率: {baudrate}")
            
        except Exception as e:
            messagebox.showerror("错误", f"打开串口失败: {str(e)}")
    
    def close_serial_port(self):
        """关闭串口"""
        try:
            self.is_serial_open = False
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            
            # 更新界面状态
            self.serial_open_button.config(state=tk.NORMAL)
            self.serial_close_button.config(state=tk.DISABLED)
            self.serial_status_label.config(text="串口状态: 未连接", foreground="red")
            
            print("串口已关闭")
            
        except Exception as e:
            messagebox.showerror("错误", f"关闭串口失败: {str(e)}")
    
    def serial_receive_loop(self):
        """串口数据接收循环"""
        while self.is_serial_open and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    if data:
                        self.receive_byte_count += len(data)
                        self.root.after(0, lambda d=data: self.update_serial_display(d))
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"串口接收错误: {e}")
                time.sleep(0.1)
    
    def update_serial_display(self, data):
        """更新串口数据显示"""
        try:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            try:
                text_data = data.decode('utf-8', errors='replace')
            except:
                text_data = data.decode('latin-1', errors='replace')
            
            display_line = f"[{timestamp}] {text_data}"
            if not display_line.endswith('\n'):
                display_line += '\n'
            
            self.serial_text.insert(tk.END, display_line)
            self.serial_text.see(tk.END)
            self.data_count_label.config(text=f"接收: {self.receive_byte_count} 字节")
            
        except Exception as e:
            print(f"串口显示更新错误: {e}")
    
    def clear_serial_data(self):
        """清空串口接收数据"""
        try:
            self.serial_text.delete(1.0, tk.END)
            self.receive_byte_count = 0
            self.data_count_label.config(text="接收: 0 字节")
            
            if self.is_serial_open and self.serial_port:
                self.serial_port.reset_input_buffer()
                self.serial_port.reset_output_buffer()
            
            print("串口接收区已清空")
            
        except Exception as e:
            print(f"清空串口数据失败: {e}")
    
    # === PWM配置发送方法 ===
    def send_pwm_config(self):
        """发送PWM配置到STM32"""
        if not self.is_serial_open or not self.serial_port:
            messagebox.showerror("错误", "串口未打开")
            return
        
        if not self.validate_values():
            return
        
        try:
            if self.control_mode.get() == "intensity":
                self.send_intensity_only()
            else:
                self.send_full_config()
                
        except Exception as e:
            self.pwm_send_status_var.set("发送失败!")
            self.send_stats['send_errors'] += 1
            self.send_stats_label.config(text=f"发送: {self.send_stats['total_sent']}次 | 错误: {self.send_stats['send_errors']}次")
            messagebox.showerror("错误", f"发送PWM配置失败: {e}")
    
    def send_intensity_only(self):
        """发送强度模式数据包"""
        # 格式: 0xB0 + TIM1强度 + TIM2强度 + TIM3强度 + TIM4强度 + 校验和
        data_packet = bytearray()
        data_packet.append(0xB0)  # 强度模式命令头
        
        intensities = []
        enabled_timers = []
        
        for tim_num in [1, 2, 3, 4]:
            enable = self.pwm_vars[f"tim{tim_num}_enable"].get()
            intensity = int(float(self.pwm_vars[f"tim{tim_num}_intensity"].get())) if enable else 0
            
            data_packet.append(intensity)
            intensities.append(intensity)
            
            if enable and intensity > 0:
                enabled_timers.append(f"TIM{tim_num}({intensity}%)")
        
        # 计算校验和
        checksum = sum(data_packet) & 0xFF
        data_packet.append(checksum)
        
        # 发送数据
        self.serial_port.write(data_packet)
        self.serial_port.flush()
        
        # 显示发送的数据
        self.display_sent_data("强度模式", data_packet, enabled_timers, intensities)
        
        # 更新状态
        if enabled_timers:
            status_text = f"已发送强度配置: {', '.join(enabled_timers)}"
        else:
            status_text = "已发送强度配置: 所有PWM禁用"
        
        self.pwm_send_status_var.set(status_text)
        self.update_send_stats()
        
        print(f"强度模式配置发送成功: {len(data_packet)} 字节 - {intensities}")
    
    def send_full_config(self):
        """发送完整模式数据包"""
        # 格式: 0xA0 + (启用标志+频率高8位+频率低8位+强度) * 4 + 校验和
        data_packet = bytearray()
        data_packet.append(0xA0)  # 完整模式命令头
        
        enabled_timers = []
        
        for tim_num in [1, 2, 3, 4]:
            enable = 1 if self.pwm_vars[f"tim{tim_num}_enable"].get() else 0
            frequency = int(float(self.pwm_vars[f"tim{tim_num}_freq"].get())) if enable else 0
            intensity = int(float(self.pwm_vars[f"tim{tim_num}_intensity"].get())) if enable else 0
            
            # 限制频率为16位 (0-65535)
            if frequency > 65535:
                frequency = 65535
            
            data_packet.append(enable)
            data_packet.append((frequency >> 8) & 0xFF)  # 频率高8位
            data_packet.append(frequency & 0xFF)         # 频率低8位
            data_packet.append(intensity)
            
            if enable:
                enabled_timers.append(f"TIM{tim_num}({frequency}Hz,{intensity}%)")
        
        # 计算校验和
        checksum = sum(data_packet) & 0xFF
        data_packet.append(checksum)
        
        # 发送数据
        self.serial_port.write(data_packet)
        self.serial_port.flush()
        
        # 显示发送的数据
        self.display_sent_data("完整模式", data_packet, enabled_timers)
        
        # 更新状态
        if enabled_timers:
            status_text = f"已发送完整配置: {', '.join(enabled_timers)}"
        else:
            status_text = "已发送完整配置: 所有PWM禁用"
        
        self.pwm_send_status_var.set(status_text)
        self.update_send_stats()
        
        print(f"完整模式配置发送成功: {len(data_packet)} 字节")
    
    def stop_all_pwm(self):
        """停止所有PWM"""
        if not self.is_serial_open or not self.serial_port:
            messagebox.showerror("错误", "串口未打开")
            return
        
        try:
            # 发送停止命令: 0xC0 0xC0
            stop_cmd = bytearray([0xC0, 0xC0])
            self.serial_port.write(stop_cmd)
            self.serial_port.flush()
            
            # 更新界面状态
            for tim_num in [1, 2, 3, 4]:
                self.pwm_vars[f"tim{tim_num}_enable"].set(False)
                self.pwm_vars[f"tim{tim_num}_intensity"].set("0")
            
            self.pwm_send_status_var.set("已停止所有PWM")
            
            # 显示发送的命令
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.serial_text.insert(tk.END, f"[{timestamp}] → [发送] 停止所有PWM命令: C0 C0\n")
            self.serial_text.see(tk.END)
            
            print("已发送停止所有PWM命令")
            
        except Exception as e:
            messagebox.showerror("错误", f"停止PWM失败: {e}")
    
    def display_sent_data(self, mode, data, enabled_timers, intensities=None):
        """显示发送的配置数据"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # 显示模式和启用的定时器
        self.serial_text.insert(tk.END, f"[{timestamp}] → [发送] {mode} PWM配置\n")
        
        if enabled_timers:
            timer_info = ', '.join(enabled_timers)
            self.serial_text.insert(tk.END, f"[{timestamp}] → [发送] 启用: {timer_info}\n")
        else:
            self.serial_text.insert(tk.END, f"[{timestamp}] → [发送] 状态: 所有PWM已禁用\n")
        
        # 显示强度详情（强度模式）
        if intensities:
            intensity_info = f"TIM1:{intensities[0]}% TIM2:{intensities[1]}% TIM3:{intensities[2]}% TIM4:{intensities[3]}%"
            self.serial_text.insert(tk.END, f"[{timestamp}] → [发送] 强度详情: {intensity_info}\n")
        
        self.serial_text.insert(tk.END, f"[{timestamp}] → [发送] 十六进制: {data.hex(' ').upper()}\n")
        self.serial_text.see(tk.END)
    
    def update_send_stats(self):
        """更新发送统计"""
        self.send_stats['total_sent'] += 1
        self.send_stats_label.config(text=f"发送: {self.send_stats['total_sent']}次 | 错误: {self.send_stats['send_errors']}次")
    
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
    app = PWMController(root)
    
    # 设置关闭事件处理
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # 启动GUI
    root.mainloop()


if __name__ == "__main__":
    main()