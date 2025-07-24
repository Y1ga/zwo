#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单串口定时器控制工具
功能：控制STM32的定时器频率（TIM1/TIM2/TIM3/TIM4）
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


class SingleSerialTimerController:
    def __init__(self, root):
        self.root = root
        self.root.title("单串口定时器控制工具")
        self.root.geometry("1200x800")
        
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
        
        # 预设频率组合
        self.frequency_presets = {
            1: {"TIM1": 113, "TIM2": 127, "TIM3": 131, "TIM4": 137},
            2: {"TIM1": 139, "TIM2": 149, "TIM3": 151, "TIM4": 157},
            3: {"TIM1": 163, "TIM2": 167, "TIM3": 173, "TIM4": 179}
        }
        
        # 串口相关变量
        self.serial_port = None
        self.is_serial_open = False
        self.serial_thread = None
        self.receive_byte_count = 0
        
        # 发送统计
        self.send_stats = {'total_sent': 0, 'send_errors': 0}
        
        # 定时器变量存储
        self.timer_vars = {}
        
        # 当前选择的频率组合
        self.current_frequency_set = tk.IntVar(value=1)
        
        # Next按钮功能：当前激活的定时器索引 (0=TIM1, 1=TIM2, 2=TIM3, 3=TIM4, -1=全部关闭)
        self.current_active_timer = -1  # 初始状态：全部关闭
        
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
        
        # === 串口控制面板 ===
        if SERIAL_AVAILABLE:
            self.create_serial_controls(left_frame)
        
        # === 定时器控制 ===
        if SERIAL_AVAILABLE:
            self.create_timer_controls(left_frame)
        
        # === 频率组合选择 ===
        if SERIAL_AVAILABLE:
            self.create_frequency_selection(left_frame)
        
        # === 全局控制 ===
        if SERIAL_AVAILABLE:
            self.create_global_controls(left_frame)
        
        # 状态信息
        self.create_status_section(left_frame)
        
        # === 串口数据接收区域 ===
        if SERIAL_AVAILABLE:
            self.create_data_display(right_frame)
        
        # 配置网格权重
        self.configure_grid_weights()
    
    def create_serial_controls(self, parent):
        """创建串口控制面板"""
        serial_frame = ttk.LabelFrame(parent, text="串口控制", padding="10")
        serial_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 端口选择
        ttk.Label(serial_frame, text="端口:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(serial_frame, textvariable=self.port_var, width=10)
        self.port_combo.grid(row=0, column=1, padx=(0, 10))
        
        # 波特率
        ttk.Label(serial_frame, text="波特率:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.baudrate_var = tk.StringVar(value="460800")
        baudrate_combo = ttk.Combobox(serial_frame, textvariable=self.baudrate_var, width=8,
                                    values=["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"])
        baudrate_combo.grid(row=0, column=3, padx=(0, 10))
        baudrate_combo.state(['readonly'])
        
        # 控制按钮
        self.serial_open_button = ttk.Button(serial_frame, text="打开串口", command=self.open_serial_port)
        self.serial_open_button.grid(row=1, column=0, pady=(10, 0), padx=(0, 10))
        
        self.serial_close_button = ttk.Button(serial_frame, text="关闭串口", command=self.close_serial_port, state=tk.DISABLED)
        self.serial_close_button.grid(row=1, column=1, pady=(10, 0), padx=(0, 10))
        
        ttk.Button(serial_frame, text="刷新端口", command=self.refresh_serial_ports).grid(row=1, column=2, pady=(10, 0), padx=(0, 10))
        
        # 状态显示
        self.serial_status_label = ttk.Label(serial_frame, text="串口状态: 未连接", foreground="red")
        self.serial_status_label.grid(row=1, column=3, sticky=tk.W, pady=(10, 0), padx=(10, 0))
    
    def create_frequency_selection(self, parent):
        """创建频率组合选择面板"""
        freq_frame = ttk.LabelFrame(parent, text="频率组合选择", padding="10")
        freq_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 显示当前选择的组合信息
        info_frame = ttk.Frame(freq_frame)
        info_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.freq_info_label = ttk.Label(info_frame, text="", font=("Arial", 10), foreground="blue")
        self.freq_info_label.pack()
        
        # 三个预设选择按钮
        for i in range(1, 4):
            preset = self.frequency_presets[i]
            freq_text = f"组合{i}: TIM1={preset['TIM1']}Hz, TIM2={preset['TIM2']}Hz, TIM3={preset['TIM3']}Hz, TIM4={preset['TIM4']}Hz"
            
            radio_btn = ttk.Radiobutton(
                freq_frame, 
                text=freq_text, 
                variable=self.current_frequency_set, 
                value=i,
                command=self.on_frequency_set_changed
            )
            radio_btn.grid(row=i, column=0, sticky=tk.W, pady=2)
        
        # 初始化显示
        self.on_frequency_set_changed()
    
    def create_timer_controls(self, parent):
        """创建定时器控制面板"""
        timer_frame = ttk.LabelFrame(parent, text="定时器控制 (TIM1/TIM2/TIM3/TIM4)", padding="10")
        timer_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        timer_vars = {}
        
        timers = [("TIM1", 1), ("TIM2", 2), ("TIM3", 3), ("TIM4", 4)]
        
        # 表头
        ttk.Label(timer_frame, text="定时器", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=10, pady=5)
        ttk.Label(timer_frame, text="频率(Hz)", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(timer_frame, text="占空比值", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=10, pady=5)
        ttk.Label(timer_frame, text="启用", font=("Arial", 10, "bold")).grid(row=0, column=3, padx=10, pady=5)
        ttk.Label(timer_frame, text="状态", font=("Arial", 10, "bold")).grid(row=0, column=4, padx=10, pady=5)
        
        for i, (label, tim_num) in enumerate(timers):
            row = i + 1
            
            # 定时器标签
            ttk.Label(timer_frame, text=label, font=("Arial", 11, "bold")).grid(row=row, column=0, padx=10, pady=8)
            
            # 频率显示（只读）
            freq_var = tk.StringVar(value="113" if tim_num == 1 else ("127" if tim_num == 2 else ("131" if tim_num == 3 else "137")))
            timer_vars[f"tim{tim_num}_freq"] = freq_var
            freq_label = ttk.Label(timer_frame, textvariable=freq_var, font=("Arial", 10), foreground="darkblue")
            freq_label.grid(row=row, column=1, padx=10, pady=8)
            
            # 占空比输入
            duty_var = tk.StringVar(value="1")
            timer_vars[f"tim{tim_num}_duty"] = duty_var
            duty_entry = ttk.Entry(timer_frame, textvariable=duty_var, width=8)
            duty_entry.grid(row=row, column=2, padx=10, pady=8)
            
            # 启用/禁用复选框
            enable_var = tk.BooleanVar(value=False)
            timer_vars[f"tim{tim_num}_enable"] = enable_var
            enable_check = ttk.Checkbutton(timer_frame, variable=enable_var)
            enable_check.grid(row=row, column=3, padx=10, pady=8)
            
            # 状态显示
            status_var = tk.StringVar(value="禁用")
            timer_vars[f"tim{tim_num}_status"] = status_var
            status_label = ttk.Label(timer_frame, textvariable=status_var, font=("Arial", 9), foreground="gray")
            status_label.grid(row=row, column=4, padx=10, pady=8)
            
            # 绑定启用状态变化
            enable_var.trace('w', lambda *args, tim=tim_num: self.on_timer_enable_changed(tim))
        
        # 快速设置按钮
        quick_frame = ttk.Frame(timer_frame)
        quick_frame.grid(row=6, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(15, 0))
        
        ttk.Label(quick_frame, text="快速设置:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="全部禁用", command=self.disable_all_timers).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="启用所有", command=self.enable_all_timers).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="占空比值10", command=lambda: self.set_all_duty(10)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="占空比值50", command=lambda: self.set_all_duty(50)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="占空比值100", command=lambda: self.set_all_duty(100)).pack(side=tk.LEFT, padx=(0, 5))
        
        # 自定义占空比
        custom_frame = ttk.Frame(timer_frame)
        custom_frame.grid(row=7, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(custom_frame, text="自定义占空比值:").pack(side=tk.LEFT, padx=(0, 5))
        self.custom_duty_var = tk.StringVar(value="30")
        duty_custom_entry = ttk.Entry(custom_frame, textvariable=self.custom_duty_var, width=6)
        duty_custom_entry.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(custom_frame, text="(0-255)").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(custom_frame, text="应用自定义", command=self.apply_custom_duty).pack(side=tk.LEFT, padx=(5, 0))
        
        # 保存定时器变量
        self.timer_vars = timer_vars
        
        # 初始化频率显示（如果频率选择面板已经创建）
        if hasattr(self, 'freq_info_label'):
            self.on_frequency_set_changed()
    
    def create_global_controls(self, parent):
        """创建全局控制"""
        global_frame = ttk.LabelFrame(parent, text="全局控制", padding="10")
        global_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 主控制按钮
        control_frame = ttk.Frame(global_frame)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(control_frame, text="发送配置", command=self.send_timer_config, width=12).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="停止所有定时器", command=self.stop_all_timers, width=15).pack(side=tk.LEFT, padx=(0, 10))
        
        # Next按钮控制
        next_frame = ttk.Frame(global_frame)
        next_frame.pack(fill=tk.X, pady=(10, 5))
        
        ttk.Label(next_frame, text="依次激活:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(next_frame, text="Next", command=self.next_timer, width=8).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(next_frame, text="全部关闭", command=self.turn_off_all, width=10).pack(side=tk.LEFT, padx=(0, 10))
        
        # 当前激活状态显示
        self.active_timer_status = tk.StringVar(value="当前激活: 无")
        ttk.Label(next_frame, textvariable=self.active_timer_status, font=("Arial", 9), foreground="green").pack(side=tk.LEFT, padx=(10, 0))
        
        # 状态显示
        status_frame = ttk.Frame(global_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.timer_send_status_var = tk.StringVar(value="等待发送...")
        ttk.Label(status_frame, textvariable=self.timer_send_status_var, font=("Arial", 9), foreground="blue").pack(side=tk.LEFT)
    
    def create_status_section(self, parent):
        """创建状态信息区域"""
        status_frame = ttk.LabelFrame(parent, text="状态信息", padding="10")
        status_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="单串口定时器控制工具就绪")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # 发送统计
        self.send_stats_label = ttk.Label(status_frame, text="发送: 0次 | 错误: 0次")
        self.send_stats_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
    
    def create_data_display(self, parent):
        """创建数据接收显示区域"""
        data_frame = ttk.LabelFrame(parent, text="串口数据接收", padding="10")
        data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.serial_text = scrolledtext.ScrolledText(data_frame, height=30, width=70, font=("Consolas", 9))
        self.serial_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        control_frame = ttk.Frame(data_frame)
        control_frame.pack(fill=tk.X)
        
        self.data_count_label = ttk.Label(control_frame, text="接收: 0 字节")
        self.data_count_label.pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="清空数据", command=self.clear_serial_data).pack(side=tk.RIGHT)
    
    def configure_grid_weights(self):
        """配置网格权重"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 获取主框架
        main_frame = self.root.grid_slaves()[0]
        main_frame.columnconfigure(0, weight=0)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    # === 频率组合相关方法 ===
    def on_frequency_set_changed(self):
        """频率组合改变时的处理"""
        # 检查定时器变量是否已初始化
        if not hasattr(self, 'timer_vars') or not self.timer_vars:
            return
            
        current_set = self.current_frequency_set.get()
        preset = self.frequency_presets[current_set]
        
        # 更新频率显示
        for tim_num in [1, 2, 3, 4]:
            freq_var = self.timer_vars[f"tim{tim_num}_freq"]
            freq_var.set(str(preset[f"TIM{tim_num}"]))
        
        # 更新信息显示
        freq_info = f"当前选择组合{current_set}: TIM1={preset['TIM1']}Hz, TIM2={preset['TIM2']}Hz, TIM3={preset['TIM3']}Hz, TIM4={preset['TIM4']}Hz"
        self.freq_info_label.config(text=freq_info)
        
        print(f"已切换到频率组合{current_set}")
    
    # === 定时器控制方法 ===
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
    
    def set_all_duty(self, duty_value):
        """设置所有定时器的占空比值"""
        for tim_num in [1, 2, 3, 4]:
            self.timer_vars[f"tim{tim_num}_duty"].set(str(duty_value))
        print(f"已设置所有定时器占空比值为: {duty_value}")
    
    def apply_custom_duty(self):
        """应用自定义占空比"""
        try:
            duty_value = float(self.custom_duty_var.get())
            
            if duty_value < 0 or duty_value > 255:
                messagebox.showwarning("警告", "占空比值必须在0-255之间")
                return
            
            self.set_all_duty(int(duty_value))
            
        except ValueError:
            messagebox.showerror("错误", "自定义占空比必须是数字")
    
    # === Next按钮控制方法 ===
    def next_timer(self):
        """按顺序激活下一个定时器 (TIM1→TIM2→TIM3→TIM4→TIM1...)"""
        # 先关闭当前激活的定时器
        if self.current_active_timer != -1:
            self.timer_vars[f"tim{self.current_active_timer + 1}_enable"].set(False)
        
        # 移动到下一个定时器
        self.current_active_timer = (self.current_active_timer + 1) % 4
        
        # 激活下一个定时器
        self.timer_vars[f"tim{self.current_active_timer + 1}_enable"].set(True)
        
        # 更新状态显示
        timer_name = f"TIM{self.current_active_timer + 1}"
        self.active_timer_status.set(f"当前激活: {timer_name}")
        
        # 自动发送配置
        self.send_timer_config()
        
        print(f"Next: 激活 {timer_name}")
    
    def turn_off_all(self):
        """关闭所有定时器"""
        # 禁用所有定时器
        for tim_num in [1, 2, 3, 4]:
            self.timer_vars[f"tim{tim_num}_enable"].set(False)
        
        # 重置当前激活状态
        self.current_active_timer = -1
        self.active_timer_status.set("当前激活: 无")
        
        # 自动发送配置
        self.send_timer_config()
        
        print("已关闭所有定时器")
    
    def validate_timer_values(self):
        """验证定时器参数的有效性"""
        try:
            for tim_num in [1, 2, 3, 4]:
                duty = float(self.timer_vars[f"tim{tim_num}_duty"].get())
                if duty < 0 or duty > 255:
                    self.timer_vars[f"tim{tim_num}_duty"].set("1")
                    messagebox.showwarning("警告", f"TIM{tim_num}占空比值必须在0-255之间")
                    return False
            return True
        except ValueError:
            messagebox.showerror("错误", "定时器参数必须是数字")
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
            
            # 自动选择第一个端口
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
    
    # === 定时器配置发送方法 ===
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
                
                data_packet.append(enable)
                data_packet.append((frequency >> 16) & 0xFF)
                data_packet.append((frequency >> 8) & 0xFF)
                data_packet.append(frequency & 0xFF)
                data_packet.append(duty)
            
            # 计算校验和
            checksum = sum(data_packet) & 0xFF
            data_packet.append(checksum)
            
            # 发送数据
            bytes_sent = self.serial_port.write(data_packet)
            self.serial_port.flush()
            
            # 显示发送的数据
            self.display_timer_sent_data(data_packet)
            
            # 更新状态
            enabled_timers = [f"TIM{i}" for i in [1,2,3,4] if self.timer_vars[f"tim{i}_enable"].get()]
            if enabled_timers:
                status_text = f"已发送配置: {', '.join(enabled_timers)}"
            else:
                status_text = "已发送配置: 所有定时器禁用"
            
            self.timer_send_status_var.set(status_text)
            
            # 更新统计
            self.send_stats['total_sent'] += 1
            self.send_stats_label.config(text=f"发送: {self.send_stats['total_sent']}次 | 错误: {self.send_stats['send_errors']}次")
            
            print(f"定时器配置发送成功: {len(data_packet)} 字节")
            
        except Exception as e:
            self.timer_send_status_var.set("发送失败!")
            self.send_stats['send_errors'] += 1
            self.send_stats_label.config(text=f"发送: {self.send_stats['total_sent']}次 | 错误: {self.send_stats['send_errors']}次")
            messagebox.showerror("错误", f"发送定时器配置失败: {e}")
    
    def stop_all_timers(self):
        """停止所有定时器"""
        if not self.is_serial_open or not self.serial_port:
            messagebox.showerror("错误", "串口未打开")
            return
        
        try:
            # 发送停止命令: 0xA1
            stop_cmd = bytearray([0xA1, 0xA1])
            bytes_sent = self.serial_port.write(stop_cmd)
            self.serial_port.flush()
            
            # 更新界面状态
            for tim_num in [1, 2, 3, 4]:
                self.timer_vars[f"tim{tim_num}_enable"].set(False)
            
            self.timer_send_status_var.set("已停止所有定时器")
            
            # 显示发送的命令
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.serial_text.insert(tk.END, f"[{timestamp}] → [发送] 停止所有定时器命令: A1 A1\n")
            self.serial_text.see(tk.END)
            
            print("已发送停止所有定时器命令")
            
        except Exception as e:
            messagebox.showerror("错误", f"停止定时器失败: {e}")
    
    def display_timer_sent_data(self, data):
        """显示发送的定时器配置数据"""
        # 解析并显示定时器数据
        enabled_timers = []
        disabled_timers = []
        
        for i, tim_num in enumerate([1, 2, 3, 4]):
            offset = 1 + i * 5
            enable = data[offset]
            frequency = (data[offset+1] << 16) | (data[offset+2] << 8) | data[offset+3]
            duty = data[offset+4]
            
            if enable == 1:
                enabled_timers.append(f"TIM{tim_num}({frequency}Hz,{duty}%)")
            else:
                disabled_timers.append(f"TIM{tim_num}(禁用)")
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # 显示启用的定时器
        if enabled_timers:
            timer_info = ', '.join(enabled_timers)
            self.serial_text.insert(tk.END, f"[{timestamp}] → [发送] 启用定时器: {timer_info}\n")
        
        # 显示禁用的定时器
        if disabled_timers:
            disabled_info = ', '.join(disabled_timers)
            self.serial_text.insert(tk.END, f"[{timestamp}] → [发送] 禁用定时器: {disabled_info}\n")
        
        # 如果全部禁用
        if not enabled_timers:
            self.serial_text.insert(tk.END, f"[{timestamp}] → [发送] 定时器状态: 所有定时器已禁用\n")
        
        self.serial_text.insert(tk.END, f"[{timestamp}] → [发送] 十六进制: {data.hex(' ').upper()}\n")
        self.serial_text.see(tk.END)
    
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
    app = SingleSerialTimerController(root)
    
    # 设置关闭事件处理
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # 启动GUI
    root.mainloop()


if __name__ == "__main__":
    main()