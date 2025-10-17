#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase C: 评分正式阶段
固定3×48试次的正式评分实验，无早停机制
与Phase B使用相同的评分机制和数据格式
"""

import os
import sys
import argparse
import hashlib
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog
import random
import math

# PsychoPy imports
from psychopy import visual, core, event, gui, data, monitors
from psychopy.hardware import keyboard
from psychopy import prefs
prefs.general['font'] = ['SimHei', 'Arial', 'Noto Sans CJK SC', 'Microsoft YaHei']

# 导入试次池管理器
from pools import create_pools_manager


class PhaseCExperiment:
    def __init__(self, config_path=None, subject_id=None, resume=False, allow_init=False):
        # 获取脚本所在目录，确保路径相对于脚本位置
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = config_path or os.path.join(self.script_dir, "config", "experiment.yaml")
        self.subject_id = subject_id
        self.resume = resume
        self.allow_init = allow_init
        self.phase = "C"
        
        # 初始化变量
        self.config = None
        self.config_checksum = None
        self.subject_profile = None
        self.stim_table = None
        self.stim_table_checksum = None
        self.win = None
        self.kb = None
        self.mouse = None
        self.face_images = {}
        self.self_image = None
        
        # 实验状态
        self.current_block = 1
        self.current_trial = 1
        self.completed_trials = []
        self.accuracy_history = []
        self.distance_accuracy_history = []
        self.vector_accuracy_history = []
        
        # 滑杆
        self.distance_slider = None
        self.com_slider = None
        self.pop_slider = None
        
        # 鼠标拖动状态
        self._dragging = None  # 当前拖动的滑杆名称
        
        # 试次池管理器
        self.pools_manager = None
        
        # 随机数生成器
        self.rng = None
        
    def load_config(self):
        """加载配置文件，如不存在则创建默认模板"""
        if not os.path.exists(self.config_path):
            if messagebox.askyesno("配置文件缺失", 
                                 f"未找到配置文件 {self.config_path}\n是否创建默认模板？"):
                self._create_default_config()
            else:
                config_file = filedialog.askopenfilename(
                    title="选择配置文件",
                    filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
                )
                if not config_file:
                    sys.exit("用户取消，程序退出")
                self.config_path = config_file
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            self.config = yaml.safe_load(config_content)
            # 计算配置文件校验和
            self.config_checksum = hashlib.sha256(config_content.encode()).hexdigest()[:16]
    
    def _create_default_config(self):
        """创建默认配置文件模板（与Phase A相同）"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        # 使用与Phase A相同的默认配置
        from phase_a import PhaseAExperiment
        phase_a = PhaseAExperiment()
        phase_a._create_default_config()

    def get_subject_info(self):
        """获取被试信息GUI"""
        # 首先尝试加载现有档案
        profile_path = os.path.join(
            self.script_dir,
            self.config['paths']['subjects_dir'],
            f"subj_{self.subject_id}_profile.json" if self.subject_id else "dummy.json"
        )
        
        existing_profile = {}
        if self.subject_id and os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                existing_profile = json.load(f)
        
        # 获取自我图像选项
        self_dir = os.path.join(self.script_dir, self.config['paths']['stimuli_self_dir'])
        self_images = []
        if os.path.exists(self_dir):
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                self_images.extend(Path(self_dir).glob(ext))
        self_images = [str(img.name) for img in self_images]
        
        if not self_images and self.config['self_stim']['required']:
            messagebox.showerror("错误", f"在 {self_dir} 中未找到自我图像文件")
            sys.exit(1)
        
        # 创建GUI
        dlg_data = {
            'SubjectID': self.subject_id or existing_profile.get('SubjectID', ''),
            'Name': existing_profile.get('Name', ''),
            'Sex': ['Male', 'Female', 'Other'],
            'Age': existing_profile.get('Age', 25),
            'Handedness': ['Right', 'Left', 'Both'],
            'Notes': existing_profile.get('Notes', ''),
            'Phase': 'C',
            'SelfImageFile': self_images
        }
        
        # 设置现有值
        if existing_profile.get('Sex') in dlg_data['Sex']:
            dlg_data['Sex'] = existing_profile['Sex']
        if existing_profile.get('Handedness') in dlg_data['Handedness']:
            dlg_data['Handedness'] = existing_profile['Handedness']
        if existing_profile.get('SelfImageFile') in self_images:
            dlg_data['SelfImageFile'] = existing_profile['SelfImageFile']
        
        dlg = gui.DlgFromDict(
            dictionary=dlg_data,
            title='被试信息 - Phase C',
            fixed=['Phase'],
            order=['SubjectID', 'Name', 'Sex', 'Age', 'Handedness', 'SelfImageFile', 'Notes']
        )
        
        if not dlg.OK:
            sys.exit("用户取消，程序退出")
        
        # 验证SubjectID
        import re
        if not re.match(r'^[A-Za-z0-9_-]+$', dlg_data['SubjectID']):
            messagebox.showerror("错误", "SubjectID只能包含字母、数字、下划线和横线")
            sys.exit(1)
        
        self.subject_id = dlg_data['SubjectID']
        
        # 生成SessionID
        session_id = f"{self.subject_id}_{self.phase}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # 计算NameHash和SelfImageSHA256
        name_hash = hashlib.sha256(dlg_data['Name'].encode()).hexdigest()[:16]
        
        self_image_sha256 = ""
        if dlg_data['SelfImageFile']:
            self_image_path = os.path.join(self_dir, dlg_data['SelfImageFile'])
            if os.path.exists(self_image_path):
                with open(self_image_path, 'rb') as f:
                    self_image_sha256 = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # 更新被试档案
        profile_data = {
            'SubjectID': self.subject_id,
            'Name': dlg_data['Name'],
            'NameHash': name_hash,
            'Sex': dlg_data['Sex'],
            'Age': dlg_data['Age'],
            'Handedness': dlg_data['Handedness'],
            'Notes': dlg_data['Notes'],
            'SelfImageFile': dlg_data['SelfImageFile'],
            'SelfImageSHA256': self_image_sha256,
            'CreatedAt': existing_profile.get('CreatedAt', datetime.now().isoformat()),
            'LastSessionAt': datetime.now().isoformat(),
            'SchemaVersion': self.config['meta']['schema_version'],
            'SessionID': session_id
        }
        
        os.makedirs(os.path.dirname(profile_path), exist_ok=True)
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        
        self.subject_profile = profile_data
        print(f"被试档案已更新: {profile_path}")

    def load_stim_table(self):
        """加载被试×刺激表"""
        stim_table_path = os.path.join(
            self.script_dir,
            self.config['paths']['subjects_dir'],
            f"subj_{self.subject_id}_stim_table.csv"
        )
        
        if not os.path.exists(stim_table_path):
            if self.allow_init:
                print("警告: Phase C 通常不应该创建新的刺激表，建议先运行 Phase A")
                # 这里可以调用创建逻辑，但会打印警告
                from phase_a import PhaseAExperiment
                phase_a = PhaseAExperiment(self.config_path, self.subject_id, False, True)
                phase_a.load_config()
                phase_a._create_stim_table(stim_table_path)
                
            else:
                raise ValueError(f"未找到被试刺激表: {stim_table_path}，请先运行 Phase A 或使用 --allow-init")
        
        # 加载现有表格
        self.stim_table = pd.read_csv(stim_table_path)
        # 计算校验和
        table_str = self.stim_table.to_csv(index=False)
        self.stim_table_checksum = hashlib.sha256(table_str.encode()).hexdigest()[:16]
        print(f"已加载被试刺激表: {stim_table_path}")

    def load_stimuli(self):
        """加载刺激图像"""
        faces_dir = os.path.join(self.script_dir, self.config['paths']['stimuli_faces_dir'])
        self_dir = os.path.join(self.script_dir, self.config['paths']['stimuli_self_dir'])
        
        # 布局参数（可调）
        self.face_size_x = 480  # 面孔大小（像素）
        self.face_size_y = 480  # 面孔大小（像素）
        self.top_height = 0.07 * self.window_height  # 上方高度
        
        # 加载他人面孔
        for stim_id in range(1, self.config['stimuli']['n_faces'] + 1):
            face_path = os.path.join(faces_dir, f"P{stim_id}.jpg")
            if not os.path.exists(face_path):
                face_path = os.path.join(faces_dir, f"P{stim_id}.png")
            
            if os.path.exists(face_path):
                self.face_images[stim_id] = visual.ImageStim(
                    win=self.win,
                    image=face_path,
                    size=(self.face_size_x, self.face_size_y),
                    pos=(0, 0)
                )
            else:
                # 创建占位符
                self.face_images[stim_id] = visual.Rect(
                    win=self.win,
                    width=self.face_size_x, height=self.face_size_y,
                    fillColor='gray',
                    pos=(0, 0)
                )
                print(f"警告: 未找到 {face_path}，使用占位符")
        
        # 加载自我面孔
        if self.subject_profile['SelfImageFile']:
            self_path = os.path.join(self_dir, self.subject_profile['SelfImageFile'])
            if os.path.exists(self_path):
                self.self_image = visual.ImageStim(
                    win=self.win,
                    image=self_path,
                    size=(self.face_size_x, self.face_size_y),
                    pos=(0, 0)
                )
                
                # 添加环形高亮（如果配置要求）
                if self.config['self_stim']['render_overlay_ring']:
                    self.self_ring = visual.Circle(
                        win=self.win,
                        radius=self.face_size_x/2 + 10,
                        lineColor='yellow',
                        lineWidth=3,
                        fillColor=None,
                        pos=(0, 0)
                    )
            else:
                print(f"警告: 未找到自我图像 {self_path}")

    def setup_experiment(self):
        """设置实验环境"""
        # >>> 新增：构建 Monitor，并创建 Window（避免 __blank__ 崩溃）
        mon, screen_idx, size_pix = self._build_monitor()

        # window units: use 'pix' globally to avoid deg conversions for GUI splash
        window_units = self.config.get('window_units', 'pix')
        # 背景色：从experiment.bg_color读取，默认纯黑
        bg = tuple(self.config.get('experiment', {}).get('bg_color', [-1, -1, -1]))

        self.win = visual.Window(
            size=size_pix,
            units=window_units,
            fullscr=bool(self.config['experiment'].get('fullscreen', True)),
            monitor=mon,
            screen=screen_idx,
            color=bg,
            allowGUI=True,
            checkTiming=False,   # ★ 关键：跳过 getActualFrameRate -> TextBox2
            waitBlanking=False   # 降低平台差异带来的卡顿
        )

        event.Mouse(visible=True, win=self.win)  # 显示鼠标，便于交互
        
        # 创建键盘和鼠标
        self.kb = keyboard.Keyboard()
        self.mouse = event.Mouse(win=self.win)
        
        # 创建滑杆
        slider_config = self.config.get('slider', {})
               
        # === 水平距离滑杆 ===
        distance_config = slider_config.get('distance_ticks', [-13, 0, 13])
        self.distance_slider = visual.Slider(
            win=self.win,
            ticks=distance_config,
            #labels=[str(x) for x in distance_config],
            labels=['比我低','比我高'],
            font='SimHei',
            pos=(0, -250),
            size=(950, 50),
            granularity=slider_config.get('distance_step', 0.1)
        )
        
        # === 右侧竖直滑杆（Com/Pop） ===
        component_config = slider_config.get('component_ticks', [-9,-6,-3,0,3,6,9])
        component_step = slider_config.get('component_step', 1)

        self.com_slider = visual.Slider(
            win=self.win,
            ticks=component_config,
            #labels=[str(x) for x in component_config],
            labels=['减小','增大'],
            font='SimHei',
            pos=(550, 40),
            size=(400, 50),        # 长边在前
            granularity=component_step,
            ori=90
        )
        self.pop_slider = visual.Slider(
            win=self.win,
            ticks=component_config,
            #labels=[str(x) for x in component_config],
            labels=['减小','增大'],
            font='SimHei',
            pos=(700, 40),
            size=(400, 50),
            granularity=component_step,
            ori=90
        )

        self.dim1_color=self.config['dimension_colors']['dim1_hint']
        self.dim2_color=self.config['dimension_colors']['dim2_hint']
        self.com_label = self.create_text_stim("能力",color=self.dim1_color, style='small',bold=True,italic=True, pos=(550, 280))
        self.pop_label = self.create_text_stim("受欢迎度",color=self.dim2_color, style='small',bold=True,italic=True, pos=(700, 280))


        # 拖拽状态
        self._dragging = None
        # 鼠标拖拽像素"死区"（避免轻微抖动触发）
        self._drag_deadzone_px = 2
        
        # 创建自我环形高亮
        if self.config.get('self_stim', {}).get('render_overlay_ring', True):
            self.self_ring = visual.Circle(
                win=self.win,
                radius=80,  # 像素单位
                lineColor='yellow',
                lineWidth=3,
                fillColor=None,
                pos=(0, 0)
            )
        
        # 初始化随机数生成器
        seed_sequence = np.random.SeedSequence([
            self.config['experiment']['seed'],
            hash(self.subject_id) % (2**31),
            hash(f"phase_{self.phase}") % (2**31)
        ])
        self.rng = np.random.default_rng(seed_sequence)
        
        # 初始化试次池管理器
        self.pools_manager = create_pools_manager(self.config, self.subject_id, self.script_dir)
        
        print("实验环境设置完成")

    def _slider_hit(self, mx, my, slider):
        """检查鼠标是否在滑杆区域内"""
        slider_x, slider_y = slider.pos
        slider_width, slider_height = slider.size
        
        # 计算滑杆的边界（给一个≥40px的hitbox）
        if slider.ori == 90:  # 垂直滑杆
            left_bound = slider_x - 40
            right_bound = slider_x + 40
            top_bound = slider_y + slider_height/2
            bottom_bound = slider_y - slider_height/2
        else:  # 水平滑杆
            left_bound = slider_x - slider_width/2
            right_bound = slider_x + slider_width/2
            top_bound = slider_y + 40
            bottom_bound = slider_y - 40
        
        return (left_bound <= mx <= right_bound and 
                bottom_bound <= my <= top_bound)
    
    def _pix_to_rating(self, mx, my, slider):
        """将鼠标像素坐标映射到滑杆值域"""
        slider_x, slider_y = slider.pos
        slider_width, slider_height = slider.size
        
        if slider.ori == 90:  # 垂直滑杆
            # 使用 y 坐标
            top_bound = slider_y + slider_height/2
            bottom_bound = slider_y - slider_height/2
            normalized_pos = (my - bottom_bound) / slider_height
        else:  # 水平滑杆
            # 使用 x 坐标
            left_bound = slider_x - slider_width/2
            right_bound = slider_x + slider_width/2
            normalized_pos = (mx - left_bound) / slider_width
        
        # 线性映射到刻度范围
        min_val = slider.ticks[0]
        max_val = slider.ticks[-1]
        mapped_value = min_val + normalized_pos * (max_val - min_val)
        
        # 裁剪到刻度范围
        clipped_value = np.clip(mapped_value, min_val, max_val)
        
        # 按步长量化
        step = slider.granularity
        quantized_value = round(clipped_value / step) * step
        
        return quantized_value

    def wait_with_deadline_or_confirm(self, deadline_s):
        """等待指定时间或按空格确认"""
        if deadline_s <= 0:
            # 无限等待直到按空格
            event.clearEvents()
            event.waitKeys(keyList=['space', 'escape'])
        else:
            # 定时等待，可提前按空格结束
            start_time = core.getTime()
            while core.getTime() - start_time < deadline_s:
                keys = event.getKeys(keyList=['space', 'escape'])
                if keys:
                    break
                core.wait(0.01)  # 避免CPU占用过高


    def create_text_stim(self, text, style='body', pos=(0, 0), alignText='center', anchorHoriz='center',color='white',bold=False,italic=False):
        """创建文本刺激的助手方法"""
        # 根据窗口大小计算文本参数
        self.window_width = self.win.size[0]
        self.window_height = self.win.size[1]
        window_width = self.window_width
        window_height = self.window_height
        wrapWidth = 0.9 * window_width
        
        if style == 'title':
            height = 0.085 * window_height  # 像素
        elif style == 'body':
            height = 0.045 * window_height  # 像素
        elif style == 'small':
            height = 0.025 * window_height  # 像素
        else:
            height = float(style) * window_height # 像素
        
        return visual.TextStim(
            win=self.win,
            text=text,
            height=height,
            wrapWidth=wrapWidth,
            pos=pos,
            alignText=alignText,
            anchorHoriz=anchorHoriz,
            color=color,
            bold=bold,
            italic=italic,
            font='SimHei'
        )

    def _build_monitor(self):
        """Create a PsychoPy Monitor object from YAML config (with safe fallbacks)."""
        cfg = self.config
        mon_cfg = cfg.get('monitor', {})
        # fallbacks
        size_pix = tuple(mon_cfg.get('size_pix', cfg['experiment'].get('window_size', [1920, 1080])))
        width_cm = float(mon_cfg.get('width_cm', 34.0))      # safe default for 24" 16:9
        distance_cm = float(mon_cfg.get('distance_cm', 60.0))
        screen_idx = int(mon_cfg.get('screen', 0))
        name = mon_cfg.get('name', 'lab_default')

        mon = monitors.Monitor(name, autoLog=False)
        mon.setSizePix(size_pix)
        mon.setWidth(width_cm)
        mon.setDistance(distance_cm)
        # PsychoPy 不要求必须 saveMon；不落地到本机 Monitor Center 以保持可移植
        # monitors.saveMonitorAttributes(name)  # <- 不要强制保存
        return mon, screen_idx, size_pix

    def load_checkpoint(self):
        """加载断点文件"""
        if not self.resume:
            return
        
        checkpoint_path = os.path.join(
            self.script_dir,
            self.config['paths']['checkpoints_dir'],
            f"subj_{self.subject_id}_{self.phase}.json"
        )
        
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            self.current_block = checkpoint.get('current_block', 1)
            self.current_trial = checkpoint.get('current_trial', 1)
            self.completed_trials = checkpoint.get('completed_trials', [])
            self.accuracy_history = checkpoint.get('accuracy_history', [])
            
            # 恢复随机数状态
            if 'rng_state' in checkpoint:
                self.rng.bit_generator.state = checkpoint['rng_state']
            
            print(f"从断点恢复: Block {self.current_block}, Trial {self.current_trial}")

    def save_checkpoint(self):
        """保存断点文件"""
        checkpoint_path = os.path.join(
            self.script_dir,
            self.config['paths']['checkpoints_dir'],
            f"subj_{self.subject_id}_{self.phase}.json"
        )
        
        checkpoint = {
            'current_block': self.current_block,
            'current_trial': self.current_trial,
            'completed_trials': self.completed_trials,
            'accuracy_history': self.accuracy_history,
            'rng_state': self.rng.bit_generator.state,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, default=str)

    def generate_trials_for_block(self, block_num):
        """从试次池生成指定块的试次"""
        return self.pools_manager.get_block_trials('C', block_num)
    
    def _normalize_trial_data(self, trial_data):
        """trial 适配器：将不同格式的试次数据标准化为字典格式"""
        if isinstance(trial_data, dict):
            if 'P1' in trial_data and 'P2' in trial_data:
                return trial_data
            else:
                raise ValueError(f"试次数据缺少 P1 或 P2 字段: {trial_data}")
        elif isinstance(trial_data, (list, tuple)) and len(trial_data) == 2:
            return {'P1': int(trial_data[0]), 'P2': int(trial_data[1])}
        else:
            raise ValueError(f"不支持的试次数据格式: {trial_data}")
    
    def _apply_vertical_direction(self, val):
        """根据配置应用垂直方向映射"""
        direction = self.config.get('slider', {}).get('vertical_direction', 'up_increase')
        if direction == 'up_decrease':
            return -val
        else:  # up_increase
            return val

    def get_stim_coords(self, stim_id):
        """获取刺激坐标"""
        if stim_id == 0:  # 自我
            return (0, 0)
        else:
            stim_info = self.stim_table[self.stim_table['StimID'] == stim_id].iloc[0]
            return (stim_info['com'], stim_info['pop'])

    def run_trial(self, trial_data):
        """执行单个评分试次（新时序：Self → FixWhite → P1 → Purple → Self → FixWhite → P2）"""
        timing = self.config['phase_c']
        
        # 标准化试次数据格式
        trial_data = self._normalize_trial_data(trial_data)
        trial_result = trial_data.copy()
        
        # 获取刺激坐标
        p1_coords = self.get_stim_coords(trial_data['P1'])
        p2_coords = self.get_stim_coords(trial_data['P2'])
        
        trial_result.update({
            'P1_com': p1_coords[0],
            'P1_pop': p1_coords[1],
            'P2_com': p2_coords[0],
            'P2_pop': p2_coords[1]
        })
        
        # 计算角度和距离信息
        for p_name, p_id in [('P1', trial_data['P1']), ('P2', trial_data['P2'])]:
            if p_id == 0:  # 自我
                trial_result[f'{p_name}_angle_deg'] = ""
                trial_result[f'{p_name}_angle_bin'] = ""
                trial_result[f'{p_name}_dist'] = 0
            else:
                stim_info = self.stim_table[self.stim_table['StimID'] == p_id].iloc[0]
                trial_result[f'{p_name}_angle_deg'] = stim_info['angle_deg']
                trial_result[f'{p_name}_angle_bin'] = stim_info['angle_bin']
                trial_result[f'{p_name}_dist'] = stim_info['distance']
        
        # === 1. Self 呈现 (2.5s) ===
        self._show_self_phase(timing['self_s'])
        trial_result['Onset_Self1_MS'] = core.getTime() * 1000
        core.wait(timing['self_s'])
        trial_result['Offset_Self1_MS'] = core.getTime() * 1000
        
        # === 2. 白色注视点 (jitter 1-3s) ===
        fixation_duration = self.rng.uniform(*timing['fixation_white_jitter_s'])
        trial_result['Onset_FixWhite1_MS'] = core.getTime() * 1000
        fixation_white  = self.create_text_stim('+', style='title',color='white')
        fixation_white.draw()
        self.win.flip()
        core.wait(fixation_duration)
        trial_result['Offset_FixWhite1_MS'] = core.getTime() * 1000
        trial_result['FixWhite1_S'] = fixation_duration
        
        # === 3. P1 评分阶段 ===
        p1_result = self.run_rating_phase(trial_data['P1'], "P1", timing['p1_s'])
        trial_result.update(p1_result)
        
        # === 4. 紫色注视点 (3s) ===
        trial_result['Onset_Purple_MS'] = core.getTime() * 1000
        fixation_purple = self.create_text_stim('+', style='title',color='purple')
        fixation_purple.draw()
        self.win.flip()
        core.wait(timing['purple_s'])
        trial_result['Offset_Purple_MS'] = core.getTime() * 1000
        
        # === 5. Self 呈现 (2.5s) ===
        self._show_self_phase(timing['p2_self_s'])
        trial_result['Onset_Self2_MS'] = core.getTime() * 1000
        core.wait(timing['p2_self_s'])
        trial_result['Offset_Self2_MS'] = core.getTime() * 1000
        
        # === 6. 白色注视点 (jitter 1-3s) ===
        fixation_duration2 = self.rng.uniform(*timing['p2_fixation_white_jitter_s'])
        trial_result['Onset_FixWhite2_MS'] = core.getTime() * 1000
        fixation_white.draw()
        self.win.flip()
        core.wait(fixation_duration2)
        trial_result['Offset_FixWhite2_MS'] = core.getTime() * 1000
        trial_result['FixWhite2_S'] = fixation_duration2
        
        # === 7. P2 评分阶段 ===
        p2_result = self.run_rating_phase(trial_data['P2'], "P2", timing['p2_s'])
        trial_result.update(p2_result)
        
        # 计算总体正确性（either 策略）
        p1_distance_correct = trial_result.get('Correct_P1_distance', 0)
        p2_distance_correct = trial_result.get('Correct_P2_distance', 0)
        p1_vector_correct = trial_result.get('Correct_P1_vector', 0)
        p2_vector_correct = trial_result.get('Correct_P2_vector', 0)
        
        trial_result['Is_Correct_distance_trial'] = 1 if (p1_distance_correct and p2_distance_correct) else 0
        trial_result['Is_Correct_vector_trial'] = 1 if (p1_vector_correct and p2_vector_correct) else 0
        trial_result['Is_Correct_either_trial'] = max(trial_result['Is_Correct_distance_trial'], trial_result['Is_Correct_vector_trial'])
        trial_result['Is_Correct'] = trial_result['Is_Correct_either_trial']  # 保持兼容性
        
        return trial_result
    
    def _show_self_phase(self, duration):
        """显示自我面孔阶段"""
        if self.self_image:
            self.self_image.pos = (0, 160)
            self.self_image.draw()
            
            # 绘制自我环形高亮
            if hasattr(self, 'self_ring'):
                self.self_ring.pos = (0, 160)
                self.self_ring.draw()
            
            self.win.flip()

    def run_rating_phase(self, stim_id, phase_name, deadline_s):
        """执行单个刺激的评分阶段（三个滑杆：距离+Com+Pop）"""
        # 重置滑杆
        self.distance_slider.reset()
        self.com_slider.reset()
        self.pop_slider.reset()
        
        # 获取刺激图像
        if stim_id == 0:
            stim_img = self.self_image
            show_ring = True
        else:
            stim_img = self.face_images[stim_id]
            show_ring = False
        
        stim_img.pos = (0, 160)  # 像素单位，居中稍靠上
        
        # 计算目标值
        coords = self.get_stim_coords(stim_id)
        target_dist = math.sqrt(coords[0]**2 + coords[1]**2)
        target_dcom = -coords[0]  # 到S(0,0)的修正向量
        target_dpop = -coords[1]
        
        # 指导语
        instruction1_text = f"""
请评估该人物与你的社会等级差异
"""     
        instruction2_text = f"""
全部确认后，按 空格 下一步，尽可能快的回答
"""     
        instruction3_text = f"""
想
象
自
己
如
何
达
到
该
人
物
的
社
会
等
级
"""     
        instruction1 = self.create_text_stim(instruction1_text, style='0.035', pos=(0, -400))
        instruction2 = self.create_text_stim(instruction2_text, color='grey',style='small',bold=True, italic=True, pos=(0, -450))
        instruction3 = self.create_text_stim(instruction3_text, style='0.035',bold=True, pos=(850,40))
        #self.com_label = self.create_text_stim("Com", style='small', pos=(550, 260))
        #self.pop_label = self.create_text_stim("Pop", style='small', pos=(750, 260))
        
        # 开始评分
        onset_time = core.getTime() * 1000
        distance_response = None
        com_response = None
        pop_response = None
        rt_first = None
        rt_final = None
        timeout = False
        first_move = True
        
        self.kb.clearEvents()
        self.mouse.clickReset()
        
        start_time = core.getTime()
        
        # 主循环
        while True:
            # 检查是否超时
            if deadline_s > 0 and (core.getTime() - start_time) >= deadline_s:
                break
            
            # 绘制刺激
            stim_img.draw()
            
            # 绘制自我环形高亮
            if show_ring and hasattr(self, 'self_ring'):
                self.self_ring.pos = stim_img.pos
                self.self_ring.draw()
            
            # 绘制指导语和标签
            instruction1.draw()
            instruction2.draw()
            instruction3.draw()
            self.com_label.draw()
            self.pop_label.draw()
            
            # 绘制滑杆
            self.distance_slider.draw()
            self.com_slider.draw()
            self.pop_slider.draw()
            
            self.win.flip()
            
            # 检查鼠标输入
            mouse_pressed = self.mouse.getPressed()[0]
            mouse_pos = self.mouse.getPos()
            mx, my = mouse_pos
            
            if mouse_pressed:
                if first_move:
                    rt_first = (core.getTime() - start_time) * 1000
                    first_move = False
                
                # 检查哪个滑杆被点击
                if self._slider_hit(mx, my, self.distance_slider):
                    self._dragging = 'distance'
                elif self._slider_hit(mx, my, self.com_slider):
                    self._dragging = 'com'
                elif self._slider_hit(mx, my, self.pop_slider):
                    self._dragging = 'pop'
            
            # 处理拖动
            if self._dragging and mouse_pressed:
                if self._dragging == 'distance':
                    rating_value = self._pix_to_rating(mx, my, self.distance_slider)
                    self.distance_slider.setRating(rating_value)
                    self.distance_slider.markerPos = rating_value
                elif self._dragging == 'com':
                    rating_value = self._pix_to_rating(mx, my, self.com_slider)
                    rating_value = self._apply_vertical_direction(rating_value)
                    self.com_slider.setRating(rating_value)
                    self.com_slider.markerPos = rating_value
                elif self._dragging == 'pop':
                    rating_value = self._pix_to_rating(mx, my, self.pop_slider)
                    rating_value = self._apply_vertical_direction(rating_value)
                    self.pop_slider.setRating(rating_value)
                    self.pop_slider.markerPos = rating_value
            elif not mouse_pressed:
                self._dragging = None
            
            # 检查键盘输入
            keys = self.kb.getKeys(['left', 'right', 'up', 'down', 'a', 'd', 'w', 's', 'shift', 'space', 'escape'])
            for key in keys:
                if key.name == 'escape':
                    self.cleanup()
                    sys.exit("用户中断实验")
                elif key.name == 'space':
                    # 确认当前评分
                    distance_response = self.distance_slider.getRating()
                    com_response = self.com_slider.getRating()
                    pop_response = self.pop_slider.getRating()
                    
                    if distance_response is not None or com_response is not None or pop_response is not None:
                        rt_final = (core.getTime() - start_time) * 1000
                        break
                elif key.name in ['left', 'right', 'a', 'd']:
                    if first_move:
                        rt_first = (core.getTime() - start_time) * 1000
                        first_move = False
                    
                    # 调节距离滑杆
                    current_rating = self.distance_slider.getRating() or 0
                    step = self.config.get('slider', {}).get('distance_step', 0.1)
                    
                    if key.name in ['left', 'a']:
                        new_rating = current_rating - step
                    else:
                        new_rating = current_rating + step
                    
                    # 裁剪到刻度范围
                    min_val = self.distance_slider.ticks[0]
                    max_val = self.distance_slider.ticks[-1]
                    clipped_rating = np.clip(new_rating, min_val, max_val)
                    
                    self.distance_slider.setRating(clipped_rating)
                    self.distance_slider.markerPos = clipped_rating
                    
                elif key.name in ['up', 'down', 'w', 's']:
                    if first_move:
                        rt_first = (core.getTime() - start_time) * 1000
                        first_move = False
                    
                    # 调节垂直滑杆
                    step = self.config.get('slider', {}).get('component_step', 1)
                    
                    if key.name in ['up', 'w']:
                        # 调节 Com 滑杆
                        current_rating = self.com_slider.getRating() or 0
                        new_rating = current_rating + step
                        # 应用垂直方向映射
                        new_rating = self._apply_vertical_direction(new_rating)
                        # 裁剪到刻度范围
                        min_val = self.com_slider.ticks[0]
                        max_val = self.com_slider.ticks[-1]
                        clipped_rating = np.clip(new_rating, min_val, max_val)
                        self.com_slider.setRating(clipped_rating)
                        self.com_slider.markerPos = clipped_rating
                    else:
                        # 调节 Pop 滑杆
                        current_rating = self.pop_slider.getRating() or 0
                        new_rating = current_rating + step
                        # 应用垂直方向映射
                        new_rating = self._apply_vertical_direction(new_rating)
                        # 裁剪到刻度范围
                        min_val = self.pop_slider.ticks[0]
                        max_val = self.pop_slider.ticks[-1]
                        clipped_rating = np.clip(new_rating, min_val, max_val)
                        self.pop_slider.setRating(clipped_rating)
                        self.pop_slider.markerPos = clipped_rating
            
            if distance_response is not None or com_response is not None or pop_response is not None:
                break
        
        offset_time = core.getTime() * 1000
        
        # 获取最终评分
        if distance_response is None:
            distance_response = self.distance_slider.getRating() or 0
        if com_response is None:
            com_response = self.com_slider.getRating() or 0
        if pop_response is None:
            pop_response = self.pop_slider.getRating() or 0
        
        if rt_final is None:
            rt_final = (core.getTime() - start_time) * 1000
        
        # 计算正确性
        # 距离任务
        abs_error_distance = abs(distance_response - target_dist)
        correct_distance = 1 if abs_error_distance <= self.config['tolerance']['distance_abs'] else 0
        
        # 向量任务
        abs_error_dcom = abs(com_response - target_dcom)
        abs_error_dpop = abs(pop_response - target_dpop)
        correct_vector = 1 if (abs_error_dcom <= self.config['tolerance']['component_abs'] and 
                              abs_error_dpop <= self.config['tolerance']['component_abs']) else 0
        
        result = {
            f'Onset_{phase_name}_MS': onset_time,
            f'Offset_{phase_name}_MS': offset_time,
            f'RT_{phase_name}_first_MS': rt_first or "",
            f'RT_{phase_name}_final_MS': rt_final or "",
            
            # 距离任务
            f'Resp_{phase_name}_distance': distance_response,
            f'TargetDist_{phase_name}': target_dist,
            f'AbsError_{phase_name}_distance': abs_error_distance,
            f'Correct_{phase_name}_distance': correct_distance,
            f'RT_{phase_name}_distance_MS': rt_final or "",
            
            # 向量任务
            f'Target_{phase_name}_dCom': target_dcom,
            f'Target_{phase_name}_dPop': target_dpop,
            f'Resp_{phase_name}_dCom': com_response,
            f'Resp_{phase_name}_dPop': pop_response,
            f'AbsError_{phase_name}_dCom': abs_error_dcom,
            f'AbsError_{phase_name}_dPop': abs_error_dpop,
            f'Correct_{phase_name}_vector': correct_vector,
            f'RT_{phase_name}_vector_MS': rt_final or "",
            
            # 兼容性字段
            f'Resp_{phase_name}': distance_response,
            f'TargetDist_{phase_name}': target_dist,
            f'AbsError_{phase_name}': abs_error_distance,
            f'Correct_{phase_name}': correct_distance,
            f'Timeout_{phase_name}': 1 if timeout else 0
        }
        
        return result

    def save_trial_data(self, trial_result):
        """保存试次数据（与Phase B使用相同的格式）"""
        output_path = os.path.join(
            self.script_dir,
            self.config['paths']['data_dir'],
            'C',
            f"subj_{self.subject_id}_trials.csv"
        )
        
        # 添加审计列
        full_result = {
            'SubID': self.subject_id,
            'SessionID': self.subject_profile['SessionID'],
            'Phase': self.phase,
            'Day': "",  # Phase C 没有天数概念
            'Block': self.current_block,
            'Trial': self.current_trial,
            'NameHash': self.subject_profile['NameHash'],
            'SchemaVer': self.config['meta']['schema_version'],
            'ConfigChecksum': self.config_checksum,
            'SubjectMapChecksum': self.stim_table_checksum,
            'SelfImageSHA256': self.subject_profile['SelfImageSHA256'],
            'Seed': self.config['experiment']['seed']
        }
        full_result.update(trial_result)
        
        # 计算累积准确率（either 策略）
        correct_distance_ratings = sum(self.distance_accuracy_history) + trial_result.get('Correct_P1_distance', 0) + trial_result.get('Correct_P2_distance', 0)
        correct_vector_ratings = sum(self.vector_accuracy_history) + trial_result.get('Correct_P1_vector', 0) + trial_result.get('Correct_P2_vector', 0)
        
        full_result['AccuToDate_distance'] = correct_distance_ratings / (len(self.distance_accuracy_history) + 2) if (len(self.distance_accuracy_history) + 2) > 0 else 0
        full_result['AccuToDate_vector'] = correct_vector_ratings / (len(self.vector_accuracy_history) + 2) if (len(self.vector_accuracy_history) + 2) > 0 else 0
        full_result['AccuToDate'] = max(full_result['AccuToDate_distance'], full_result['AccuToDate_vector'])  # either 策略
        
        # 确保列顺序（包含所有新字段）
        columns = [
            'SubID', 'SessionID', 'Phase', 'Day', 'Block', 'Trial', 'NameHash',
            'SchemaVer', 'ConfigChecksum', 'SubjectMapChecksum', 'SelfImageSHA256', 'Seed',
            'P1', 'P2', 'P1_com', 'P1_pop', 'P2_com', 'P2_pop',
            'P1_angle_deg', 'P2_angle_deg', 'P1_angle_bin', 'P2_angle_bin',
            'P1_dist', 'P2_dist',
            # 时序事件
            'Onset_Self1_MS', 'Offset_Self1_MS',
            'Onset_FixWhite1_MS', 'Offset_FixWhite1_MS', 'FixWhite1_S',
            'Onset_P1_MS', 'Offset_P1_MS',
            'Onset_Purple_MS', 'Offset_Purple_MS',
            'Onset_Self2_MS', 'Offset_Self2_MS',
            'Onset_FixWhite2_MS', 'Offset_FixWhite2_MS', 'FixWhite2_S',
            'Onset_P2_MS', 'Offset_P2_MS',
            # 距离任务
            'Resp_P1_distance', 'TargetDist_P1', 'AbsError_P1_distance', 'Correct_P1_distance', 'RT_P1_distance_MS',
            'Resp_P2_distance', 'TargetDist_P2', 'AbsError_P2_distance', 'Correct_P2_distance', 'RT_P2_distance_MS',
            # 向量任务
            'Target_P1_dCom', 'Target_P1_dPop', 'Resp_P1_dCom', 'Resp_P1_dPop', 'AbsError_P1_dCom', 'AbsError_P1_dPop', 'Correct_P1_vector', 'RT_P1_vector_MS',
            'Target_P2_dCom', 'Target_P2_dPop', 'Resp_P2_dCom', 'Resp_P2_dPop', 'AbsError_P2_dCom', 'AbsError_P2_dPop', 'Correct_P2_vector', 'RT_P2_vector_MS',
            # 总体正确性
            'Is_Correct_distance_trial', 'Is_Correct_vector_trial', 'Is_Correct_either_trial', 'Is_Correct',
            'AccuToDate_distance', 'AccuToDate_vector', 'AccuToDate',
            # 兼容性字段
            'Resp_P1', 'Timeout_P1', 'AbsError_P1', 'Correct_P1', 'RT_P1_first_MS', 'RT_P1_final_MS',
            'Resp_P2', 'Timeout_P2', 'AbsError_P2', 'Correct_P2', 'RT_P2_first_MS', 'RT_P2_final_MS'
        ]
        
        # 确保所有列都存在
        for col in columns:
            if col not in full_result:
                full_result[col] = ""
        
        # 写入文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df_row = pd.DataFrame([full_result])
        df_row = df_row[columns]  # 确保列顺序
        
        if not os.path.exists(output_path):
            df_row.to_csv(output_path, index=False)
        else:
            df_row.to_csv(output_path, mode='a', header=False, index=False)
        
        self.completed_trials.append(full_result)
        
        # 保存精简版CSV
        self.save_simplified_csv(full_result)
        
        # 更新准确率历史（按评分次数计算）
        self.distance_accuracy_history.extend([
            trial_result.get('Correct_P1_distance', 0),
            trial_result.get('Correct_P2_distance', 0)
        ])
        self.vector_accuracy_history.extend([
            trial_result.get('Correct_P1_vector', 0),
            trial_result.get('Correct_P2_vector', 0)
        ])
        # 保持兼容性
        self.accuracy_history.extend([
            trial_result.get('Correct_P1', 0),
            trial_result.get('Correct_P2', 0)
        ])

    def save_simplified_csv(self, trial_result):
        """保存精简版CSV（包含距离和向量任务）"""
        output_path = os.path.join(
            self.script_dir,
            self.config['paths']['data_dir'],
            'C',
            f"subj_{self.subject_id}.csv"
        )
        
        # 构建精简版数据
        simplified_data = {
            'SubID': trial_result.get('SubID', ''),
            'SessionID': trial_result.get('SessionID', ''),
            'Block': trial_result.get('Block', ''),
            'Trial': trial_result.get('Trial', ''),
            'P1': trial_result.get('P1', ''),
            'P2': trial_result.get('P2', ''),
            'P1_com': trial_result.get('P1_com', ''),
            'P1_pop': trial_result.get('P1_pop', ''),
            'P2_com': trial_result.get('P2_com', ''),
            'P2_pop': trial_result.get('P2_pop', ''),
            # 距离任务
            'Dist_P1': trial_result.get('TargetDist_P1', ''),
            'Dist_P2': trial_result.get('TargetDist_P2', ''),
            'P1_Rate': trial_result.get('Resp_P1_distance', ''),
            'P1_Correct_Dist': trial_result.get('Correct_P1_distance', ''),
            'Onset_P1': trial_result.get('Onset_P1_MS', ''),
            'Offset_P1': trial_result.get('Offset_P1_MS', ''),
            'P1_RT_Dist': trial_result.get('RT_P1_distance_MS', ''),
            'P2_Rate': trial_result.get('Resp_P2_distance', ''),
            'P2_Correct_Dist': trial_result.get('Correct_P2_distance', ''),
            'Onset_P2': trial_result.get('Onset_P2_MS', ''),
            'Offset_P2': trial_result.get('Offset_P2_MS', ''),
            'P2_RT_Dist': trial_result.get('RT_P2_distance_MS', ''),
            # 向量任务
            'P1_Target_dCom': trial_result.get('Target_P1_dCom', ''),
            'P1_Target_dPop': trial_result.get('Target_P1_dPop', ''),
            'P1_Resp_dCom': trial_result.get('Resp_P1_dCom', ''),
            'P1_Resp_dPop': trial_result.get('Resp_P1_dPop', ''),
            'P1_Correct_Vector': trial_result.get('Correct_P1_vector', ''),
            'P1_RT_Vector': trial_result.get('RT_P1_vector_MS', ''),
            'P2_Target_dCom': trial_result.get('Target_P2_dCom', ''),
            'P2_Target_dPop': trial_result.get('Target_P2_dPop', ''),
            'P2_Resp_dCom': trial_result.get('Resp_P2_dCom', ''),
            'P2_Resp_dPop': trial_result.get('Resp_P2_dPop', ''),
            'P2_Correct_Vector': trial_result.get('Correct_P2_vector', ''),
            'P2_RT_Vector': trial_result.get('RT_P2_vector_MS', '')
        }
        
        # 确保所有列都存在
        columns = [
            'SubID', 'SessionID', 'Block', 'Trial',
            'P1', 'P2', 'P1_com', 'P1_pop', 'P2_com', 'P2_pop',
            'Dist_P1', 'Dist_P2', 'P1_Rate', 'P1_Correct_Dist', 'Onset_P1', 'Offset_P1', 'P1_RT_Dist',
            'P2_Rate', 'P2_Correct_Dist', 'Onset_P2', 'Offset_P2', 'P2_RT_Dist',
            'P1_Target_dCom', 'P1_Target_dPop', 'P1_Resp_dCom', 'P1_Resp_dPop', 'P1_Correct_Vector', 'P1_RT_Vector',
            'P2_Target_dCom', 'P2_Target_dPop', 'P2_Resp_dCom', 'P2_Resp_dPop', 'P2_Correct_Vector', 'P2_RT_Vector'
        ]
        
        for col in columns:
            if col not in simplified_data:
                simplified_data[col] = ''
        
        # 写入文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df_row = pd.DataFrame([simplified_data])
        df_row = df_row[columns]  # 确保列顺序
        
        if not os.path.exists(output_path):
            df_row.to_csv(output_path, index=False)
        else:
            df_row.to_csv(output_path, mode='a', header=False, index=False)

    def print_block_summary(self, block_num):
        """打印block总结"""
        if len(self.accuracy_history) == 0:
            return
        
        # 计算当前block的统计
        ratings_per_trial = 2
        start_idx = (block_num - 1) * self.config['phase_c']['trials_per_block'] * ratings_per_trial
        end_idx = block_num * self.config['phase_c']['trials_per_block'] * ratings_per_trial
        
        if start_idx < len(self.accuracy_history):
            block_accuracy = self.accuracy_history[start_idx:min(end_idx, len(self.accuracy_history))]
            block_acc_rate = sum(block_accuracy) / len(block_accuracy) if len(block_accuracy) > 0 else 0
            
            overall_acc_rate = sum(self.accuracy_history) / len(self.accuracy_history)
            
            print(f"Block {block_num} 完成:")
            print(f"  - Block准确率: {block_acc_rate:.3f}")
            print(f"  - 总体准确率: {overall_acc_rate:.3f}")
            print(f"  - 总评分次数: {len(self.accuracy_history)}")

    def run_experiment(self):
        """运行实验主循环"""
        print(f"\n开始 Phase C 实验 - 被试: {self.subject_id}")
        print("=" * 50)
        
        total_blocks = self.config['phase_c']['blocks']
        trials_per_block = self.config['phase_c']['trials_per_block']
        
        print(f"Phase C: 固定 {total_blocks} × {trials_per_block} 试次的正式评分实验")
        
        # 显示指导语
        instruction_title_text = f""" “谁是更优秀的企业家”
自我参照判断任务"""
        instruction_body_text =f"""
恭喜您顺利完成第二轮实验。接下来进入第二轮。在本轮中,您将学习如何以“自己”为参照来做一系列比较判断。

实验会依次呈现来自81位企业家的面孔(其中包含您自己的照片)。这些企业家先前已分别依据两个彼此独立的

标准进行评定:能力与受欢迎度。您在上一轮已完成对这81位企业家社会等级的学习。请再次注意:两个标准

彼此独立,一位企业家在某一标准上的等级并不决定其在另一标准上的等级。本轮中,您需要同时考虑这两个

标准,判断每位企业家的社会等级相对于您本人的差异,并想象自己应如何调整才能达到该人物的社会等级。

按空格进入下一页。"""
        instruction_title_text1 = """ 实验流程:"""

        instruction_body_text1 =f"""
在每一次判断中,您将依次对两名企业家做基于自我参照的判断。画面会先呈现您的面孔,随后出现白色“+”

注视点,然后出现其中一位企业家的面孔;当您完成一轮后,会出现紫色“+”,每当您看到到白色“+”时请做好准备;

当企业家的面孔出现时请立即作答。如此循环进行,直到您完成全部判断。当前试次结束时会出现绿色“+”,

随后进入下一试次;随后再次呈现您的面孔与白色“+”,以相同方式继续。屏幕底部会显示一个刻度条,

用于报告社会等级差异的大小(左侧表示“比您低”,右侧表示“比您高”);屏幕右侧会同时显示两个刻度条

分别对应先前学习过的“能力”和“受欢迎度”,用于报告您认为自己需要如何调整：向上表示在该标准上提高,

向下表示在该标准上降低。您可以用鼠标拖动游标,也可使用“↑”“↓”或“←”“→”或“A”“D”进行微调;按空格锁

定当前游标位置,完成本轮所有判断后按Enter进入下一页。本次实验不设时间限制,请尽可能准确地完成

自我参照判断与想象;若总体准确率低于95%,将需要回到第一轮继续学习直至达到95%方可进入下一轮。

理解无误后按空格开始实验。
"""
        
        instruction_title = self.create_text_stim(instruction_title_text,bold=True, pos=(0,450),style='title')
        instruction_title.draw()
        instruction_text = self.create_text_stim(instruction_body_text, pos=(0,0),style='0.03')
        instruction_text.draw()
        self.win.flip()
        # 清空事件缓冲区
        event.clearEvents()
        event.waitKeys(keyList=['space', 'escape'])
        instruction_title = self.create_text_stim(instruction_title_text1,bold=True, pos=(0,450),style='title')
        instruction_title.draw()
        instruction_text = self.create_text_stim(instruction_body_text1, pos=(0,0),style='0.03')
        instruction_text.draw()
        self.win.flip()
        event.waitKeys(keyList=['space', 'escape'])
        
        # 实验主循环：固定3个blocks
        while self.current_block <= total_blocks:
            # 从试次池获取当前块的试次
            trials = self.generate_trials_for_block(self.current_block)
            
            print(f"开始 Block {self.current_block}/{total_blocks}，共 {len(trials)} 个试次")
            
            # 显示block开始信息
            if self.current_block > 1:
                block_start_text = f"""
即将开始Block {self.current_block}/{total_blocks}，共 {len(trials)} 个试次

准备好后，按空格键继续..."""
                block_start_stim = self.create_text_stim(block_start_text, style='body')
                block_start_stim.draw()
                self.win.flip()
                event.waitKeys(keyList=['space'])
            
            # 执行当前block的所有试次
            trials_in_block = 0
            for trial in trials:
                result = self.run_trial(trial)
                self.save_trial_data(result)
                self.current_trial += 1
                trials_in_block += 1
                self.save_checkpoint()
                
                # 显示进度（每10个试次）
                if trials_in_block % 10 == 0:
                    if len(self.distance_accuracy_history) > 0 or len(self.vector_accuracy_history) > 0:
                        distance_acc = sum(self.distance_accuracy_history) / len(self.distance_accuracy_history) if len(self.distance_accuracy_history) > 0 else 0
                        vector_acc = sum(self.vector_accuracy_history) / len(self.vector_accuracy_history) if len(self.vector_accuracy_history) > 0 else 0
                        max_acc = max(distance_acc, vector_acc)
                        print(f"  Trial {trials_in_block}/{len(trials)}: 当前最高准确率 {max_acc:.3f} (距离: {distance_acc:.3f}, 向量: {vector_acc:.3f})")
            
            # Block完成总结
            self.print_block_summary(self.current_block)
            
            # 显示block完成信息（除了最后一个block）
            if self.current_block < total_blocks:
                distance_acc = sum(self.distance_accuracy_history) / len(self.distance_accuracy_history) if len(self.distance_accuracy_history) > 0 else 0
                vector_acc = sum(self.vector_accuracy_history) / len(self.vector_accuracy_history) if len(self.vector_accuracy_history) > 0 else 0
                max_acc = max(distance_acc, vector_acc)
                
                block_text = f"Block {self.current_block}/{total_blocks} 完成"
                progress_text = f"""
当前最高准确率: {max_acc:.3f}

距离任务: {distance_acc:.3f}

向量任务: {vector_acc:.3f}

休息一下，准备好后，按空格键继续..."""
                
                block_stim = self.create_text_stim(block_text, style='title',pos=(0,200))
                progress_stim = self.create_text_stim(progress_text, style='body',pos=(0,-50))
                block_stim.draw()
                progress_stim.draw()
                self.win.flip()
                event.waitKeys(keyList=['space'])
            
            self.current_block += 1
        
        # 实验完成
        final_distance_acc = sum(self.distance_accuracy_history) / len(self.distance_accuracy_history) if len(self.distance_accuracy_history) > 0 else 0
        final_vector_acc = sum(self.vector_accuracy_history) / len(self.vector_accuracy_history) if len(self.vector_accuracy_history) > 0 else 0
        final_max_acc = max(final_distance_acc, final_vector_acc)
        
        completion_text = f""" 实验结束！"""
        accuracy_text = f"""
最终最高准确率: {final_max_acc:.3f}

距离任务准确率: {final_distance_acc:.3f}

向量任务准确率: {final_vector_acc:.3f}

总评分次数: {len(self.distance_accuracy_history) + len(self.vector_accuracy_history)}

总试次数: {len(self.completed_trials)}

感谢参与！"""
                
        final_stim = self.create_text_stim(completion_text, pos=(0,400),style='title',bold=True)
        final_stim.draw()
        accuracy_stim = self.create_text_stim(accuracy_text, style='body',pos=(0,-50))
        accuracy_stim.draw()
        self.win.flip()
        event.waitKeys(keyList=['space'])
        return

    def cleanup(self):
        """清理资源"""
        if self.win:
            self.win.close()
        core.quit()

    def run(self):
        """主运行函数"""
        try:
            self.load_config()
            self.get_subject_info()
            self.load_stim_table()
            self.setup_experiment()
            self.load_stimuli()
            self.load_checkpoint()
            self.run_experiment()
        except Exception as e:
            print(f"实验出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Phase C: 评分正式阶段')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--subject', type=str, help='被试ID')
    parser.add_argument('--resume', action='store_true', help='从断点继续')
    parser.add_argument('--allow-init', action='store_true', help='允许初始化新被试')
    
    args = parser.parse_args()
    
    experiment = PhaseCExperiment(
        config_path=args.config,
        subject_id=args.subject,
        resume=args.resume,
        allow_init=args.allow_init
    )
    
    experiment.run()


if __name__ == '__main__':
    main()
