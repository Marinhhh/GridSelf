#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase D: 拖拽重建阶段
被试将所有面孔（包括自我）拖放到2D空间中重建社会等级结构
自我面孔固定在原点(0,0)且不可拖拽
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
from scipy.spatial.distance import pdist
from scipy.spatial import procrustes
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# PsychoPy imports
from psychopy import prefs
prefs.general['font'] = ['SimHei', 'Arial', 'Noto Sans CJK SC', 'Microsoft YaHei']
from psychopy import visual, core, event, gui, data, monitors
from psychopy.hardware import keyboard


class PhaseDExperiment:
    def __init__(self, config_path=None, subject_id=None, resume=False, allow_init=False):
        # 获取脚本所在目录，确保路径相对于脚本位置
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = config_path or os.path.join(self.script_dir, "config", "experiment.yaml")
        self.subject_id = subject_id
        self.resume = resume
        self.allow_init = allow_init
        self.phase = "D"
        
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
        
        # 拖拽界面元素
        self.face_stims = {}  # 所有面孔刺激对象
        self.face_positions = {}  # 当前位置
        self.face_drag_counts = {}  # 拖拽次数
        self.face_drag_times = {}  # 拖拽时间记录
        self.dragging_stim = None
        self.drag_offset = (0, 0)
        
        # 网格和画布
        self.grid_lines = []
        self.canvas_bounds = None
        
        # 实验状态
        self.experiment_start_time = None
        self.completed = False
        
        # 随机数生成器
        self.rng = None
        
        # 拖拽与布局
        self.tiles = []           # 左侧/网格中的可拖拽项
        self.dragging = None      # 当前被拖拽 tile
        self.drag_offset = (0, 0) # 鼠标在 tile 内偏移
        self.grid = None          # 网格对象（含 cells）
        self.placed_map = {}      # cell_idx -> tile
        self.interactions_log = []  # 操作日志（pick/release/snap/return）
        
        # 统一配置缓存
        self._cfg = None
        
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
            'Phase': 'D',
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
            title='被试信息 - Phase D',
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

    def _load_phase_d_config(self):
        """加载并统一 Phase D 配置，提供向下兼容映射"""
        phase_d_cfg = self.config.get('phase_d', {})
        slider_cfg = self.config.get('slider', {})
        dimension_colors = self.config.get('dimension_colors', {})
        
        # 向下兼容映射
        def get_with_fallback(key_path, default=None):
            """支持点号分隔的键路径，如 'grid.n'"""
            keys = key_path.split('.')
            value = phase_d_cfg
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        
        # 兼容旧字段
        old_grid_n = phase_d_cfg.get('grid_n', 5)
        old_canvas = phase_d_cfg.get('canvas', {})
        old_self_marker_color = old_canvas.get('self_marker_color', [0.8, 0.8, 0.0])
        
        # 构建统一配置
        self._cfg = type('PhaseDConfig', (), {
            # 基本设置
            'timeout_s': phase_d_cfg.get('timeout_s', 1800),
            'include_self': phase_d_cfg.get('include_self', True),
            'can_drag_self': phase_d_cfg.get('can_drag_self', True),
            
            # 左侧平铺区
            'left_panel': type('LeftPanel', (), {
                'width_frac': get_with_fallback('left_panel.width_frac', 0.35),
                'hpad_px': get_with_fallback('left_panel.hpad_px', 24),
                'top_pad_px': get_with_fallback('left_panel.top_pad_px', 24),
                'vgap_px': get_with_fallback('left_panel.vgap_px', 16),
            })(),
            
            # 头像 tile
            'tile': type('Tile', (), {
                'max_width_px': get_with_fallback('tile.max_width_px', old_canvas.get('face_thumb_size', [200, 200])[0] if isinstance(old_canvas.get('face_thumb_size'), list) else 200),
                'border_px': get_with_fallback('tile.border_px', 2),
                'preload_thumb_px': get_with_fallback('tile.preload_thumb_px', 160),
            })(),
            
            # 网格
            'grid': type('Grid', (), {
                'n': get_with_fallback('grid.n', old_grid_n),
                'stroke_px': get_with_fallback('grid.stroke_px', 30),
                'gap_px': get_with_fallback('grid.gap_px', 2),
                'color': get_with_fallback('grid.color', 'white'),
                'show_cells': get_with_fallback('grid.show_cells', True),
                'center_on_screen': get_with_fallback('grid.center_on_screen', True),
                'x_offset_px': get_with_fallback('grid.x_offset_px', 0),
                'y_offset_px': get_with_fallback('grid.y_offset_px', 0),
            })(),
            
            # 坐标轴
            'axes': type('Axes', (), {
                'show': get_with_fallback('axes.show', False),
                'range': get_with_fallback('axes.range', slider_cfg.get('component_ticks', [-9, -6, -3, 0, 3, 6, 9])),
                'stroke_px': get_with_fallback('axes.stroke_px', None),
                'color': get_with_fallback('axes.color', None),
            })(),
            
            # 标签
            'labels': type('Labels', (), {
                'show': get_with_fallback('labels.show', True),
                'x': get_with_fallback('labels.x', '能力'),
                'y': get_with_fallback('labels.y', '受欢迎度'),
                'xmin': get_with_fallback('labels.xmin', str(int(slider_cfg.get('component_ticks', [-2, 0, 2])[0]))),
                'xmax': get_with_fallback('labels.xmax', str(int(slider_cfg.get('component_ticks', [-2, 0, 2])[-1]))),
                'ymin': get_with_fallback('labels.ymin', str(int(slider_cfg.get('component_ticks', [-2, 0, 2])[0]))),
                'ymax': get_with_fallback('labels.ymax', str(int(slider_cfg.get('component_ticks', [-2, 0, 2])[-1]))),
                'color': get_with_fallback('labels.color', 'white'),
                'offsets_px': type('Offsets', (), {
                    'x': get_with_fallback('labels.offsets_px.x', { 'dx': 0, 'dy': 30 }),
                    'y': get_with_fallback('labels.offsets_px.y', { 'dx': -40, 'dy': 0 }),
                    'xmin': get_with_fallback('labels.offsets_px.xmin', { 'dx': 0, 'dy': 30 }),
                    'xmax': get_with_fallback('labels.offsets_px.xmax', { 'dx': 0, 'dy': 30 }),
                    'ymin': get_with_fallback('labels.offsets_px.ymin', { 'dx': -40, 'dy': 0 }),
                    'ymax': get_with_fallback('labels.offsets_px.ymax', { 'dx': -40, 'dy': 0 }),
                })(),
            })(),
            
            # 颜色
            'colors': type('Colors', (), {
                'self_marker': get_with_fallback('colors.self_marker', old_self_marker_color),
            })(),
            
            # 量化参数
            'axis_step': slider_cfg.get('component_step', 1.0),
            'axis_min': None,  # 将在 _build_layout_params 中设置
            'axis_max': None,  # 将在 _build_layout_params 中设置
        })()
        
        # 设置轴范围
        self._cfg.axis_min = float(self._cfg.axes.range[0])
        self._cfg.axis_max = float(self._cfg.axes.range[1])
        
        # 设置轴颜色和线宽（使用回退逻辑）
        if self._cfg.axes.color is None:
            self._cfg.axes.color = self._cfg.grid.color
        if self._cfg.axes.stroke_px is None:
            self._cfg.axes.stroke_px = self._cfg.grid.stroke_px

    def load_stim_table(self):
        """加载被试×刺激表"""
        stim_table_path = os.path.join(
            self.script_dir,
            self.config['paths']['subjects_dir'],
            f"subj_{self.subject_id}_stim_table.csv"
        )
        
        if not os.path.exists(stim_table_path):
            if self.allow_init:
                print("警告: Phase D 通常不应该创建新的刺激表，建议先运行 Phase A")
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
        
        # 使用统一配置
        preload = self._cfg.tile.preload_thumb_px
        thumb_size = (preload, preload)
        self_marker_color = self._cfg.colors.self_marker
        n_faces = self.config['stimuli']['n_faces']

        # 加载他人面孔
        for stim_id in range(1, n_faces + 1):
            face_path = os.path.join(faces_dir, f"P{stim_id}.jpg")
            if not os.path.exists(face_path):
                face_path = os.path.join(faces_dir, f"P{stim_id}.png")
            
            if os.path.exists(face_path):
                self.face_images[stim_id] = visual.ImageStim(
                    win=self.win,
                    image=face_path,
                    size=thumb_size,
                    pos=(0, 0)
                )
            else:
                # 创建占位符
                self.face_images[stim_id] = visual.Rect(
                    win=self.win,
                    width=thumb_size[0], height=thumb_size[1],
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
                    size=thumb_size,
                    pos=(0, 0)
                )
                
                # 添加自我标记
                self.self_marker = visual.Circle(
                    win=self.win,
                    radius=thumb_size[0] * 0.6,
                    lineColor=self_marker_color,
                    lineWidth=3,
                    fillColor=None,
                    pos=(0, 0)
                )
            else:
                print(f"警告: 未找到自我图像 {self_path}")

    def setup_experiment(self):
        """设置实验环境"""
        # 加载统一配置
        self._load_phase_d_config()
        
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

        self.load_stimuli()
        
        # 初始化随机数生成器
        seed_sequence = np.random.SeedSequence([
            self.config['experiment']['seed'],
            hash(self.subject_id) % (2**31),
            hash(f"phase_{self.phase}") % (2**31)
        ])
        self.rng = np.random.default_rng(seed_sequence)
        
        # 初始化新布局
        self._build_layout_params()
        self._make_left_panel_tiles()
        self._make_grid()
        
        print("实验环境设置完成")

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

    def _build_layout_params(self):
        """构建布局参数 - 读取统一配置并计算所有位置/尺寸参数"""
        W, H = self.win.size

        # 左侧平铺区
        self._left_w = int(W * self._cfg.left_panel.width_frac)
        self._right_w = W - self._left_w
        
        self._left_xmin = -W//2
        self._left_xmax = self._left_xmin + self._left_w
        self._right_xmin = self._left_xmax
        self._right_xmax = W//2

        self._left_hpad = self._cfg.left_panel.hpad_px
        self._left_top = H//2 - self._cfg.left_panel.top_pad_px

        # 头像 tile
        self._tile_size = self._cfg.tile.max_width_px
        self._tile_border = self._cfg.tile.border_px
        self._vgap = self._cfg.left_panel.vgap_px

        # 网格
        self._grid_n = self._cfg.grid.n
        self._grid_stroke = self._cfg.grid.stroke_px
        self._grid_gap = self._cfg.grid.gap_px

        # 网格大小
        side = min(self._right_w - 2*self._left_hpad,
        H - 2*self._left_hpad)
        self._grid_side = int(max(100, side))
        self._grid_cell = self._grid_side / self._grid_n
        
        # 网格左上角（用 center-based 坐标求值）
        self._grid_origin = (
            self._right_xmin + (self._right_w - self._grid_side)/2,
            +self._grid_side/2
        )
        
        # 轴范围与刻度量化（使用统一配置）
        self._axis_min = self._cfg.axis_min
        self._axis_max = self._cfg.axis_max
        self._axis_step = self._cfg.axis_step

        # 左侧是否允许拖自我
        self._can_drag_self = self._cfg.can_drag_self

    def _make_left_panel_tiles(self):
        """随机顺序把刺激平铺到左侧：从上到下，满一列后换到下一列。StimID=0 表示 S。"""
        ids = sorted(list(self.face_images.keys()))
        include_self = self._cfg.include_self
        if include_self and self.self_image:
            ids = [0] + ids

        rng = np.random.default_rng(self.config['experiment']['seed'] + 9871)
        rng.shuffle(ids)

        # tile 尺寸不超过格子
        tile_size = min(self._tile_size, int(self._grid_cell - 2*self._grid_gap))
        self._tile_size = tile_size  # 后面 hit test 用

        # 能在一列里放下多少张
        usable_h = self.win.size[1] - 2*self._left_hpad
        per_col = max(1, int((usable_h + self._vgap) // (tile_size + self._vgap)))

        # 多列 x 坐标
        col_x0 = self._left_xmin + self._left_hpad + tile_size/2
        col_w  = tile_size + self._left_hpad
        tiles = []
        col = 0
        row = 0

        for stim_id in ids:
            # 准备图像内容
            if stim_id == 0 and self.self_image:
                img = self.self_image.image
            else:
                stim = self.face_images.get(stim_id)
                if stim is None:
                    continue
                img = stim.image

            x = col_x0 + col * col_w
            y_top = self.win.size[1]/2 - self._left_hpad
            y = y_top - (row * (tile_size + self._vgap)) - tile_size/2

            tile = visual.ImageStim(
                win=self.win,
                image=img,
                size=(tile_size, tile_size),
                pos=(x, y)
            )
            tiles.append({
                'stim_id': stim_id,
                'stim': tile,
                'home_pos': (x, y),
                'cur_pos': (x, y),
                'placed_cell': None,
                'is_self': (stim_id == 0)
            })

            row += 1
            if row >= per_col:
                row = 0
                col += 1

        self.tiles = tiles

    def _make_grid(self):
        """生成右侧网格几何（cells）以及可选的可视化（仅轴线 + 标签） - 使用统一配置"""
        n = self._cfg.grid.n
        cell = self._grid_cell
        side = self._grid_side

        # 计算网格原点位置
        if self._cfg.grid.center_on_screen:
            x0 = -side / 2 + self._cfg.grid.x_offset_px
            y0 = -side / 2 + self._cfg.grid.y_offset_px
        else:
            x0 = self._right_xmin + (self._right_w - side) / 2 + self._cfg.grid.x_offset_px
            y0 = -side/2 + self._cfg.grid.y_offset_px

        # cells：用于碰撞/吸附/记录（不可见）
        cells = []
        for r in range(n):
            for c in range(n):
                cx = x0 + (c + 0.5) * cell
                cy = y0 + (r + 0.5) * cell
                xmin = cx - cell/2 + self._grid_gap
                xmax = cx + cell/2 - self._grid_gap
                ymin = cy - cell/2 + self._grid_gap
                ymax = cy + cell/2 - self._grid_gap
                cells.append({
                    'idx': r * n + c,
                    'row': r,
                    'col': c,
                    'center': (cx, cy),
                    'bounds': (xmin, xmax, ymin, ymax),
                    'occupied_by': None
                })

        # 可选：是否画小格线
        lines = []
        if self._cfg.grid.show_cells:
            for i in range(n + 1):
                y = y0 + i * cell
                lines.append(visual.Line(self.win, start=(x0, y), end=(x0 + side, y),
                                         lineWidth=self._cfg.grid.stroke_px, lineColor=self._cfg.grid.color))
                x = x0 + i * cell
                lines.append(visual.Line(self.win, start=(x, y0), end=(x, y0 + side),
                                         lineWidth=self._cfg.grid.stroke_px, lineColor=self._cfg.grid.color))

        # 轴线（根据配置决定是否绘制）
        x_axis = None
        y_axis = None
        if self._cfg.axes.show:
            cx_mid = x0 + side/2
            cy_mid = y0 + side/2
            x_axis = visual.Line(self.win, start=(x0, cy_mid), end=(x0 + side, cy_mid),
                                 lineWidth=self._cfg.axes.stroke_px, lineColor=self._cfg.axes.color)
            y_axis = visual.Line(self.win, start=(cx_mid, y0), end=(cx_mid, y0 + side),
                                 lineWidth=self._cfg.axes.stroke_px, lineColor=self._cfg.axes.color)

        # 轴标签（根据配置决定是否绘制）
        axis_labels = []
        if self._cfg.labels.show:
            cx_mid = x0 + side/2
            cy_mid = y0 + side/2
            
            # 获取维度颜色
            dimension_colors = self.config.get('dimension_colors', {})
            label_x_color = dimension_colors.get('dim1_hint', "orange")
            label_y_color = dimension_colors.get('dim2_hint', "dodgerblue")
            
            label_x_text =f"""能力"""
            label_y_text =f"""
受
欢
迎
度"""
            
            # 轴标签
            axis_labels = [
                self.create_text_stim(label_x_text, style='small', bold=True, italic=True,
                                     pos=(cx_mid+self._cfg.labels.offsets_px.x['dx'], 
                                     y0 + self._cfg.labels.offsets_px.x['dy']), color=label_x_color),
                self.create_text_stim(label_y_text, style='small', bold=True, italic=True,
                                     pos=(x0 + self._cfg.labels.offsets_px.y['dx'], 
                                     cy_mid+self._cfg.labels.offsets_px.y['dy']), color=label_y_color),

                self.create_text_stim(self._cfg.labels.xmin, style='small', 
                                     pos=(x0+self._cfg.labels.offsets_px.xmin['dx'], 
                                     y0 + self._cfg.labels.offsets_px.xmin['dy']), color=label_x_color),
                self.create_text_stim(self._cfg.labels.xmax, style='small', 
                                     pos=(x0 + side+self._cfg.labels.offsets_px.xmax['dx'], 
                                     y0 + self._cfg.labels.offsets_px.xmax['dy']), color=label_x_color),
                
                self.create_text_stim(self._cfg.labels.ymin, style='small', 
                                     pos=(x0 + self._cfg.labels.offsets_px.ymin['dx'], 
                                     y0+self._cfg.labels.offsets_px.ymin['dy']), color=label_y_color),
                self.create_text_stim(self._cfg.labels.ymax, style='small', 
                                     pos=(x0 + self._cfg.labels.offsets_px.ymax['dx'], 
                                     y0 + side+self._cfg.labels.offsets_px.ymax['dy']), color=label_y_color),
            ]

        self.grid = {
            'cells': cells,
            'lines': lines,          # 仅在 show_cells=True 时有
            'x_axis': x_axis,
            'y_axis': y_axis,
            'axis_labels': axis_labels,
            'x0': x0, 'y0': y0, 'side': side, 'cell': cell
        }
        self.placed_map = {}

    def _pix_to_axis(self, x_pix, y_pix):
        """像素坐标 → 连续实验坐标（Com/Pop） - 使用统一配置的轴范围"""
        g = self.grid
        x0, y0, side = g['x0'], g['y0'], g['side']
        # 将像素相对网格左下角归一化到 [0,1]
        nx = (x_pix - x0) / side
        ny = (y_pix - y0) / side
        # 再映射到 [-0.5, 0.5]，最后到 [axis_min, axis_max]
        vx = (nx - 0.5) * 2.0  # -1..1
        vy = (ny - 0.5) * 2.0
        half = (self._cfg.axis_max - self._cfg.axis_min) / 2.0
        com = vx * half  # X 轴：能力
        pop = vy * half  # Y 轴：受欢迎度
        return com, pop

    def _quantize_axis(self, v):
        """把连续实验坐标量化到刻度 - 使用统一配置的步长"""
        step = self._cfg.axis_step
        q = round(v / step) * step
        # 夹紧到范围
        q = max(self._cfg.axis_min, min(self._cfg.axis_max, q))
        # 避免 -0.0
        if abs(q) < 1e-9:
            q = 0.0
        return float(q)

    def _tile_at(self, mx, my):
        """获取指定位置的 tile"""
        for T in reversed(self.tiles):
            x, y = T['cur_pos']; s = self._tile_size/2
            if (x - s) <= mx <= (x + s) and (y - s) <= my <= (y + s):
                return T
        return None

    def _cell_at(self, mx, my):
        """获取指定位置的 cell"""
        for cell in self.grid['cells']:
            xmin, xmax, ymin, ymax = cell['bounds']
            if xmin <= mx <= xmax and ymin <= my <= ymax:
                return cell
        return None

    def _snap_to_cell(self, tile, cell):
        """将 tile 吸附到 cell"""
        if cell['occupied_by'] is not None and cell['occupied_by'] != tile['stim_id']:
            return False
        # 清除旧占位
        if tile['placed_cell'] is not None:
            old = tile['placed_cell']
            self.grid['cells'][old]['occupied_by'] = None
            self.placed_map.pop(old, None)
        # 设新占位
        idx = cell['idx']
        cell['occupied_by'] = tile['stim_id']
        self.placed_map[idx] = tile
        tile['placed_cell'] = idx
        tile['cur_pos'] = cell['center']
        return True

    def _release_tile(self, tile, mx, my):
        """释放 tile"""
        # 若不允许拖自我，直接退回
        if tile.get('is_self', False) and not self._can_drag_self:
            tile['cur_pos'] = tile['home_pos']
            self.interactions_log.append({'t': core.getTime(), 'event': 'return_self', 'stim_id': tile['stim_id']})
            return
            
        cell = self._cell_at(mx, my)
        if cell and (cell['occupied_by'] in (None, tile['stim_id'])):
            if self._snap_to_cell(tile, cell):
                self.interactions_log.append({'t': core.getTime(), 'event': 'snap', 'stim_id': tile['stim_id'], 'cell_idx': cell['idx']})
                return
        # 否则退回
        if tile['placed_cell'] is not None:
            prev = tile['placed_cell']
            self.grid['cells'][prev]['occupied_by'] = None
            self.placed_map.pop(prev, None)
            tile['placed_cell'] = None
        tile['cur_pos'] = tile['home_pos']
        self.interactions_log.append({'t': core.getTime(), 'event': 'return', 'stim_id': tile['stim_id']})

    def _draw_frame(self, remain_s, all_placed):
        # 左侧 tiles
        for T in self.tiles:
            T['stim'].pos = T['cur_pos']
            T['stim'].draw()

        # 右侧可选小格线
        for ln in self.grid['lines']:
            ln.draw()

        # 画轴线 + 标签（根据配置决定是否绘制）
        if self.grid['x_axis'] is not None:
            self.grid['x_axis'].draw()
        if self.grid['y_axis'] is not None:
            self.grid['y_axis'].draw()
        for lab in self.grid['axis_labels']:
            lab.draw()

        # 状态条
        placed = sum(1 for T in self.tiles if T['placed_cell'] is not None)
        total = len(self.tiles)
        hint_text = f"""
已放置 {placed}/{total},剩余时间: {int(remain_s)//60} 分钟 {int(remain_s)%60} 秒"""
        hint_color = 'white'
        if all_placed:
            hint_text += f"""

确认无误后按Enter提交"""
            hint_color = 'green'
        # 放到右侧下方，避免挡住左侧

        x0, y0 = self.grid['x0'], self.grid['y0']
        
        tip_text = self.create_text_stim(hint_text, style='small',bold=True,
                                    pos=(x0 + self._cfg.labels.offsets_px.ymin['dx'] - 300, 
                                    y0+self._cfg.labels.offsets_px.ymin['dy']), color=hint_color) 
        tip_text.draw()

    def _save_reconstruction_outputs(self):
        out_dir = os.path.join(self.script_dir, self.config['paths']['data_dir'], 'D')
        os.makedirs(out_dir, exist_ok=True)

        # 为取 TrueX/TrueY 做个查表（StimID=0 为自我）
        true_map = {int(row['StimID']): (float(row['com']), float(row['pop']))
                    for _, row in self.stim_table.iterrows()}
        true_map[0] = (0.0, 0.0)

        tol = float(self.config.get('tolerance', {}).get('component_abs', 0.5))

        rows = []
        for T in self.tiles:
            cell_idx = T['placed_cell']
            cx, cy = T['cur_pos']
            # 连续坐标
            rx, ry = self._pix_to_axis(cx, cy)
            # 量化坐标
            qx = self._quantize_axis(rx)
            qy = self._quantize_axis(ry)
            # 真值
            tru = true_map.get(int(T['stim_id']), (None, None))
            tx, ty = tru if tru is not None else (None, None)
            # 误差与正确性
            abs_ex = "" if tx is None else abs(qx - tx)
            abs_ey = "" if ty is None else abs(qy - ty)
            is_correct = ""
            if (tx is not None) and (ty is not None):
                is_correct = int((abs(qx - tx) <= tol) and (abs(qy - ty) <= tol))

            row = {
                'SubID': self.subject_id,
                'SessionID': self.subject_profile.get('SessionID', ''),
                'StimID': T['stim_id'],
                'IsSelf': int(T.get('is_self', False)),
                'Placed': int(cell_idx is not None),
                'CellIdx': -1 if cell_idx is None else cell_idx,
                'Row': '' if cell_idx is None else self.grid['cells'][cell_idx]['row'],
                'Col': '' if cell_idx is None else self.grid['cells'][cell_idx]['col'],
                'CenterX_pix': cx,
                'CenterY_pix': cy,
                'ReconRawX': rx,
                'ReconRawY': ry,
                'ReconX': qx,
                'ReconY': qy,
                'TrueX': "" if tx is None else tx,
                'TrueY': "" if ty is None else ty,
                'AbsErrX': abs_ex,
                'AbsErrY': abs_ey,
                'IsCorrect': is_correct,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, f"subj_{self.subject_id}_reconstruction.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"重建明细已保存: {csv_path}")

        summary = pd.DataFrame([{
            'SubID': self.subject_id,
            'SessionID': self.subject_profile.get('SessionID', ''),
            'N_Placed': int(df['Placed'].sum()),
            'N_Total': int(len(df)),
            'All_Placed': int(df['Placed'].sum() == len(df)),
            'Correct_Rate': "" if 'IsCorrect' not in df else float(df['IsCorrect'].replace('', np.nan).dropna().mean() if (df['IsCorrect']!='').any() else np.nan),
            'Timestamp': datetime.now().isoformat()
        }])
        sum_path = os.path.join(out_dir, f"subj_{self.subject_id}_reconstruction_summary.csv")
        if not os.path.exists(sum_path):
            summary.to_csv(sum_path, index=False, encoding='utf-8-sig')
        else:
            summary.to_csv(sum_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"重建汇总已保存: {sum_path}")

        # 截图
        png_path = os.path.join(out_dir, f"subj_{self.subject_id}_reconstruction.png")
        self._draw_frame(remain_s=0, all_placed=all(t['placed_cell'] is not None for t in self.tiles))
        self.win.flip()
        core.wait(0.03)
        try:
            self.win.movieFrames = []
        except Exception:
            pass
        self.win.getMovieFrame(buffer='front')
        self.win.saveMovieFrames(png_path)
        print(f"截图已保存: {png_path}")

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


    def run_experiment(self):
        """运行拖拽重建实验"""
        print(f"\n开始 Phase D 实验 - 被试: {self.subject_id}")
        print("=" * 50)
        
        # 显示指导语
        instruction = f"""Phase D: 2D空间重建任务

请将所有面孔拖放到右侧网格中，重建他们的社会等级结构。

操作说明：
- 左侧：所有面孔（包括你自己）随机排列

- 右侧：{self._cfg.grid.n}×{self._cfg.grid.n} 网格
- 用鼠标拖拽左侧面孔到右侧网格中
- 拖拽到网格单元格中心会自动吸附
- 冲突时不允许覆盖，会退回原位
- 考虑能力和受欢迎度两个维度

完成条件：
- 全部放好后按 Enter 确认
- 或超时 {self._cfg.timeout_s} 秒

按空格键开始..."""
        
        instr_text = self.create_text_stim(instruction, style='body')
        instr_text.draw()
        self.win.flip()
        
        # 清空事件缓冲区
        event.clearEvents()
        event.waitKeys(keyList=['space', 'escape'])
        
        # 开始实验
        start_t = core.getTime()
        deadline = start_t + float(self._cfg.timeout_s)
        early_confirm = False

        self.kb.clearEvents()
        self.mouse.clickReset()
        self.dragging = None
        self.drag_offset = (0, 0)

        while True:
            now = core.getTime()
            remain = max(0.0, deadline - now)

            all_placed = all(T['placed_cell'] is not None for T in self.tiles)
            keys = event.getKeys()
            if remain <= 0:
                break
            if all_placed and ('return' in keys or 'enter' in keys):
                early_confirm = True
                break
            if 'escape' in keys:
                break

            # 鼠标事件
            pressed = self.mouse.getPressed()[0]
            mx, my = self.mouse.getPos()

            if pressed and self.dragging is None:
                T = self._tile_at(mx, my)
                if T is not None:
                    self.dragging = T
                    tx, ty = T['cur_pos']
                    self.drag_offset = (mx - tx, my - ty)
                    self.interactions_log.append({'t': now, 'event': 'pick', 'stim_id': T['stim_id']})

            if pressed and self.dragging is not None:
                ox, oy = self.drag_offset
                self.dragging['cur_pos'] = (mx - ox, my - oy)

            if (not pressed) and self.dragging is not None:
                self._release_tile(self.dragging, mx, my)
                self.interactions_log.append({'t': now, 'event': 'release', 'stim_id': self.dragging['stim_id']})
                self.dragging = None

            # 绘制一帧
            self._draw_frame(remain, all_placed)
            self.win.flip()
            core.wait(0.01)
        
        # 保存结果
        self._save_reconstruction_outputs()
        
        # 显示结束语
        end_text = f"""阶段 D 结束。
感谢参与！

按任意键退出。"""
        end_stim = self.create_text_stim(end_text, style='body')
        end_stim.draw()
        self.win.flip()
        event.waitKeys()


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
            self.run_experiment()
        except Exception as e:
            print(f"实验出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Phase D: 拖拽重建阶段')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--subject', type=str, help='被试ID')
    parser.add_argument('--resume', action='store_true', help='从断点继续（Phase D不支持断点）')
    parser.add_argument('--allow-init', action='store_true', help='允许初始化新被试')
    
    args = parser.parse_args()
    
    experiment = PhaseDExperiment(
        config_path=args.config,
        subject_id=args.subject,
        resume=args.resume,
        allow_init=args.allow_init
    )
    
    experiment.run()


if __name__ == '__main__':
    main()
