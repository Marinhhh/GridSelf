#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase A: 两日训练阶段 - 学习、测试、交错
包含学习阶段（有反馈）、测试阶段（无反馈）、交错测试（定时序列）
直到总体准确率达到90%为止
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
from PIL import Image, ImageDraw, ImageFont
import random
import math

# PsychoPy imports
from psychopy import visual, core, event, gui, data, sound, monitors
from psychopy.hardware import keyboard
from psychopy import prefs
prefs.general['font'] = ['SimHei', 'Arial', 'Noto Sans CJK SC', 'Microsoft YaHei']


class PhaseAExperiment:
    def __init__(self, config_path=None, subject_id=None, resume=False, allow_init=False, run_day=None):
        # 获取脚本所在目录,确保路径相对于脚本位置
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = config_path or os.path.join(self.script_dir, "config", "experiment.yaml")
        self.subject_id = subject_id
        self.resume = resume
        self.allow_init = allow_init
        self.phase = "A"
        self.run_day = run_day  # 从命令行参数获取,可能为None
        
        # 初始化变量
        self.config = None
        self.config_checksum = None
        self.subject_profile = None
        self.stim_table = None
        self.stim_table_checksum = None
        self.win = None
        self.kb = None
        self.face_images = {}
        self.self_image = None
        
        # 实验状态（将在GUI后设置）
        self.current_day = None  # 将在get_subject_info后设置
        self.current_block = 1
        self.current_trial = 1
        self.completed_trials = []
        self.accuracy_history = []
        self.test_accuracy_history = []  # 仅测试阶段的准确率
        self.interleaved_accuracy_history = []  # 交错阶段的准确率
        
        # 随机数生成器
        self.rng = None
        
    def load_config(self):
        """加载配置文件,如不存在则创建默认模板"""
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
                    sys.exit("用户取消,程序退出")
                self.config_path = config_file
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            self.config = yaml.safe_load(config_content)
            # 计算配置文件校验和
            self.config_checksum = hashlib.sha256(config_content.encode()).hexdigest()[:16]
    
    def _create_default_config(self):
        """创建默认配置文件模板"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        default_config = """meta:
  schema_version: "cfg-1.0"
  owner: "YourLab"

experiment:
  name: "SelfReferenceGridTask"
  seed: 20250927
  fullscreen: true
  window_size: [1920, 1080]
  fps_hard_lock: false

paths:
  stimuli_faces_dir: "stimuli/faces"
  stimuli_self_dir: "stimuli/self"
  data_dir: "Data"
  subjects_dir: "Data/subjects"
  checkpoints_dir: "Data/checkpoints"

monitor:
  name: "lab_default"
  size_pix: [1920, 1080]
  width_cm: 34.0
  distance_cm: 60.0
  screen: 0

window_units: "pix"

stimuli:
  n_faces: 80
  coord_limit: 9
  exclude_origin: true
  assign_coords_if_missing: true
  angle_bins: 12

self_stim:
  required: true
  allow_in_phase_a: true
  allow_in_phase_b: true
  allow_in_phase_c: true
  allow_in_phase_d: true
  quota_per_block_a_learn: 4
  quota_per_block_a_test: 4
  quota_per_block_a_inter: 4
  quota_per_block_b: 4
  quota_per_block_c: 4
  treat_self_as_own_bin: true
  render_overlay_ring: true

tolerance:
  distance_abs: 0.5

phase_a:
  learn: { fixation_s: 0.8, stim_window_s: 2.5, response_deadline_s: 2.5, feedback_s: 0.5, iti_s_uniform: [1.0, 3.0] }
  test:  { fixation_s: 0.8, stim_window_s: 2.5, response_deadline_s: 2.5, feedback_s: 0.0, iti_s_uniform: [1.0, 3.0] }
  interleaved_test:
    dim_cue_s: 1.0
    fixation_white_s: 1.5
    p1_s: 2.0
    fixation_purple_uniform_s: [4.0, 10.0]
    p2_s: 2.0
    fixation_green_uniform_s: [4.0, 10.0]
    next_dim_cue_s: 1.0
  criteria:
    accuracy_target: 0.90
    policy: "until_threshold"
    trials_batch_size: 48

dimension_colors:
  dim1_hint: "orange"
  dim2_hint: "dodgerblue"
"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write(default_config)
        print(f"已创建默认配置文件: {self.config_path}")

    def get_subject_info(self):
        """获取被试信息GUI"""
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
            'SubjectID': self.subject_id or '',
            'Name': '',
            'Sex': ['Male', 'Female', 'Other'],
            'Age': 25,
            'Handedness': ['Right', 'Left', 'Both'],
            'Notes': '',
            'Phase': 'A',
            'RunDay': self.run_day or ['Day1', 'Day2'],  # 预填命令行参数或显示选项
            'SelfImageFile': self_images
        }
        
        dlg = gui.DlgFromDict(
            dictionary=dlg_data,
            title='被试信息 - Phase A',
            fixed=['Phase'],
            order=['SubjectID', 'Name', 'Sex', 'Age', 'Handedness', 'RunDay', 'SelfImageFile', 'Notes']
        )
        
        if not dlg.OK:
            sys.exit("用户取消,程序退出")
        
        # 验证SubjectID
        import re
        if not re.match(r'^[A-Za-z0-9_-]+$', dlg_data['SubjectID']):
            messagebox.showerror("错误", "SubjectID只能包含字母、数字、下划线和横线")
            sys.exit(1)
        
        self.subject_id = dlg_data['SubjectID']
        self.run_day = dlg_data['RunDay']
        
        # 设置current_day（在GUI后立即设置）
        self.current_day = 1 if self.run_day == 'Day1' else 2
        
        # 生成SessionID
        session_id = f"{self.subject_id}_{self.phase}_{self.run_day}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # 计算NameHash和SelfImageSHA256
        name_hash = hashlib.sha256(dlg_data['Name'].encode()).hexdigest()[:16]
        
        self_image_sha256 = ""
        if dlg_data['SelfImageFile']:
            self_image_path = os.path.join(self_dir, dlg_data['SelfImageFile'])
            if os.path.exists(self_image_path):
                with open(self_image_path, 'rb') as f:
                    self_image_sha256 = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # 初始化随机数生成器（用于维度分配）
        temp_rng = np.random.default_rng(self.config['experiment']['seed'])
        
        # 确定维度分配
        if self.run_day == 'Day1':
            # Day1: 随机分配维度
            day1_dim = temp_rng.choice(['com', 'pop'])
            day2_dim = 'pop' if day1_dim == 'com' else 'com'
        else:  # Day2
            # 从profile读取Day1维度
            profile_path = os.path.join(
                self.script_dir,
                self.config['paths']['subjects_dir'],
                f"subj_{self.subject_id}_profile.json"
            )
            if os.path.exists(profile_path):
                with open(profile_path, 'r', encoding='utf-8') as f:
                    existing_profile = json.load(f)
                day1_dim = existing_profile.get('day1_dim')
                if not day1_dim:
                    messagebox.showerror("错误", "尚未完成Day1,请先运行Day1")
                    sys.exit(1)
                day2_dim = 'pop' if day1_dim == 'com' else 'com'
            else:
                messagebox.showerror("错误", "未找到被试档案,请先完成Day1")
                sys.exit(1)
        
        # 保存/更新被试档案
        profile_path = os.path.join(
            self.script_dir,
            self.config['paths']['subjects_dir'],
            f"subj_{self.subject_id}_profile.json"
        )
        
        # 加载现有档案或创建新档案
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                existing_profile = json.load(f)
            profile_data = existing_profile.copy()
        else:
            profile_data = {}
        
        # 更新档案
        profile_data.update({
            'SubjectID': self.subject_id,
            'Name': dlg_data['Name'],
            'NameHash': name_hash,
            'Sex': dlg_data['Sex'],
            'Age': dlg_data['Age'],
            'Handedness': dlg_data['Handedness'],
            'Notes': dlg_data['Notes'],
            'SelfImageFile': dlg_data['SelfImageFile'],
            'SelfImageSHA256': self_image_sha256,
            'LastSessionAt': datetime.now().isoformat(),
            'SchemaVersion': self.config['meta']['schema_version'],
            'SessionID': session_id,
            'RunDay': self.run_day,
            'day1_dim': day1_dim,
            'day2_dim': day2_dim
        })
        
        # 如果是首次创建,添加创建时间
        if 'CreatedAt' not in profile_data:
            profile_data['CreatedAt'] = datetime.now().isoformat()
        
        os.makedirs(os.path.dirname(profile_path), exist_ok=True)
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        
        self.subject_profile = profile_data
        print(f"被试档案已保存: {profile_path}")

    def load_or_create_stim_table(self):
        """加载或创建被试×刺激表"""
        stim_table_path = os.path.join(
            self.script_dir,
            self.config['paths']['subjects_dir'],
            f"subj_{self.subject_id}_stim_table.csv"
        )
        
        if os.path.exists(stim_table_path):
            # 加载现有表格
            self.stim_table = pd.read_csv(
                stim_table_path,
                dtype={'StimID': int, 'com': int, 'pop': int}
            )
            
            # 检查是否需要重建（一次性迁移）
            need_rebuild = False
            coord_limit = self.config['stimuli']['coord_limit']
            grid_n = 2 * coord_limit + 1
            expected_n_faces = grid_n * grid_n - 1  # 排除(0,0)
            
            # 检查行数
            if len(self.stim_table) != expected_n_faces:
                print(f"刺激表行数不匹配：期望{expected_n_faces},实际{len(self.stim_table)}")
                need_rebuild = True
            
            # 检查坐标范围
            if not need_rebuild:
                com_out_of_range = ((self.stim_table['com'] < -coord_limit) | 
                                  (self.stim_table['com'] > coord_limit)).any()
                pop_out_of_range = ((self.stim_table['pop'] < -coord_limit) | 
                                  (self.stim_table['pop'] > coord_limit)).any()
                if com_out_of_range or pop_out_of_range:
                    print(f"刺激表坐标超出范围[-{coord_limit}, +{coord_limit}]")
                    need_rebuild = True
            
            # 检查是否包含原点(0,0)
            if not need_rebuild:
                has_origin = ((self.stim_table['com'] == 0) & (self.stim_table['pop'] == 0)).any()
                if has_origin:
                    print("刺激表包含原点(0,0),需要重建")
                    need_rebuild = True
            
            if need_rebuild:
                print("检测到旧格式刺激表,按新规则重建...")
                self._create_stim_table(stim_table_path)
            else:
                # 计算校验和
                table_str = self.stim_table.to_csv(index=False)
                self.stim_table_checksum = hashlib.sha256(table_str.encode()).hexdigest()[:16]
                print(f"已加载被试刺激表: {stim_table_path}")
                
                # 渲染网格图
                self.render_subject_grid_mosaic()
                
                # === cache ids & size (no hard-coded 80) ===
                self.stim_ids = sorted(self.stim_table['StimID'].astype(int).tolist())
                self.n_faces = len(self.stim_ids)
        else:
            # 创建新表格（仅Phase A允许）
            if not self.allow_init and self.phase != 'A':
                raise ValueError(f"Phase {self.phase} 不允许创建新的刺激表,请先运行 Phase A")
            
            print("正在为被试创建新的刺激表...")
            self._create_stim_table(stim_table_path)
    
    def _create_stim_table(self, output_path):
        """创建被试×刺激表（覆盖式分配）"""
        # 初始化随机数生成器
        seed_sequence = np.random.SeedSequence([
            self.config['experiment']['seed'], 
            hash(self.subject_id) % (2**31)
        ])
        rng = np.random.default_rng(seed_sequence)
        
        # 由coord_limit自动计算格点数量
        coord_limit = int(self.config['stimuli']['coord_limit'])
        grid_n = 2 * coord_limit + 1
        n_faces = grid_n * grid_n - 1  # 覆盖YAML,排除(0,0)
        
        # 生成所有 (com, pop) 整数格点,范围 [-coord_limit, +coord_limit]×[-coord_limit, +coord_limit]
        all_coords = [(com, pop)
                      for com in range(-coord_limit, coord_limit + 1)
                      for pop in range(-coord_limit, coord_limit + 1)
                      if not (com == 0 and pop == 0)]
        
        # 验证格点数量
        assert len(all_coords) == n_faces, f"坐标数量异常: {len(all_coords)} vs {n_faces}"
        
        # 用被试种子shuffle格点
        rng.shuffle(all_coords)
        
        # 按 StimID=1..n_faces 顺序一一分配
        stim_data = []
        for i in range(n_faces):
            com, pop = all_coords[i]
            stim_id = i + 1  # 1-n_faces
            
            # 计算角度和距离
            angle_rad = math.atan2(pop, com)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
            
            angle_bin = int(angle_deg // 30)  # 0-11
            distance = math.sqrt(com**2 + pop**2)
            
            stim_data.append({
                'StimID': stim_id,
                'com': com,
                'pop': pop,
                'angle_deg': angle_deg,
                'angle_bin': angle_bin,
                'distance': distance
            })
        
        self.stim_table = pd.DataFrame(stim_data)
        
        # 保存并计算校验和
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.stim_table.to_csv(output_path, index=False)
        
        table_str = self.stim_table.to_csv(index=False)
        self.stim_table_checksum = hashlib.sha256(table_str.encode()).hexdigest()[:16]
        
        print(f"刺激表已创建: {output_path}")
        print(f"角度桶分布: {self.stim_table['angle_bin'].value_counts().sort_index().to_dict()}")
        
        # 渲染网格图
        self.render_subject_grid_mosaic()
        
        # === cache ids & size (no hard-coded 80) ===
        self.stim_ids = sorted(self.stim_table['StimID'].astype(int).tolist())
        self.n_faces = len(self.stim_ids)

    def render_subject_grid_mosaic(self):
        """渲染被试9×9坐标网格图"""
        try:
            out_path = render_subject_grid_mosaic(
                self.subject_profile, 
                self.stim_table, 
                self.config, 
                self.script_dir
            )
            print(f"已生成被试网格图: {out_path}")
        except Exception as e:
            print(f"生成被试网格图失败（不影响实验继续）: {e}")

    def load_stimuli(self):
        """加载刺激图像"""
        faces_dir = os.path.join(self.script_dir, self.config['paths']['stimuli_faces_dir'])
        self_dir = os.path.join(self.script_dir, self.config['paths']['stimuli_self_dir'])
        
        # 布局参数（可调）
        self.face_size_x = 360  # 面孔大小（像素）
        self.face_size_y = 360  # 面孔大小（像素）
        self.left_offset = -350  # 左脸X偏移
        self.right_offset = 350  # 右脸X偏移
        self.top_height = 200  # 上方高度
        self.dim_block_size = 150 # 维度色块大小
        self.dim_block_y = -350  # 维度色块Y位置
        
        # 加载他人面孔
        for stim_id in self.stim_ids:
            face_path = os.path.join(faces_dir, f"P{stim_id}.jpg")
            if not os.path.exists(face_path):
                face_path = os.path.join(faces_dir, f"P{stim_id}.png")
            
            if os.path.exists(face_path):
                self.face_images[stim_id] = visual.ImageStim(
                    win=self.win,
                    image=face_path,
                    size=(self.face_size_x, self.face_size_y),
                    pos=(0, 0)  # 位置在运行时设置
                )
            else:
                # 创建占位符
                self.face_images[stim_id] = visual.Rect(
                    win=self.win,
                    width=self.face_size_x, height=self.face_size_y,
                    fillColor='gray',
                    pos=(0, 0)
                )
                print(f"警告: 未找到 {face_path},使用占位符")
        
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
        
        # 创建反馈方框
        self.feedback_box_left = visual.Rect(
            win=self.win,
            width=self.face_size_x + 20, height=self.face_size_y + 20,
            lineColor='green', lineWidth=5, fillColor=None,
            pos=(self.left_offset, self.top_height)
        )
        self.feedback_box_right = visual.Rect(
            win=self.win,
            width=self.face_size_x + 20, height=self.face_size_y + 20,
            lineColor='green', lineWidth=5, fillColor=None,
            pos=(self.right_offset, self.top_height)
        )
        
        # 创建反馈文字
        self.feedback_text = visual.TextStim(
            win=self.win,
            font='SimHei',
            bold=True,
            text='',
            height=40,
            color='green',
            pos=(0, self.top_height + 150)
        )

    def setup_experiment(self):
        """设置实验环境"""
        # >>> 新增：构建 Monitor,并创建 Window（避免 __blank__ 崩溃）
        mon, screen_idx, size_pix = self._build_monitor()

        # window units: use 'pix' globally to avoid deg conversions for GUI splash
        window_units = self.config.get('window_units', 'pix')
        # 背景色：从experiment.bg_color读取,默认纯黑
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

        event.Mouse(visible=True, win=self.win)  # 显示鼠标,便于交互
        
        # 创建键盘
        self.kb = keyboard.Keyboard()
        
        # 初始化随机数生成器
        seed_sequence = np.random.SeedSequence([
            self.config['experiment']['seed'],
            hash(self.subject_id) % (2**31),
            hash(f"phase_{self.phase}") % (2**31)
        ])
        self.rng = np.random.default_rng(seed_sequence)
        
        # 确定当前运行的维度
        if self.run_day == 'Day1':
            self.current_dim = self.subject_profile['day1_dim']
        else:  # Day2
            self.current_dim = self.subject_profile['day2_dim']
        
        # 初始化候选池
        self.candidate_pools = {}
        self._build_candidate_pools()
        
        # 初始化快速测试相关变量（按维度隔离）
        self.recent_learn_stimuli = {'com': set(), 'pop': set()}  # 最近学习批次的刺激集合
        self.recent_learn_pairs = {'com': set(), 'pop': set()}  # 最近学习批次的无序对集合
        self.fast_test_enabled = self.config.get('dev', {}).get('fast_test', False)
        
        if self.fast_test_enabled:
            print("快速测试模式已启用")
        else:
            print("快速测试模式未启用")
        
        print("实验环境设置完成")

    def _ui_yield_and_focus_guard(self, sleep=0.01):
        """让出时间片,并在可能的情况下调度窗口事件,避免失焦时忙等假死。"""
        core.wait(sleep)
        try:
            # 某些平台上 pyglet/窗口事件调度能改善失焦后的卡顿
            if hasattr(self.win, 'winHandle') and self.win.winHandle:
                self.win.winHandle.dispatch_events()
        except Exception:
            pass

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
        # PsychoPy 不要求必须 saveMon;不落地到本机 Monitor Center 以保持可移植
        # monitors.saveMonitorAttributes(name)  # <- 不要强制保存
        return mon, screen_idx, size_pix

    def _build_candidate_pools(self):
        """构建候选池"""
        print("正在构建候选池...")
        
        # 获取所有刺激坐标
        coords = {}
        for _, row in self.stim_table.iterrows():
            coords[row['StimID']] = (row['com'], row['pop'])
        coords[0] = (0, 0)  # 自我
        
        # 为当前维度构建候选池
        dim = self.current_dim
        self.candidate_pools[dim] = {
            'others': {'delta_1': [], 'delta_ge_1': []},
            'self': {'delta_1': [], 'delta_ge_1': []}
        }
        
        ids = self.stim_ids
        idx_len = len(ids)
        
        # 他人-他人（无序对）
        for a in range(idx_len):
            i = ids[a]
            for b in range(a + 1, idx_len):
                j = ids[b]
                delta = abs(coords[i][0 if dim == 'com' else 1] - coords[j][0 if dim == 'com' else 1])
                if delta == 1:
                    self.candidate_pools[dim]['others']['delta_1'].append((i, j))
                elif delta >= 1:
                    self.candidate_pools[dim]['others']['delta_ge_1'].append((i, j))
        
        # 自我-他人
        for j in ids:
            delta = abs(coords[0][0 if dim == 'com' else 1] - coords[j][0 if dim == 'com' else 1])
            if delta == 1:
                self.candidate_pools[dim]['self']['delta_1'].append((0, j))
            elif delta >= 1:
                self.candidate_pools[dim]['self']['delta_ge_1'].append((0, j))
        
        # 打乱所有池
        for pool_type in ['others', 'self']:
            for delta_type in ['delta_1', 'delta_ge_1']:
                self.rng.shuffle(self.candidate_pools[dim][pool_type][delta_type])
        
        # 打印池构建完成信息
        others_delta1 = len(self.candidate_pools[dim]['others']['delta_1'])
        others_deltage1 = len(self.candidate_pools[dim]['others']['delta_ge_1'])
        self_delta1 = len(self.candidate_pools[dim]['self']['delta_1'])
        self_deltage1 = len(self.candidate_pools[dim]['self']['delta_ge_1'])
        
        print(f"候选池构建完成 - 维度: {dim}")
        print(f"  Δ=1/自我: {self_delta1} ")
        print(f"  Δ=1/他人: {others_delta1} ")
        print(f"  Δ≥1/自我: {self_deltage1}")
        print(f"  Δ≥1/他人: {others_deltage1}")

    def ensure_pool_for_dim(self, dim):
        """确保指定维度的候选池存在且完整,必要时重建"""
        # 检查是否需要构建或重建该维度的池
        need_build = False
        
        if dim not in self.candidate_pools:
            need_build = True
        else:
            # 检查池是否完整（包含所有必需的桶）
            required_keys = ['others', 'self']
            required_delta_keys = ['delta_1', 'delta_ge_1']
            
            for pool_type in required_keys:
                if pool_type not in self.candidate_pools[dim]:
                    need_build = True
                    break
                for delta_type in required_delta_keys:
                    if delta_type not in self.candidate_pools[dim][pool_type]:
                        need_build = True
                        break
                    # 检查池是否为空（可能被消耗完）
                    if len(self.candidate_pools[dim][pool_type][delta_type]) == 0:
                        need_build = True
                        break
        
        if need_build:
            # 保存当前维度,临时切换到目标维度
            original_dim = self.current_dim
            self.current_dim = dim
            
            # 获取所有刺激坐标
            coords = {}
            for _, row in self.stim_table.iterrows():
                coords[row['StimID']] = (row['com'], row['pop'])
            coords[0] = (0, 0)  # 自我
            
            # 构建目标维度的候选池
            self.candidate_pools[dim] = {
                'others': {'delta_1': [], 'delta_ge_1': []},
                'self': {'delta_1': [], 'delta_ge_1': []}
            }
            
            ids = self.stim_ids
            idx_len = len(ids)
            
            # 他人-他人（无序对）
            for a in range(idx_len):
                i = ids[a]
                for b in range(a + 1, idx_len):
                    j = ids[b]
                    delta = abs(coords[i][0 if dim == 'com' else 1] - coords[j][0 if dim == 'com' else 1])
                    if delta == 1:
                        self.candidate_pools[dim]['others']['delta_1'].append((i, j))
                    elif delta >= 1:
                        self.candidate_pools[dim]['others']['delta_ge_1'].append((i, j))
            
            # 自我-他人
            for j in ids:
                delta = abs(coords[0][0 if dim == 'com' else 1] - coords[j][0 if dim == 'com' else 1])
                if delta == 1:
                    self.candidate_pools[dim]['self']['delta_1'].append((0, j))
                elif delta >= 1:
                    self.candidate_pools[dim]['self']['delta_ge_1'].append((0, j))
            
            # 打乱所有池
            for pool_type in ['others', 'self']:
                for delta_type in ['delta_1', 'delta_ge_1']:
                    self.rng.shuffle(self.candidate_pools[dim][pool_type][delta_type])
            
            # 打印池构建完成信息
            others_delta1 = len(self.candidate_pools[dim]['others']['delta_1'])
            others_deltage1 = len(self.candidate_pools[dim]['others']['delta_ge_1'])
            self_delta1 = len(self.candidate_pools[dim]['self']['delta_1'])
            self_deltage1 = len(self.candidate_pools[dim]['self']['delta_ge_1'])
            
            print(f"候选池构建完成 - 维度: {dim}")
            print(f"  Δ=1/自我: {self_delta1} (仅按当前维度计算与自我差1的他人数量)")
            print(f"  Δ=1/他人: {others_delta1} (仅按当前维度计算,任意另一维)")
            print(f"  Δ≥1/自我: {self_deltage1}")
            print(f"  Δ≥1/他人: {others_deltage1}")
            
            # 恢复原始维度
            self.current_dim = original_dim

    def _sample_trials_from_pool(self, trial_type, n_trials):
        """从候选池采样试次"""
        dim = self.current_dim
        
        # 确保当前维度的候选池存在
        self.ensure_pool_for_dim(dim)
        
        trials = []
        
        # 获取自我配额
        if trial_type == 'learn':
            self_quota = self.config['self_stim']['quota_per_block_a_learn']
            pool_key = 'delta_1'
        elif trial_type == 'test':
            self_quota = self.config['self_stim']['quota_per_block_a_test']
            pool_key = 'delta_ge_1'
        else:  # interleaved - 保持原有逻辑
            return self.generate_trials(trial_type, None, n_trials)
        
        # 分层采样
        self_trials = min(self_quota, n_trials)
        others_trials = n_trials - self_trials
        
        # 安全访问自我池
        try:
            self_pool = self.candidate_pools[dim]['self'][pool_key]
        except KeyError as e:
            print(f"警告: 无法访问自我池 {dim}.self.{pool_key}: {e}")
            # 尝试重建池
            self.ensure_pool_for_dim(dim)
            self_pool = self.candidate_pools[dim]['self'][pool_key]
        if len(self_pool) < self_trials:
            print(f"警告: 自我{pool_key}池不足,需要{self_trials},可用{len(self_pool)}")
            self_trials = len(self_pool)
            others_trials = n_trials - self_trials
        
        for _ in range(self_trials):
            if self_pool:
                pair = self_pool.pop(0)
                # 方向随机化
                if self.rng.random() < 0.5:
                    p1, p2 = pair
                else:
                    p2, p1 = pair
                trials.append({
                    'Type': trial_type,
                    'CueDim': dim,
                    'P1': p1,
                    'P2': p2
                })
        
        # 安全访问他人池
        try:
            others_pool = self.candidate_pools[dim]['others'][pool_key]
        except KeyError as e:
            print(f"警告: 无法访问他人池 {dim}.others.{pool_key}: {e}")
            # 尝试重建池
            self.ensure_pool_for_dim(dim)
            others_pool = self.candidate_pools[dim]['others'][pool_key]
        if len(others_pool) < others_trials:
            print(f"警告: 他人{pool_key}池不足,需要{others_trials},可用{len(others_pool)}")
            others_trials = len(others_pool)
        
        for _ in range(others_trials):
            if others_pool:
                pair = others_pool.pop(0)
                # 方向随机化
                if self.rng.random() < 0.5:
                    p1, p2 = pair
                else:
                    p2, p1 = pair
                trials.append({
                    'Type': trial_type,
                    'CueDim': dim,
                    'P1': p1,
                    'P2': p2
                })
        
        return trials

    def _update_recent_learn_pool(self, trials):
        """更新最近学习批次的刺激池（按维度隔离）"""
        if not self.fast_test_enabled:
            return
        
        dim = self.current_dim
        for trial in trials:
            p1, p2 = trial['P1'], trial['P2']
            # 添加刺激ID到当前维度
            self.recent_learn_stimuli[dim].add(p1)
            self.recent_learn_stimuli[dim].add(p2)
            # 添加无序对（确保顺序无关）到当前维度
            pair = tuple(sorted([p1, p2]))
            self.recent_learn_pairs[dim].add(pair)

    def _sample_fast_test_trials(self, n_trials):
        """快速测试采样"""
        if not self.fast_test_enabled:
            return self._sample_trials_from_pool('test', n_trials)
        
        dev_config = self.config.get('dev', {})
        test_delta_exact_1 = dev_config.get('test_delta_exact_1', True)
        test_from_recent_learn = dev_config.get('test_from_recent_learn', True)
        test_only_seen_faces = dev_config.get('test_only_seen_faces', True)
        
        dim = self.current_dim
        trials = []
        
        # 确定Δ约束
        if test_delta_exact_1:
            delta_key = 'delta_1'
        else:
            delta_key = 'delta_ge_1'
        
        # 获取所有刺激坐标
        coords = {}
        for _, row in self.stim_table.iterrows():
            coords[row['StimID']] = (row['com'], row['pop'])
        coords[0] = (0, 0)  # 自我
        
        def calculate_delta(p1, p2):
            return abs(coords[p1][0 if dim == 'com' else 1] - coords[p2][0 if dim == 'com' else 1])
        
        # 优先级1：从最近学习批次的无序对中采样（仅当前维度）
        if test_from_recent_learn and self.recent_learn_pairs[dim]:
            available_pairs = []
            for pair in self.recent_learn_pairs[dim]:
                p1, p2 = pair
                delta = calculate_delta(p1, p2)
                if (test_delta_exact_1 and delta == 1) or (not test_delta_exact_1 and delta >= 1):
                    available_pairs.append(pair)
            
            if available_pairs:
                self.rng.shuffle(available_pairs)
                for pair in available_pairs[:n_trials]:
                    p1, p2 = pair
                    # 方向随机化
                    if self.rng.random() < 0.5:
                        trials.append({'Type': 'test', 'CueDim': dim, 'P1': p1, 'P2': p2})
                    else:
                        trials.append({'Type': 'test', 'CueDim': dim, 'P1': p2, 'P2': p1})
                
                if len(trials) >= n_trials:
                    return trials[:n_trials]
        
        # 优先级2：从最近学习批次的刺激集合中采样（仅当前维度）
        if test_only_seen_faces and self.recent_learn_stimuli[dim] and len(trials) < n_trials:
            remaining = n_trials - len(trials)
            available_pairs = []
            
            stimuli_list = list(self.recent_learn_stimuli[dim])
            for i in range(len(stimuli_list)):
                for j in range(i + 1, len(stimuli_list)):
                    p1, p2 = stimuli_list[i], stimuli_list[j]
                    delta = calculate_delta(p1, p2)
                    if (test_delta_exact_1 and delta == 1) or (not test_delta_exact_1 and delta >= 1):
                        available_pairs.append((p1, p2))
            
            if available_pairs:
                self.rng.shuffle(available_pairs)
                for pair in available_pairs[:remaining]:
                    p1, p2 = pair
                    # 方向随机化
                    if self.rng.random() < 0.5:
                        trials.append({'Type': 'test', 'CueDim': dim, 'P1': p1, 'P2': p2})
                    else:
                        trials.append({'Type': 'test', 'CueDim': dim, 'P1': p2, 'P2': p1})
        
        # 优先级3：回退到全局候选池
        if len(trials) < n_trials:
            remaining = n_trials - len(trials)
            fallback_trials = self._sample_trials_from_pool('test', remaining)
            trials.extend(fallback_trials)
        
        return trials[:n_trials]

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
            
            self.current_day = checkpoint.get('current_day', 1)
            self.current_block = checkpoint.get('current_block', 1)
            self.current_trial = checkpoint.get('current_trial', 1)
            self.completed_trials = checkpoint.get('completed_trials', [])
            self.accuracy_history = checkpoint.get('accuracy_history', [])
            
            # 恢复随机数状态
            if 'rng_state' in checkpoint:
                self.rng.bit_generator.state = checkpoint['rng_state']
            
            print(f"从断点恢复: Day {self.current_day}, Block {self.current_block}, Trial {self.current_trial}")

    def save_checkpoint(self):
        """保存断点文件"""
        checkpoint_path = os.path.join(
            self.script_dir,
            self.config['paths']['checkpoints_dir'],
            f"subj_{self.subject_id}_{self.phase}.json"
        )
        
        checkpoint = {
            'current_day': self.current_day,
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

    def generate_trials(self, trial_type, dimension, n_trials):
        """生成试次"""
        trials = []
        
        if trial_type in ['learn', 'test']:
            # 学习和测试阶段：成对比较
            for _ in range(n_trials):
                # 选择两个刺激
                if self.rng.random() < 0.3:  # 30%概率包含自我
                    p1 = 0  # 自我
                    p2 = int(self.rng.choice(self.stim_ids))
                else:
                    # 选择两个不同的他人刺激
                    selected = self.rng.choice(self.stim_ids, size=2, replace=False)
                    p1, p2 = map(int, selected)
                
                # 随机交换位置
                if self.rng.random() < 0.5:
                    p1, p2 = p2, p1
                
                trial_data = {
                    'Type': trial_type,
                    'CueDim': dimension,
                    'P1': p1,
                    'P2': p2
                }
                trials.append(trial_data)
        
        elif trial_type == 'interleaved':
            # 交错测试：定时序列
            for _ in range(n_trials):
                # 随机选择两个刺激（可能包含自我）
                if self.rng.random() < 0.3:
                    p1 = 0
                    p2 = int(self.rng.choice(self.stim_ids))
                else:
                    selected = self.rng.choice(self.stim_ids, size=2, replace=False)
                    p1, p2 = map(int, selected)
                
                # 随机选择维度
                dim = self.rng.choice(['com', 'pop'])
                
                trial_data = {
                    'Type': 'interleaved',
                    'CueDim': dim,
                    'P1': p1,
                    'P2': p2
                }
                trials.append(trial_data)
        
        return trials

    def run_trial(self, trial_data):
        """执行单个试次"""
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
        
        # 计算正确答案
        dim = trial_data['CueDim']
        if dim == 'com':
            c1, c2 = p1_coords[0], p2_coords[0]
        else:  # pop
            c1, c2 = p1_coords[1], p2_coords[1]
        
        trial_result['C1'] = c1
        trial_result['C2'] = c2
        trial_result['DeltaDim'] = abs(c2 - c1)
        
        correct_choice = 'P1' if c1 > c2 else 'P2'
        trial_result['Correct_Choice'] = correct_choice
        
        # 执行不同类型的试次
        if trial_data['Type'] in ['learn', 'test']:
            result = self.run_simultaneous_trial(trial_data, trial_result)
        else:  # interleaved
            result = self.run_interleaved_trial(trial_data, trial_result)
        
        return result

    def get_stim_coords(self, stim_id):
        """获取刺激坐标"""
        if stim_id == 0:  # 自我
            return (0, 0)
        else:
            stim_info = self.stim_table[self.stim_table['StimID'] == stim_id].iloc[0]
            return (stim_info['com'], stim_info['pop'])

    def run_simultaneous_trial(self, trial_data, trial_result):
        """执行同时呈现试次（学习/测试）"""
        timing = self.config['phase_a'][trial_data['Type']]
        
        # 注视点
        fixation = visual.TextStim(self.win, '+', height=100, color='white')
        fixation.draw()
        self.win.flip()
        core.wait(timing['fixation_s'])
        
        # 刺激呈现
        onset_time = core.getTime() * 1000
        
        # 获取刺激图像
        p1_stim = self.self_image if trial_data['P1'] == 0 else self.face_images[trial_data['P1']]
        p2_stim = self.self_image if trial_data['P2'] == 0 else self.face_images[trial_data['P2']]
        
        # 设置位置（倒『品』布局）
        p1_stim.pos = (self.left_offset, self.top_height)
        p2_stim.pos = (self.right_offset, self.top_height)
        
        # 维度提示色块
        dim_color = self.config['dimension_colors']['dim1_hint'] if trial_data['CueDim'] == 'com' else self.config['dimension_colors']['dim2_hint']
        dim_cue = visual.Rect(
            self.win, 
            width=self.dim_block_size, 
            height=self.dim_block_size, 
            fillColor=dim_color, 
            pos=(0, self.dim_block_y)
        )
        
        # 绘制刺激
        p1_stim.draw()
        p2_stim.draw()
        dim_cue.draw()
        
        # 绘制自我环形高亮
        if trial_data['P1'] == 0 and hasattr(self, 'self_ring'):
            self.self_ring.pos = (self.left_offset, self.top_height)
            self.self_ring.draw()
        if trial_data['P2'] == 0 and hasattr(self, 'self_ring'):
            self.self_ring.pos = (self.right_offset, self.top_height)
            self.self_ring.draw()
        
        self.win.flip()
        
        # 等待反应（不限时）
        self.kb.clearEvents()
        response = None
        rt = None
        
        start_time = core.getTime()
        while True:  # 无时限等待,但避免忙等
            keys = self.kb.getKeys(['left', 'right', 'escape'])
            if keys:
                key = keys[0]
                if key.name == 'escape':
                    self.cleanup()
                    sys.exit("用户中断实验")
                elif key.name in ['left', 'right']:
                    response = 'P1' if key.name == 'left' else 'P2'
                    rt = (key.rt * 1000) if key.rt else (core.getTime() - start_time) * 1000
                    break
            self._ui_yield_and_focus_guard()  # ★ 新增：避免忙等
        
        offset_time = core.getTime() * 1000
        
        # 记录结果
        trial_result.update({
            'Choice': response or "",
            'Correct': 1 if response == trial_result['Correct_Choice'] else 0,
            'Is_Correct': 1 if (response and response == trial_result['Correct_Choice']) else 0,
            'Onset_Pair_MS': onset_time,
            'Offset_Pair_MS': offset_time,
            'RT_Choice_MS': rt or "",
            'Timeout': 0,  # 不再有超时
            'Onset_P1_MS': "",
            'Offset_P1_MS': "",
            'Onset_P2_MS': "",
            'Offset_P2_MS': ""
        })
        
        # 反馈（仅学习阶段）
        if trial_data['Type'] == 'learn' and response:
            is_correct = response == trial_result['Correct_Choice']
            
            # 设置反馈颜色和文字
            feedback_color = 'green' if is_correct else 'red'
            feedback_text = "正确!" if is_correct else "错误!"
            feedback_bold = True
            
            # 更新反馈方框颜色
            self.feedback_box_left.lineColor = feedback_color
            self.feedback_box_right.lineColor = feedback_color
            
            # 更新反馈文字
            self.feedback_text.text = feedback_text
            self.feedback_text.color = feedback_color
            self.feedback_text.bold = feedback_bold
            self.feedback_text.height = 60
            self.feedback_text.pos = (0, self.top_height + 200)
            # 绘制反馈
            p1_stim.draw()
            p2_stim.draw()
            dim_cue.draw()
            
            # 绘制选中的反馈方框
            if response == 'P1':
                self.feedback_box_left.draw()
            else:
                self.feedback_box_right.draw()
            
            self.feedback_text.draw()
            
            # 绘制自我环形高亮
            if trial_data['P1'] == 0 and hasattr(self, 'self_ring'):
                self.self_ring.pos = (self.left_offset, self.top_height)
                self.self_ring.draw()
            if trial_data['P2'] == 0 and hasattr(self, 'self_ring'):
                self.self_ring.pos = (self.right_offset, self.top_height)
                self.self_ring.draw()
            
            self.win.flip()
            core.wait(timing['feedback_s'])
        
        # ITI
        self.win.flip()  # 清屏
        iti_duration = self.rng.uniform(*timing['iti_s_uniform'])
        trial_result['ITI_S'] = iti_duration
        core.wait(iti_duration)
        
        return trial_result

    def run_interleaved_trial(self, trial_data, trial_result):
        """执行交错试次"""

        #清除按键
        timing = self.config['phase_a']['interleaved_test']
        
        # 维度提示
        dim_color = self.config['dimension_colors']['dim1_hint'] if trial_data['CueDim'] == 'com' else self.config['dimension_colors']['dim2_hint']
        dim_cue = visual.Rect(
            self.win, 
            width=self.dim_block_size, 
            height=self.dim_block_size, 
            fillColor=dim_color, 
            pos=(0, self.dim_block_y)
        )
        dim_cue.draw()
        self.win.flip()
        core.wait(timing['dim_cue_s'])
        
        # 白色注视点
        fixation = visual.TextStim(self.win, '+', height=100, color='white')
        fixation.draw()
        self.win.flip()
        core.wait(timing['fixation_white_s'])
        
        # P1呈现（居中）
        p1_onset = core.getTime() * 1000
        p1_stim = self.self_image if trial_data['P1'] == 0 else self.face_images[trial_data['P1']]
        p1_stim.pos = (0, 0)  # 居中显示
        p1_stim.draw()
        
        if trial_data['P1'] == 0 and hasattr(self, 'self_ring'):
            self.self_ring.pos = (0, 0)
            self.self_ring.draw()
        
        self.win.flip()
        core.wait(timing['p1_s'])
        p1_offset = core.getTime() * 1000
        
        # 紫色注视点
        fixation_purple = visual.TextStim(self.win, '+', height=100, color='purple')
        fixation_purple.draw()
        self.win.flip()
        purple_duration = self.rng.uniform(*timing['fixation_purple_uniform_s'])
        core.wait(purple_duration)
        
        # P2呈现（居中,需要作答）
        p2_onset = core.getTime() * 1000
        p2_stim = self.self_image if trial_data['P2'] == 0 else self.face_images[trial_data['P2']]
        p2_stim.pos = (0, 0)  # 居中显示
        p2_stim.draw()
        
        if trial_data['P2'] == 0 and hasattr(self, 'self_ring'):
            self.self_ring.pos = (0, 0)
            self.self_ring.draw()
        
        self.win.flip()
        
        # 等待反应
        self.kb.clearEvents()
        response = None
        rt = None
        
        start_time = core.getTime()
        while core.getTime() - start_time < timing['p2_s']:
            keys = self.kb.getKeys(['left', 'right', 'escape'])
            if keys:
                key = keys[0]
                if key.name == 'escape':
                    self.cleanup()
                    sys.exit("用户中断实验")
                elif key.name in ['left', 'right']:
                    # 在交错测试中,left=P1更高,right=P2更高
                    if key.name == 'left':
                        response = 'P1'
                    else:
                        response = 'P2'
                    rt = (key.rt * 1000) if key.rt else (core.getTime() - start_time) * 1000
                    break
            self._ui_yield_and_focus_guard()  # ★ 新增：避免忙等
        
        p2_offset = core.getTime() * 1000
        
        # 绿色注视点
        fixation_green = visual.TextStim(self.win, '+', height=100, color='green')
        fixation_green.draw()
        self.win.flip()
        green_duration = self.rng.uniform(*timing['fixation_green_uniform_s'])
        core.wait(green_duration)
        
        # 记录结果
        trial_result.update({
            'Choice': response or "",
            'Correct': 1 if response == trial_result['Correct_Choice'] else 0,
            'Is_Correct': 1 if (response and response == trial_result['Correct_Choice']) else 0,
            'Onset_P1_MS': p1_onset,
            'Offset_P1_MS': p1_offset,
            'Onset_P2_MS': p2_onset,
            'Offset_P2_MS': p2_offset,
            'RT_Choice_MS': rt or "",
            'Timeout': 1 if response is None else 0,
            'Onset_Pair_MS': "",
            'Offset_Pair_MS': "",
            'FixWhite_S': timing['fixation_white_s'],
            'FixPurple_S': purple_duration,
            'FixGreen_S': green_duration
        })
        
        return trial_result

    def save_trial_data(self, trial_result):
        """保存试次数据"""
        output_path = os.path.join(
            self.script_dir,
            self.config['paths']['data_dir'],
            'A',
            f"subj_{self.subject_id}_trials.csv"
        )
        
        # 添加审计列
        full_result = {
            'SubID': self.subject_id,
            'SessionID': self.subject_profile['SessionID'],
            'Phase': self.phase,
            'Day': self.current_day,
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
        
        # 确保列顺序
        columns = [
            'SubID', 'SessionID', 'Phase', 'Day', 'Block', 'Trial', 'NameHash', 
            'SchemaVer', 'ConfigChecksum', 'SubjectMapChecksum', 'SelfImageSHA256', 'Seed',
            'Type', 'CueDim', 'P1', 'P2', 'P1_com', 'P1_pop', 'P2_com', 'P2_pop',
            'P1_angle_deg', 'P2_angle_deg', 'P1_angle_bin', 'P2_angle_bin',
            'P1_dist', 'P2_dist', 'C1', 'C2', 'DeltaDim', 'Choice', 'Correct_Choice', 'Is_Correct',
            'Onset_Pair_MS', 'Offset_Pair_MS', 'Onset_P1_MS', 'Offset_P1_MS',
            'Onset_P2_MS', 'Offset_P2_MS', 'RT_Choice_MS', 'Timeout',
            'FixWhite_S', 'FixPurple_S', 'FixGreen_S', 'ITI_S'
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
            # 首次写表头
            df_row.to_csv(output_path, index=False)
        else:
            # 检查现有文件的列结构
            existing_df = pd.read_csv(output_path, nrows=0)  # 只读表头
            existing_columns = list(existing_df.columns)
            
            if existing_columns != columns:
                missing_cols = set(columns) - set(existing_columns)
                extra_cols = set(existing_columns) - set(columns)
                
                error_msg = f"CSV文件列结构不匹配!\n"
                if missing_cols:
                    error_msg += f"缺少列: {sorted(missing_cols)}\n"
                if extra_cols:
                    error_msg += f"多余列: {sorted(extra_cols)}\n"
                error_msg += f"期望列: {columns}\n"
                error_msg += f"现有列: {existing_columns}\n"
                error_msg += "请先统一表头或删除旧文件重新开始。"
                
                raise ValueError(error_msg)
            
            df_row.to_csv(output_path, mode='a', header=False, index=False)
        
        self.completed_trials.append(full_result)
        
        # 更新准确率历史
        if full_result.get('Choice'):
            self.accuracy_history.append(full_result.get('Correct', 0))
            
            # 根据试次类型分别记录准确率
            if full_result.get('Type') == 'test':
                self.test_accuracy_history.append(full_result.get('Correct', 0))
            elif full_result.get('Type') == 'interleaved':
                self.interleaved_accuracy_history.append(full_result.get('Correct', 0))

    def check_completion_criteria(self):
        """检查是否达到完成标准"""
        if len(self.accuracy_history) == 0:
            return False
        
        target_accuracy = self.config['phase_a']['criteria']['accuracy_target']
        current_accuracy = sum(self.accuracy_history) / len(self.accuracy_history)
        
        print(f"当前总体准确率: {current_accuracy:.3f} (目标: {target_accuracy})")
        
        return current_accuracy >= target_accuracy

    def show_instruction(self, text, wait_for_key=True):
        """显示指导语"""
        instr_text = visual.TextStim(
            self.win, text, height=0.03*self.win.size[1], wrapWidth=0.9*self.win.size[0], 
            alignText='center',anchorHoriz='center',color='white', font='SimHei'
        )
        instr_text.draw()
        self.win.flip()
        if wait_for_key:
            event.waitKeys(keyList=['space'])
    
    def show_break(self, duration_s=60):
        """显示休息页面"""
        break_text = f"休息 {duration_s} 秒\n\n按空格键跳过休息"
        break_stim = visual.TextStim(
            self.win, break_text, height=0.035*self.win.size[1], wrapWidth=0.9*self.win.size[0], 
            alignText='center',anchorHoriz='center',color='white', font='SimHei'
        )
        break_stim.draw()
        self.win.flip()
        
        # 倒计时
        start_time = core.getTime()
        while core.getTime() - start_time < duration_s:
            remaining = int(duration_s - (core.getTime() - start_time))
            if remaining > 0:
                countdown_text = f"休息 {remaining} 秒\n\n按空格键跳过休息"
                countdown_stim = visual.TextStim(
                    self.win, countdown_text, height=0.035*self.win.size[1], wrapWidth=0.9*self.win.size[0],
                    alignText='center',anchorHoriz='center',color='white', font='SimHei'
                )
                countdown_stim.draw()
                self.win.flip()
            
            # 检查是否按空格跳过
            keys = event.getKeys(['space'])
            if keys:
                break
    
    def show_accuracy_page(self, accuracy, target, duration_s=2):
        """显示准确率页面"""
        accuracy_text = f"总体准确率: {accuracy:.1%}\n\n目标: {target:.1%}"
        if accuracy >= target:
            accuracy_text += "\n\n达标!"
        else:
            accuracy_text += "\n\n未达标,继续训练..."
        
        accuracy_stim = visual.TextStim(
            self.win, accuracy_text, height=0.035*self.win.size[1], wrapWidth=0.9*self.win.size[0],
            alignText='center',anchorHoriz='center',color='white', font='SimHei'
            )
        accuracy_stim.draw()
        self.win.flip()
        core.wait(duration_s)

    def _log_test_start(self, batch_size, is_fast,text='Test'):
        """测试阶段开场日志"""
        print(f"[{text}-START] Sub:{self.subject_id} | Day:{self.current_day} | Dim:{self.current_dim} "
            f"| TrialsThisBatch:{batch_size} | FastTest:{is_fast}")

    def _log_test_progress(self, i, n, result,text='Test'):
        """每完成一个trial后的进度日志（在 save_trial_data(result) 之后调用）"""
        # 当前测试阶段累计准确率（save_trial_data 已经把这次结果计入 test_accuracy_history）
        if len(self.test_accuracy_history) > 0:
            acc = sum(self.test_accuracy_history) / len(self.test_accuracy_history)
            acc_txt = f"{acc:.1%}"
        else:
            acc_txt = "n/a"

        status = "OK" if result.get('Is_Correct', 0) == 1 else "ERR"
        print(
            f"[{text}] {i+1}/{n} | P1={result.get('P1')} P2={result.get('P2')} "
            f"| Δ={result.get('DeltaDim','')} | Resp={result.get('Choice','')} "
            f"| Ans={result.get('Correct_Choice','')} | {status} "
            f"| RT={result.get('RT_Choice_MS','')}ms | AccSoFar={acc_txt}"
        )


    def run_day1_learning(self):
        """运行Day1学习阶段"""
        # 确保当前维度的候选池存在
        self.ensure_pool_for_dim(self.current_dim)
        
        dim1_color=self.config['dimension_colors']['dim1_hint']
        dim2_color=self.config['dimension_colors']['dim2_hint']
        # Day1学习指导语
        dim_name = "能力" if self.current_dim == 'com' else "受欢迎度"
        dim_color = dim1_color if self.current_dim == 'com' else dim2_color
        
        learn_instruction = f"""第1天学习阶段

每一次试验中,屏幕上会出现来自81位企业家中的两张面孔,请您根据第一印象来判断两人之间“谁是更优秀的企业家？”。

当两位企业家之间的方块为深蓝色时,请您判断谁更加受到欢迎,两人之间只相差一个等级。

当两位企业家之间的方块为橙色时,请您判断谁具有更高的能力,两人之间只相差一个等级。

请通过按键盘上的左箭头或右箭头键选择您认为的“更优秀的一方”。左键选择左边的面孔,右键选择右边的面孔。

您的选择正确与否将得到反馈：屏幕将显示“正确!”或“错误!”。每个试次作答完成后进入下一次试次。

当屏幕上显示“休息”时,请稍作休息,并在准备好后按空格键继续任务。

学习环节结束,请稍作休息,并在准备好后按空格键进入测试环节。"""
        
        self.show_instruction(learn_instruction)

        # 学习循环
        while True:
            batch_size = self.config['phase_a']['criteria']['trials_batch_size']
            learn_trials = self._sample_trials_from_pool('learn', batch_size)
            
            # 打印学习阶段开场日志
            self._log_test_start(batch_size, self.fast_test_enabled,'学习')
            # 更新最近学习池
            self._update_recent_learn_pool(learn_trials)
            
            for i, trial in enumerate(learn_trials):
                result = self.run_trial(trial)
                # 打印每完成一个trial后的进度日志
                self._log_test_progress(i, len(learn_trials), result,'学习')
                self.save_trial_data(result)
                self.current_trial += 1
                self.save_checkpoint()
            
            # 休息
            self.show_break(30)
            
            # 测试
            test_instruction = f"""第1天测试阶段

与学习阶段类似,当人物呈现时,请您通过左右按键选择“更优秀的一方”,两人之间可能相差一个至多个等级。

请注意：本阶段您的选择正确与否将不会得到反馈。每个试次作答完成后进入下一次试次。

当屏幕上显示“休息”时,请稍作休息,并在准备好后按空格键继续任务。"""
            
            self.show_instruction(test_instruction)
            
            dev_config = self.config.get('dev', {})
            if self.fast_test_enabled:
                # “快速测试”模式：小批量 + 专用采样 + 可早停
                min_test_trials = dev_config.get('min_test_trials', 3)
                test_batch_size = min(min_test_trials, batch_size)
                test_trials = self._sample_fast_test_trials(test_batch_size)
                stop_on_first_pass = dev_config.get('stop_on_first_pass', True)
            else:
                # 普通测试：用整批大小采样，不早停，不打印“快速测试早停”
                test_batch_size = batch_size
                test_trials = self._sample_trials_from_pool('test', test_batch_size)
                stop_on_first_pass = False

            # 打印测试阶段开场日志
            self._log_test_start(test_batch_size, self.fast_test_enabled,'测试')
            
            for i, trial in enumerate(test_trials):
                result = self.run_trial(trial)
                # 打印每完成一个trial后的进度日志
                self._log_test_progress(i, len(test_trials), result,'测试')
                self.save_trial_data(result)
                self.current_trial += 1
                self.save_checkpoint()
                
                # 早停检查
                if stop_on_first_pass and len(self.test_accuracy_history) > 0:
                    current_accuracy = sum(self.test_accuracy_history) / len(self.test_accuracy_history)
                    if current_accuracy >= 0.90:
                        print(f"快速测试早停：第{i+1}题后达到90%准确率")
                        break
            
            # 检查准确率
            if len(self.test_accuracy_history) > 0:
                test_accuracy = sum(self.test_accuracy_history) / len(self.test_accuracy_history)
                self.show_accuracy_page(test_accuracy, 0.90)
                
                if test_accuracy >= 0.90:
                    # Day1达标,保存day1_dim到profile
                    self._save_day1_dim_to_profile()
                    break
                else:
                    # 重置测试准确率历史,重新开始学习
                    self.test_accuracy_history = []
    
    def _save_day1_dim_to_profile(self):
        """保存Day1维度到profile.json"""
        profile_path = os.path.join(
            self.script_dir,
            self.config['paths']['subjects_dir'],
            f"subj_{self.subject_id}_profile.json"
        )
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
        else:
            profile_data = {}
        
        # 更新day1_dim（仅当不存在时写入）
        if 'day1_dim' not in profile_data:
            profile_data['day1_dim'] = self.current_dim
            profile_data['day2_dim'] = 'pop' if self.current_dim == 'com' else 'com'
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
            
            print(f"已保存Day1维度: {self.current_dim}")
    
    def run_day2_learning(self):
        """运行Day2学习阶段"""
        # 确保当前维度的候选池存在
        self.ensure_pool_for_dim(self.current_dim)
        
        dim1_color=self.config['dimension_colors']['dim1_hint']
        dim2_color=self.config['dimension_colors']['dim2_hint']
        # Day2学习指导语
        dim_name = "能力" if self.current_dim == 'com' else "受欢迎度"
        dim_color = dim1_color if self.current_dim == 'com' else dim2_color
        
        learn_instruction = f"""第2天学习阶段

每一次试验中,屏幕上会出现来自81位企业家中的两张面孔,请您根据第一印象来判断两人之间“谁是更优秀的企业家？”。

当两位企业家之间的方块为深蓝色时,请您判断谁更加受到欢迎,两人之间只相差一个等级。

当两位企业家之间的方块为橙色时,请您判断谁具有更高的能力,两人之间只相差一个等级。

请通过按键盘上的左箭头或右箭头键选择您认为的“更优秀的一方”。左键选择左边的面孔,右键选择右边的面孔。

您的选择正确与否将得到反馈：屏幕将显示“正确!”或“错误!”。每个试次作答完成后进入下一次试次。

当屏幕上显示“休息”时,请稍作休息,并在准备好后按空格键继续任务。

学习环节结束,请稍作休息,并在准备好后按空格键进入测试环节。"""
        
        self.show_instruction(learn_instruction)


        
        # 学习循环
        while True:
            batch_size = self.config['phase_a']['criteria']['trials_batch_size']
            learn_trials = self._sample_trials_from_pool('learn', batch_size)
            # 打印学习阶段开场日志
            self._log_test_start(batch_size, self.fast_test_enabled,'学习')
            # 更新最近学习池
            self._update_recent_learn_pool(learn_trials)
            
            for i, trial in enumerate(learn_trials):
                result = self.run_trial(trial)
                # 打印每完成一个trial后的进度日志
                self._log_test_progress(i, len(learn_trials), result,'学习')
                self.save_trial_data(result)
                self.current_trial += 1
                self.save_checkpoint()
            
            # 休息
            self.show_break(30)
            
            # 测试
            test_instruction = f"""第2天测试阶段

与学习阶段类似,当人物呈现时,请您通过左右按键选择“更优秀的一方”,两人之间可能相差一个至多个等级。

请注意：本阶段您的选择正确与否将不会得到反馈。每个试次作答完成后进入下一次试次 。

当屏幕上显示“休息”时,请稍作休息,并在准备好后按空格键继续任务。"""
            
            self.show_instruction(test_instruction)
            
            # 使用快速测试采样
            dev_config = self.config.get('dev', {})
            if self.fast_test_enabled:
                # “快速测试”模式：小批量 + 专用采样 + 可早停
                min_test_trials = dev_config.get('min_test_trials', 3)
                test_batch_size = min(min_test_trials, batch_size)
                test_trials = self._sample_fast_test_trials(test_batch_size)
                stop_on_first_pass = dev_config.get('stop_on_first_pass', True)
            else:
                # 普通测试：用整批大小采样，不早停，不打印“快速测试早停”
                test_batch_size = batch_size
                test_trials = self._sample_trials_from_pool('test', test_batch_size)
                stop_on_first_pass = False
            
            # 打印测试阶段开场日志
            self._log_test_start(test_batch_size, self.fast_test_enabled,'测试')

            for i, trial in enumerate(test_trials):
                result = self.run_trial(trial)
                # 打印每完成一个trial后的进度日志
                self._log_test_progress(i, len(test_trials), result,'测试')
                self.save_trial_data(result)
                self.current_trial += 1
                self.save_checkpoint()
                
                # 早停检查
                if stop_on_first_pass and len(self.test_accuracy_history) > 0:
                    current_accuracy = sum(self.test_accuracy_history) / len(self.test_accuracy_history)
                    if current_accuracy >= 0.90:
                        print(f"快速测试早停：第{i+1}题后达到90%准确率")
                        break
            
            # 检查准确率
            if len(self.test_accuracy_history) > 0:
                test_accuracy = sum(self.test_accuracy_history) / len(self.test_accuracy_history)
                self.show_accuracy_page(test_accuracy, 0.90)
                
                if test_accuracy >= 0.90:
                    break
                else:
                    # 重置测试准确率历史,重新开始学习
                    self.test_accuracy_history = []
    
    def run_interleaved_test(self):
        """运行交错测试阶段"""
        interleaved_instruction = """交错测试阶段

恭喜您已经顺利完成对81位企业家优秀程度的学习!

在本测试任务中,请您根据方块的颜色,来判断两位企业家之间谁是“更优秀的一方”。

回答正确越多,获得的奖金就越多。若您在本测试阶段总体正确率超过90%,将获得额外奖金;正确率超过95%,奖金将翻倍。

如果您的总体正确率未达到90%,将重新进行学习,直到达到90% 才可进行后续的实验。"""
        
        self.show_instruction(interleaved_instruction)
        
        # 交错测试循环
        while True:
            batch_size = self.config['phase_a']['criteria']['trials_batch_size']
            inter_trials = self.generate_trials('interleaved', None, batch_size)
            # 打印交错 测试阶段开场日志
            self._log_test_start(batch_size, self.fast_test_enabled,'交错测试')
            for i, trial in enumerate(inter_trials):
                result = self.run_trial(trial)
                # 打印每完成一个trial后的进度日志
                self._log_test_progress(i, len(inter_trials), result,'交错测试')
                self.save_trial_data(result)
                self.current_trial += 1
                self.save_checkpoint()
            
            # 检查准确率
            if len(self.interleaved_accuracy_history) > 0:
                interleaved_accuracy = sum(self.interleaved_accuracy_history) / len(self.interleaved_accuracy_history)
                self.show_accuracy_page(interleaved_accuracy, 0.80)
                
                if interleaved_accuracy >= 0.80:
                    break
                else:
                    # 不达标,按顺序重复：Day1学习→Day2学习→交错
                    print("交错测试未达标,重新进行Day1学习...")
                    self.test_accuracy_history = []
                    self.interleaved_accuracy_history = []
                    
                    # 清空快速测试的最近池（维度隔离）
                    if self.fast_test_enabled:
                        self.recent_learn_stimuli = {'com': set(), 'pop': set()}
                        self.recent_learn_pairs = {'com': set(), 'pop': set()}
                        print("快速测试模式：已清空最近学习池")
                    
                    # Day1学习
                    self.current_dim = self.subject_profile['day1_dim']
                    self.run_day1_learning()
                    
                    # Day2学习
                    self.current_dim = self.subject_profile['day2_dim']
                    self.run_day2_learning()
    
    def run_experiment(self):
        """运行实验主循环"""
        print(f"\n开始 Phase A 实验 - 被试: {self.subject_id}, 运行: {self.run_day}")
        print(f"当前维度: {self.current_dim}\n共进行{self.config['phase_a']['criteria']['trials_batch_size']}个试次")
        print("=" * 50)
        
        # 总指导语
        total_instruction = f""" 实验任务:判断谁是更优秀的企业家 

在本任务中,您将看到来自81位企业家(其中一个是自己)中两位的面孔配对呈现。这些企业家在此之前已根据

两个独立标准进行评估其优秀程度:能力和受欢迎度。请注意:两个标准是独立的,即每位企业家在一方面的等级

与另一方面无关。您将通过左右按键选择,来判断二者之中谁更优秀。

受欢迎度将被定义为个人通过众筹筹集资金的能力。例如,如果A在一个项目筹集了1000万美元,

而商人B只筹集了500万美元,那么A的受欢迎度相较于B会被认为更高。能力将被定义为个人的社会职位等级。

例如,A在社会团体中的职位为总经理,B为主管,那么A的能力相较于B会被认为更高。

本任务将进行连续两天的训练,您将在任务中进行学习,来了解81位企业家。训练期间每天分别进行学习,

并在学习后进行测试,以便了解您当日的学习情况。请确保您已经充分理解以上内容,即将开始学习阶段,

准备好后按空格继续。"""
        
        self.show_instruction(total_instruction)
        
        # 根据RunDay执行对应流程
        if self.run_day == 'Day1':
            self.run_day1_learning()
        else:  # Day2
            self.run_day2_learning()
            self.run_interleaved_test()
        
        # 实验完成
        final_accuracy = sum(self.accuracy_history) / len(self.accuracy_history) if self.accuracy_history else 0
        completion_text = f"""{self.run_day} 完成!

最终准确率: {final_accuracy:.1%}
总试次数: {len(self.completed_trials)}

感谢参与!"""
        
        self.show_instruction(completion_text, wait_for_key=False)
        core.wait(3)

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
            self.load_or_create_stim_table()
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
    parser = argparse.ArgumentParser(description='Phase A: 两日训练阶段')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--subject', type=str, help='被试ID')
    parser.add_argument('--resume', action='store_true', help='从断点继续')
    parser.add_argument('--allow-init', action='store_true', help='允许初始化新被试')
    parser.add_argument('--day', type=str, choices=['Day1', 'Day2'], help='预选运行天数')
    
    args = parser.parse_args()
    
    experiment = PhaseAExperiment(
        config_path=args.config,
        subject_id=args.subject,
        resume=args.resume,
        allow_init=args.allow_init,
        run_day=args.day
    )
    
    experiment.run()


def render_subject_grid_mosaic(subject_profile, stim_table, config, script_dir):
    """
    渲染被试坐标网格图（自适应尺寸）
    
    Args:
        subject_profile: 被试档案字典
        stim_table: 刺激表DataFrame
        config: 配置字典
        script_dir: 脚本目录路径
    
    Returns:
        Path: 输出PNG文件路径
    """
    from pathlib import Path
    
    # 获取网格尺寸参数
    coord_limit = int(config['stimuli']['coord_limit'])
    grid_n = 2 * coord_limit + 1 - coord_limit%2
    center_idx = coord_limit
    
    # 获取渲染参数
    render_config = config.get('render', {}).get('grid', {})
    cell_px = render_config.get('cell_px', 110)
    gap_px = render_config.get('gap_px', 16)
    pad_px = render_config.get('pad_px', 28)
    face_inset_ratio = render_config.get('face_inset_ratio', 0.9)
    draw_axes = render_config.get('draw_axes', True)
    show_ids = render_config.get('show_ids', False)
    highlight_self_ring = render_config.get('highlight_self_ring', True)
    ring_line_px = render_config.get('ring_line_px', 6)
    ring_color = render_config.get('ring_color', '#FFD400')
    
    # 计算画布尺寸（自适应）
    canvas_width = pad_px * 2 + grid_n * cell_px + (grid_n - 1) * gap_px
    canvas_height = pad_px * 2 + grid_n * cell_px + (grid_n - 1) * gap_px
    
    # 创建画布
    img = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # 加载字体（修复getmask错误）
    try:
        font_large = ImageFont.truetype("SimHei.ttf", 24)
        font_medium = ImageFont.truetype("SimHei.ttf", 18)
        font_small = ImageFont.truetype("SimHei.ttf", 12)
    except:
        font_large = font_medium = font_small = ImageFont.load_default()
    
    # 获取刺激目录路径
    faces_dir = os.path.join(script_dir, config['paths']['stimuli_faces_dir'])
    self_dir = os.path.join(script_dir, config['paths']['stimuli_self_dir'])
    
    # 过滤坐标在[-coord_limit,+coord_limit]范围内的刺激
    filtered_table = stim_table[
        (stim_table['com'] >= -coord_limit) & (stim_table['com'] <= coord_limit) &
        (stim_table['pop'] >= -coord_limit) & (stim_table['pop'] <= coord_limit)
    ].copy()
    
    if len(filtered_table) < len(stim_table):
        excluded_count = len(stim_table) - len(filtered_table)
        print(f"{excluded_count} 个刺激超出 {grid_n}×{grid_n} 可视范围,未显示。")
    
    # 创建坐标到刺激ID的映射（确保int类型）
    coord_to_stim = {(int(r['com']), int(r['pop'])): int(r['StimID'])
                     for _, r in stim_table.iterrows()}

    
    # 绘制网格线（自适应）
    for i in range(grid_n + 1):
        x = pad_px + i * (cell_px + gap_px)
        y = pad_px + i * (cell_px + gap_px)
        line_w = 2 if (draw_axes and i == center_idx) else 1
        color = 'black' if (draw_axes and i == center_idx) else 'lightgray'
        draw.line([(x, pad_px), (x, canvas_height - pad_px)], fill=color, width=line_w)
        draw.line([(pad_px, y), (canvas_width - pad_px, y)], fill=color, width=line_w)

    dim1_color=config['dimension_colors']['dim1_hint']
    dim2_color=config['dimension_colors']['dim2_hint']
    # 绘制坐标轴标签（自适应）
    if draw_axes:
        # X轴标签 (com)
        for i in range(grid_n):
            com_val = i - coord_limit
            x = pad_px + i * (cell_px + gap_px) + cell_px // 2
            y = canvas_height - pad_px + 8
            draw.text((x, y), str(com_val), bold=True, fill='black', anchor='mt', font=font_small)
        
        # Y轴标签 (pop)（上正下负）
        for i in range(grid_n):
            pop_val = coord_limit - i
            x = pad_px - 12
            y = pad_px + i * (cell_px + gap_px) + cell_px // 2
            draw.text((x, y), str(pop_val), bold=True,  fill='black', anchor='rm', font=font_small)
        
        # 轴标题
        draw.text((canvas_width // 2, canvas_height-20), 'Com', fill=dim1_color, anchor='mt', font=font_medium)
        draw.text((5, canvas_height // 2-10), 'Pop', fill=dim2_color, anchor='lm', font=font_medium)
    
    # 绘制标题（自适应尺寸）
    title = f"Sub {subject_profile['SubjectID']}__{grid_n}*{grid_n} Grid"
    title_bbox = draw.textbbox((0, 0), title, font=font_large)
    title_x = (canvas_width - (title_bbox[2] - title_bbox[0])) // 2
    draw.text((title_x, 8), title, fill='black', font=font_large)
    
    # 绘制每个格子（自适应）
    for row in range(grid_n):
        for col in range(grid_n):
            com = col - coord_limit
            pop = coord_limit - row  # 上正下负
            
            # 计算格子位置
            cell_x = pad_px + col * (cell_px + gap_px)
            cell_y = pad_px + row * (cell_px + gap_px)
            
            # 检查是否是自我位置
            is_self = (com, pop) == (0, 0)
            
            if is_self:
                # 绘制自我图像
                self_image_path = os.path.join(self_dir, subject_profile['SelfImageFile'])
                if os.path.exists(self_image_path):
                    try:
                        self_img = Image.open(self_image_path)
                        self_img = self_img.convert('RGBA')
                        _place_image_in_cell(img, self_img, cell_x, cell_y, cell_px, face_inset_ratio)
                    except Exception as e:
                        print(f"无法加载自我图像 {self_image_path}: {e}")
                        _draw_placeholder(draw, cell_x, cell_y, cell_px, face_inset_ratio, "SELF", font_medium)
                else:
                    print(f"自我图像文件不存在: {self_image_path}")
                    _draw_placeholder(draw, cell_x, cell_y, cell_px, face_inset_ratio, "SELF", font_medium)
                
                # 绘制自我环
                if highlight_self_ring:
                    ring_radius = int(cell_px * face_inset_ratio // 2) + 8
                    ring_center_x = cell_x + cell_px // 2
                    ring_center_y = cell_y + cell_px // 2
                    draw.ellipse([
                        ring_center_x - ring_radius, ring_center_y - ring_radius,
                        ring_center_x + ring_radius, ring_center_y + ring_radius
                    ], outline=ring_color, width=ring_line_px)
            else:
                # 绘制他人图像
                stim_id = coord_to_stim.get((com, pop))
                if stim_id:
                    stim_id = int(stim_id)  # 关键：防止 66.0
                    # 尝试加载图像文件（大小写兼容）
                    image_path = None
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        test_path = os.path.join(faces_dir, f"P{stim_id}{ext}")
                        if os.path.exists(test_path):
                            image_path = test_path
                            break
                    
                    if image_path:
                        try:
                            face_img = Image.open(image_path)
                            face_img = face_img.convert('RGBA')
                            _place_image_in_cell(img, face_img, cell_x, cell_y, cell_px, face_inset_ratio)
                            
                            # 显示ID（如果启用）
                            if show_ids:
                                id_text = f"P{stim_id}"
                                text_bbox = draw.textbbox((0, 0), id_text, font=font_small)
                                text_width = text_bbox[2] - text_bbox[0]
                                text_height = text_bbox[3] - text_bbox[1]
                                text_x = cell_x + (cell_px - text_width) // 2
                                text_y = cell_y + (cell_px - text_height) // 2
                                # 绘制半透明背景
                                draw.rectangle([text_x-2, text_y-2, text_x+text_width+2, text_y+text_height+2], 
                                             fill=(255, 255, 255, 128))
                                draw.text((text_x, text_y), id_text, bold=True,  fill='black', font=font_small)
                        except Exception as e:
                            print(f"无法加载图像 P{stim_id}: {e}")
                            _draw_placeholder(draw, cell_x, cell_y, cell_px, face_inset_ratio, f"P{stim_id}", font_medium)
                    else:
                        print(f"缺图告警: P{stim_id} 图像文件不存在")
                        _draw_placeholder(draw, cell_x, cell_y, cell_px, face_inset_ratio, f"P{stim_id}", font_medium)
    
    # 保存PNG
    output_path = os.path.join(
        script_dir,
        config['paths']['subjects_dir'],
        f"subj_{subject_profile['SubjectID']}_grid.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, 'PNG', dpi=(300, 300))
    
    return Path(output_path)


def _place_image_in_cell(canvas, image, cell_x, cell_y, cell_px, face_inset_ratio):
    """在格子中放置图像"""
    # 计算图像尺寸
    face_size = int(cell_px * face_inset_ratio)
    
    # 计算图像位置（居中）
    face_x = cell_x + (cell_px - face_size) // 2
    face_y = cell_y + (cell_px - face_size) // 2
    
    # 调整图像大小
    image = image.resize((face_size, face_size), Image.Resampling.LANCZOS)
    
    # 粘贴图像（处理透明度）
    if image.mode == 'RGBA':
        canvas.paste(image, (face_x, face_y), image)
    else:
        canvas.paste(image, (face_x, face_y))


def _draw_placeholder(draw, cell_x, cell_y, cell_px, face_inset_ratio, text, font):
    """绘制占位符"""
    # 计算占位符尺寸
    face_size = int(cell_px * face_inset_ratio)
    face_x = cell_x + (cell_px - face_size) // 2
    face_y = cell_y + (cell_px - face_size) // 2
    
    # 绘制灰色背景
    draw.rectangle([face_x, face_y, face_x + face_size, face_y + face_size], 
                   fill='lightgray', outline='gray')
    
    # 绘制文本
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = face_x + (face_size - text_width) // 2
    text_y = face_y + (face_size - text_height) // 2
    draw.text((text_x, text_y), text, fill='black', font=font)


if __name__ == '__main__':
    main()
