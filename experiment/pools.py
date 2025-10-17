#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
试次池生成和持久化模块
为 Phase B 和 Phase C 生成不重叠的试次池
"""

import os
import json
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path


class TrialPoolsManager:
    """试次池管理器"""
    
    def __init__(self, config, subject_id, script_dir):
        self.config = config
        self.subject_id = subject_id
        self.script_dir = script_dir
        self.pools_config = config.get('pools', {})
        self.n_faces = config['stimuli']['n_faces']
        
        # 生成被试级子种子
        self.subject_seed = self._generate_subject_seed()
        self.rng = np.random.default_rng(self.subject_seed)
        
    def _generate_subject_seed(self):
        """生成被试级子种子"""
        global_seed = self.config['experiment']['seed']
        subject_hash = hash(self.subject_id) % (2**31)
        return global_seed + subject_hash
    
    def _get_pools_path(self):
        """获取试次池文件路径"""
        pattern = self.pools_config.get('save_path_pattern', 'Data/subjects/subj_{id}_pools.json')
        return os.path.join(self.script_dir, pattern.format(id=self.subject_id))
    
    def _generate_all_pairs(self):
        """生成所有有序配对 (i,j) where i != j, i,j in [1, n_faces]"""
        pairs = []
        for i in range(1, self.n_faces + 1):
            for j in range(1, self.n_faces + 1):
                if i != j:
                    pairs.append([i, j])
        return pairs
    
    def _split_pairs_into_blocks(self, pairs, n_blocks, trials_per_block):
        """将配对列表分割成指定数量的块"""
        total_trials = n_blocks * trials_per_block
        if len(pairs) < total_trials:
            # 如果配对不够，允许重复
            pairs = pairs * ((total_trials // len(pairs)) + 1)
        
        # 随机打乱
        self.rng.shuffle(pairs)
        
        # 分割成块
        blocks = []
        for i in range(n_blocks):
            start_idx = i * trials_per_block
            end_idx = start_idx + trials_per_block
            blocks.append(pairs[start_idx:end_idx])
        
        return blocks
    
    def build_or_load_pools(self):
        """构建或加载试次池"""
        pools_path = self._get_pools_path()
        
        # 检查文件是否存在
        if os.path.exists(pools_path):
            try:
                with open(pools_path, 'r', encoding='utf-8') as f:
                    pools_data = json.load(f)
                
                # 检查配置校验和是否一致
                current_config_checksum = self._calculate_config_checksum()
                stored_checksum = pools_data.get('config_checksum', '')
                
                if current_config_checksum == stored_checksum:
                    print(f"加载现有试次池: {pools_path}")
                    return pools_data
                else:
                    print(f"警告: 试次池配置校验和不匹配，将重新生成")
                    print(f"  存储的校验和: {stored_checksum}")
                    print(f"  当前校验和: {current_config_checksum}")
            except Exception as e:
                print(f"加载试次池失败: {e}，将重新生成")
        
        # 生成新的试次池
        print("生成新的试次池...")
        return self._generate_pools()
    
    def _calculate_config_checksum(self):
        """计算配置校验和"""
        relevant_config = {
            'n_faces': self.n_faces,
            'phase_b': self.config.get('phase_b', {}),
            'phase_c': self.config.get('phase_c', {}),
            'pools': self.pools_config
        }
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _generate_pools(self):
        """生成试次池"""
        # 生成所有有序配对
        all_pairs = self._generate_all_pairs()
        print(f"生成了 {len(all_pairs)} 个有序配对")
        
        # 获取配置参数
        phase_b_config = self.config.get('phase_b', {})
        phase_c_config = self.config.get('phase_c', {})
        
        b_blocks = phase_b_config.get('blocks', 3)
        b_trials_per_block = phase_b_config.get('trials_per_block', 24)
        c_blocks = phase_c_config.get('blocks', 3)
        c_trials_per_block = phase_c_config.get('trials_per_block', 24)
        
        # 计算需要的总试次数
        b_total = b_blocks * b_trials_per_block
        c_total = c_blocks * c_trials_per_block
        
        print(f"Phase B 需要: {b_blocks} 块 × {b_trials_per_block} 试次 = {b_total} 试次")
        print(f"Phase C 需要: {c_blocks} 块 × {c_trials_per_block} 试次 = {c_total} 试次")
        
        # 分割配对
        b_blocks_data = self._split_pairs_into_blocks(all_pairs, b_blocks, b_trials_per_block)
        
        # 为 Phase C 生成剩余配对
        used_pairs = set()
        for block in b_blocks_data:
            for pair in block:
                used_pairs.add(tuple(pair))
        
        remaining_pairs = [list(pair) for pair in all_pairs if tuple(pair) not in used_pairs]
        print(f"Phase B 使用后剩余 {len(remaining_pairs)} 个配对")
        
        c_blocks_data = self._split_pairs_into_blocks(remaining_pairs, c_blocks, c_trials_per_block)
        
        # 构建池数据
        pools_data = {
            'schema': 'pools-1.0',
            'subject': self.subject_id,
            'created_at': datetime.now().isoformat(),
            'config_checksum': self._calculate_config_checksum(),
            'subject_seed': self.subject_seed,
            'n_faces': self.n_faces,
            'B': b_blocks_data,
            'C': c_blocks_data
        }
        
        # 保存到文件
        os.makedirs(os.path.dirname(self._get_pools_path()), exist_ok=True)
        with open(self._get_pools_path(), 'w', encoding='utf-8') as f:
            json.dump(pools_data, f, indent=2, ensure_ascii=False)
        
        print(f"试次池已保存: {self._get_pools_path()}")
        return pools_data
    
    def get_phase_pools(self, phase):
        """获取指定阶段的试次池"""
        pools_data = self.build_or_load_pools()
        return pools_data.get(phase.upper(), [])
    
    def get_block_trials(self, phase, block_num):
        """获取指定阶段和块的试次"""
        phase_pools = self.get_phase_pools(phase)
        if block_num <= 0 or block_num > len(phase_pools):
            return []
        
        # 将 [p1, p2] 格式转换为 {'P1': p1, 'P2': p2} 格式
        trials = []
        for pair in phase_pools[block_num - 1]:  # 块编号从1开始，数组索引从0开始
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                trials.append({'P1': int(pair[0]), 'P2': int(pair[1])})
            elif isinstance(pair, dict) and 'P1' in pair and 'P2' in pair:
                trials.append(pair)
            else:
                print(f"警告: 无效的试次格式: {pair}")
        
        return trials


def create_pools_manager(config, subject_id, script_dir):
    """创建试次池管理器的工厂函数"""
    return TrialPoolsManager(config, subject_id, script_dir)
