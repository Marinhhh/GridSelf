#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片批量处理工具
功能：
1. 统一图片尺寸为224x224
2. 转换为灰度图
3. 标准化亮度和对比度
4. 居中背景处理
"""

import os
import sys
from pathlib import Path

def install_requirements():
    """安装必要的依赖"""
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageEnhance
        return True
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行: pip install opencv-python pillow numpy")
        return False

def process_single_image(input_path, output_path, target_size=(360, 360)):
    """处理单张图片"""
    try:
        # 使用PIL处理图片
        from PIL import Image, ImageEnhance
        
        # 打开图片
        with Image.open(input_path) as img:
            # 转换为灰度图
            gray_img = img.convert('L')
            
            # 计算缩放比例，保持宽高比
            original_width, original_height = gray_img.size
            target_width, target_height = target_size
            
            # 计算缩放比例
            scale = min(target_width / original_width, target_height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 缩放图片
            resized_img = gray_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 创建目标尺寸的背景（灰色背景）
            background = Image.new('L', target_size, 128)  # 灰色背景
            
            # 计算居中位置
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # 将图片粘贴到背景上
            background.paste(resized_img, (x_offset, y_offset))
            
            # 调整对比度
            enhancer = ImageEnhance.Contrast(background)
            contrast_img = enhancer.enhance(1.1)
            
            # 调整亮度
            brightness_enhancer = ImageEnhance.Brightness(contrast_img)
            final_img = brightness_enhancer.enhance(1.05)
            
            # 保存图片
            final_img.save(output_path, 'JPEG', quality=95)
            
        return True
        
    except Exception as e:
        print(f"处理图片 {input_path} 时出错: {e}")
        return False

def batch_process():
    """批量处理图片"""
    # 检查依赖
    if not install_requirements():
        return False
    
    input_dir = Path("D:/桌面/网格/experiment/stimuli/stim_input")
    output_dir = Path("D:/桌面/网格/experiment/stimuli/stim_output")
    
    # 检查输入目录
    if not input_dir.exists():
        print(f"错误: 输入目录 {input_dir} 不存在")
        return False
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("没有找到图片文件")
        return False
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    success_count = 0
    error_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        output_file = output_dir / image_file.name
        
        print(f"[{i}/{len(image_files)}] 处理: {image_file.name}")
        
        if process_single_image(image_file, output_file):
            print(f"  ✓ 成功")
            success_count += 1
        else:
            print(f"  ✗ 失败")
            error_count += 1
    
    print(f"\n处理完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")
    print(f"总计: {len(image_files)} 个文件")
    
    return error_count == 0

def main():
    """主函数"""
    print("=" * 50)
    print("图片批量处理工具")
    print("=" * 50)
    
    # 确认操作
    try:
        confirm = input("确认要开始批量处理吗? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes', '是']:
            print("操作已取消")
            return
    except:
        # 如果无法获取输入，直接执行
        pass
    
    # 执行批量处理
    success = batch_process()
    
    if success:
        print("\n所有图片处理成功!")
    else:
        print("\n部分图片处理失败")

if __name__ == "__main__":
    main()
