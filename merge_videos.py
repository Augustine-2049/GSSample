#!/usr/bin/env python3
"""
视频合并工具
将两个视频合并为左右分屏显示
"""

import cv2
import numpy as np
import os
from pathlib import Path

def estimate_video_size(width, height, fps, frame_count, bitrate_factor=0.1):
    """
    估算视频文件大小
    
    Args:
        width: 视频宽度
        height: 视频高度
        fps: 帧率
        frame_count: 帧数
        bitrate_factor: 码率因子（0.1表示较高质量，0.05表示中等质量）
    
    Returns:
        估算的文件大小（字节）
    """
    # 计算像素数
    pixels = width * height
    
    # 估算每像素的比特数（基于经验值）
    bits_per_pixel = bitrate_factor
    
    # 计算总比特数
    total_bits = pixels * bits_per_pixel * frame_count
    
    # 转换为字节
    total_bytes = total_bits / 8
    
    return total_bytes

def calculate_scale_factor(current_size, target_size):
    """
    计算需要的缩放因子
    
    Args:
        current_size: 当前估算大小（字节）
        target_size: 目标大小（字节）
    
    Returns:
        缩放因子（小于1的值）
    """
    # 视频大小与分辨率的平方成正比
    scale_factor = np.sqrt(target_size / current_size)
    
    # 确保缩放因子在合理范围内（不小于0.3）
    scale_factor = max(scale_factor, 0.3)
    
    return scale_factor

def merge_videos_side_by_side(
    video1_path, 
    video2_path, 
    output_path, 
    label1="Baseline", 
    label2="Point Render",
    scale_factor=1.0
):
    """
    将两个视频合并为左右分屏显示
    自动处理不同尺寸：将小视频扩展到与大视频相同尺寸，输出宽度为大视频的两倍
    自动控制输出视频大小在指定范围内
    
    Args:
        video1_path: 左侧视频路径
        video2_path: 右侧视频路径  
        output_path: 输出视频路径
        label1: 左侧视频标签
        label2: 右侧视频标签
        max_size_mb: 最大文件大小（MB），默认10MB
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(video1_path):
        print(f"Error: 左侧视频文件不存在: {video1_path}")
        return False
    
    if not os.path.exists(video2_path):
        print(f"Error: 右侧视频文件不存在: {video2_path}")
        return False
    
    # 打开视频文件
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened():
        print(f"Error: 无法打开左侧视频: {video1_path}")
        return False
    
    if not cap2.isOpened():
        print(f"Error: 无法打开右侧视频: {video2_path}")
        return False
    
    # 获取视频属性
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 获取视频尺寸
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"左侧视频: {fps1:.2f} FPS, {frame_count1} 帧, {width1}x{height1}")
    print(f"右侧视频: {fps2:.2f} FPS, {frame_count2} 帧, {width2}x{height2}")
    
    # 使用较高的帧率
    output_fps = max(fps1, fps2)
    # 使用较短的视频长度
    max_frames = min(frame_count1, frame_count2)
    
    # 确定目标尺寸：使用较大的视频尺寸作为基准
    target_width = max(width1, width2)
    target_height = max(height1, height2)
    
    # 输出视频尺寸：宽度为基准宽度的两倍
    output_width = target_width * 2
    output_height = target_height

    output_width = int(output_width * scale_factor)
    output_height = int(output_height * scale_factor)
    target_width = int(target_width * scale_factor)
    target_height = int(target_height * scale_factor)
    print(f"调整后尺寸: {output_width}x{output_height}")
    
    print(f"基准尺寸: {target_width}x{target_height}")
    print(f"输出尺寸: {output_width}x{output_height}")
    print(f"输出视频: {output_fps:.2f} FPS, {max_frames} 帧")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
    
    if not out.isOpened():
        print(f"Error: 无法创建输出视频: {output_path}")
        return False
    
    # 设置字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5 * scale_factor
    font_thickness = max(1, int(3 * scale_factor))
    font_color = (255, 255, 255)  # 白色文字
    bg_color = (0, 0, 0)  # 黑色背景
    
    frame_idx = 0
    
    try:
        while frame_idx < max_frames:
            # 读取帧
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # 调整帧大小到目标尺寸（保持宽高比）
            def resize_with_padding(frame, target_w, target_h):
                """调整帧大小，保持宽高比，用黑色填充"""
                h, w = frame.shape[:2]
                
                # 计算缩放比例
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # 缩放帧
                resized = cv2.resize(frame, (new_w, new_h))
                
                # 创建目标尺寸的黑色画布
                padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                # 计算居中位置
                x_offset = (target_w - new_w) // 2
                y_offset = (target_h - new_h) // 2
                
                # 将缩放后的帧放在中心
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                return padded
            
            # 调整两个帧到相同的目标尺寸
            frame1_resized = resize_with_padding(frame1, target_width, target_height)
            frame2_resized = resize_with_padding(frame2, target_width, target_height)
            
            # 合并帧
            combined_frame = np.hstack([frame1_resized, frame2_resized])
            
            # 添加标签
            # 左侧标签 - 放在左上角
            label1_size = cv2.getTextSize(label1, font, font_scale, font_thickness)[0]
            label1_x = 10
            label1_y = label1_size[1] + 10
            
            # 绘制背景矩形
            cv2.rectangle(combined_frame, 
                         (label1_x - 5, label1_y - label1_size[1] - 5),
                         (label1_x + label1_size[0] + 5, label1_y + 5),
                         bg_color, -1)
            
            # 绘制文字
            cv2.putText(combined_frame, label1, (label1_x, label1_y), 
                       font, font_scale, font_color, font_thickness)
            
            # 右侧标签 - 放在右上角
            label2_size = cv2.getTextSize(label2, font, font_scale, font_thickness)[0]
            label2_x = target_width + 10
            label2_y = label2_size[1] + 10
            
            # 绘制背景矩形
            cv2.rectangle(combined_frame, 
                         (label2_x - 5, label2_y - label2_size[1] - 5),
                         (label2_x + label2_size[0] + 5, label2_y + 5),
                         bg_color, -1)
            
            # 绘制文字
            cv2.putText(combined_frame, label2, (label2_x, label2_y), 
                       font, font_scale, font_color, font_thickness)
            
            # 添加分隔线
            cv2.line(combined_frame, (target_width, 0), (target_width, target_height), 
                    (128, 128, 128), 2)
            
            # 写入帧
            out.write(combined_frame)
            
            frame_idx += 1
            
            # 显示进度
            if frame_idx % 30 == 0:
                progress = (frame_idx / max_frames) * 100
                print(f"进度: {progress:.1f}% ({frame_idx}/{max_frames})")
    
    except Exception as e:
        print(f"Error: 处理视频时出错: {e}")
        return False
    
    finally:
        # 释放资源
        cap1.release()
        cap2.release()
        out.release()
    
    print(f"视频合并完成: {output_path}")
    
    # 检查最终文件大小
    if os.path.exists(output_path):
        final_size = os.path.getsize(output_path)
        final_size_mb = final_size / (1024 * 1024)
        print(f"最终文件大小: {final_size_mb:.2f} MB")
        
    
    return True

def main():
    """主函数"""
    
    # 视频路径
    baseline_path = "render_video_gs.mp4"  # 左侧视频 baseline
    render_path = "render_video.mp4"  # 右侧视频
    output_path = "comparison_video.mp4"  # 输出视频
    
    # 文件大小限制（MB）
    max_size_mb = 10
    
    # 检查文件是否存在
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline视频不存在: {baseline_path}")
        print("请确保baseline.mp4文件在当前目录下")
        return
    
    if not os.path.exists(render_path):
        print(f"Error: Render视频不存在: {render_path}")
        print("请确保render_video.mp4文件在当前目录下")
        return
    
    print("=== 开始合并视频 ===")
    print(f"左侧视频: {baseline_path}")
    print(f"右侧视频: {render_path}")
    print(f"输出视频: {output_path}")
    print(f"文件大小限制: {max_size_mb} MB")
    
    # 合并视频
    success = merge_videos_side_by_side(
        video1_path=baseline_path,
        video2_path=render_path,
        output_path=output_path,
        label1="Baseline",
        label2="Point Render",
        scale_factor=0.4
    )
    
    if success:
        print("=== 视频合并成功 ===")
        print(f"输出文件: {output_path}")
    else:
        print("=== 视频合并失败 ===")

if __name__ == "__main__":
    main() 