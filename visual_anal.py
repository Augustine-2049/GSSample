from plyfile import PlyData, PlyElement
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import moderngl
import sys
from pathlib import Path
import glob
import cv2
import time
import matplotlib.pyplot as plt

# 动态获取项目根目录（假设 point_render.py 在 glsl/ 子目录下）
ROOT_DIR = Path(__file__).parent  # 向上两级到 GSSample/
sys.path.append(str(ROOT_DIR))  # 将根目录加入 Python 路径
mode = "GPU"
FPS = 10
NUM_POINTS_TO_KEEP = 20_000_000
NUM_SAMPLES_PER_POINT = 16384
USE_HALF_PRECISION = False
version = 0.4

# 导入log系统，设置日志输出位置
import logging
# 1. 获取一个logger对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # 设置logger的级别

# 2. 避免重复添加handler
if not logger.handlers:
    # 3. 创建一个文件handler，明确指定编码
    log_file_path = f'point_render_{mode}.log'
    # 'w' 表示每次运行都覆盖旧日志, 如果想追加, 可以用 'a'
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8') 
    file_handler.setLevel(logging.INFO) # 设置handler的级别

    # 4. 创建一个日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 5. 将handler添加到logger
    logger.addHandler(file_handler)


def analyze(
    data_dir=".", 
    output_dir=".", 
    num_points_to_keep=1_000_000, 
    opacity_threshold=0.01,
    img_width=100, 
    img_height=100,
    fps=10,
    delete_images=True,
    point_size=2,
    logger=None
):
    from point_render_pipeline import zip_point_cloud
    points = zip_point_cloud(data_dir, num_points_to_keep, opacity_threshold=opacity_threshold).elements[0].data
    colors = np.column_stack([points['red'], points['green'], points['blue']])
    scales = np.column_stack([points['scale_0'], points['scale_1'], points['scale_2']])
    rotates = np.column_stack([points['rot_0'], points['rot_1'], points['rot_2'], points['rot_3']])
    points_world = np.column_stack([points['x'], points['y'], points['z']])
    opacity = points['opacity']
    
    # 检查scale字段是否存在
    print(f"Available fields: {points.dtype.names}")
    
    # 计算scale - 检查不同的可能字段名
    if 'scale_0' in points.dtype.names and 'scale_1' in points.dtype.names and 'scale_2' in points.dtype.names:
        scale = points['scale_0'] * points['scale_1'] * points['scale_2']
        print("Using scale_0 * scale_1 * scale_2")
    elif 'scale' in points.dtype.names:
        scale = points['scale']
        print("Using scale field")
    else:
        # 尝试其他可能的字段名
        scale_fields = [name for name in points.dtype.names if 'scale' in name.lower()]
        if scale_fields:
            print(f"Found scale related fields: {scale_fields}")
            scale = points[scale_fields[0]]  # 使用第一个找到的scale字段
        else:
            print("Scale field not found, using default value")
            scale = np.ones(len(points))
    
    # 计算统计信息
    scale_mean = np.mean(scale)
    scale_min = np.min(scale)
    scale_max = np.max(scale)
    scale_std = np.std(scale)
    
    opacity_mean = np.mean(opacity)
    opacity_min = np.min(opacity)
    opacity_max = np.max(opacity)
    opacity_std = np.std(opacity)
    
    print(f"=== Scale Statistics ===")
    print(f"Mean: {scale_mean:.6f}")
    print(f"Min: {scale_min:.6f}")
    print(f"Max: {scale_max:.6f}")
    print(f"Std: {scale_std:.6f}")
    print(f"Median: {np.median(scale):.6f}")
    print(f"Non-zero count: {np.sum(scale > 0)}")
    print(f"Zero count: {np.sum(scale == 0)}")
    
    print(f"\n=== Opacity Statistics ===")
    print(f"Mean: {opacity_mean:.6f}")
    print(f"Min: {opacity_min:.6f}")
    print(f"Max: {opacity_max:.6f}")
    print(f"Std: {opacity_std:.6f}")
    
    # 可视化scale分布 - 使用数量级作为横轴
    plt.figure(figsize=(12, 8))
    
    # 过滤掉零值和负值
    scale_positive = scale[scale > 0]
    
    if len(scale_positive) > 0:
        # 计算数量级（以10为底的对数）
        scale_log = np.log10(scale_positive)
        
        # 只显示1E-3到1E-10之间的分布
        scale_filtered = scale_positive[(scale_positive >= 1e-10) & (scale_positive <= 1e-3)]
        scale_log_filtered = np.log10(scale_filtered)
        
        print(f"Filtered scale range: {np.min(scale_filtered):.2e} to {np.max(scale_filtered):.2e}")
        print(f"Filtered magnitude range: {np.min(scale_log_filtered):.1f} to {np.max(scale_log_filtered):.1f}")
        
        # 主图：数量级分布直方图
        plt.subplot(2, 2, 1)
        if len(scale_filtered) > 0:
            # 使用数量级作为横轴
            plt.hist(scale_log_filtered, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel('Scale Magnitude (log10)')
            plt.ylabel('Point Count')
            plt.title(f'Scale Magnitude Distribution (1E-10 to 1E-3)\nTotal Points: {len(scale_filtered):,}')
            plt.grid(True, alpha=0.3)
            
            # 设置x轴刻度为数量级
            min_mag = int(np.floor(np.min(scale_log_filtered)))
            max_mag = int(np.ceil(np.max(scale_log_filtered)))
            plt.xticks(range(min_mag, max_mag + 1), [f'1E{mag}' for mag in range(min_mag, max_mag + 1)])
        else:
            plt.text(0.5, 0.5, 'No data in range 1E-10 to 1E-3', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Scale Distribution (No Data in Range)')
        
        # 对比图：原始scale值分布（对数横轴）
        plt.subplot(2, 2, 2)
        if len(scale_filtered) > 0:
            plt.hist(scale_filtered, bins=30, alpha=0.7, color='red', edgecolor='black')
            plt.xscale('log')
            plt.xlabel('Scale Value (log scale)')
            plt.ylabel('Point Count')
            plt.title(f'Scale Value Distribution (1E-10 to 1E-3)\nTotal Points: {len(scale_filtered):,}')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No data in range 1E-10 to 1E-3', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Scale Distribution (No Data in Range)')
    
    # 可视化opacity分布 - 0-1范围
    plt.subplot(2, 2, 3)
    print(opacity[:10])
    plt.hist(opacity, bins=100, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Opacity')
    plt.ylabel('Frequency')
    plt.title(f'Opacity Distribution (0-1)\nMean: {opacity_mean:.6f}, Min: {opacity_min:.6f}, Max: {opacity_max:.6f}')
    plt.grid(True, alpha=0.3)
    
    # 可视化opacity分布 - 0-0.1范围（放大查看低值区域）
    plt.subplot(2, 2, 4)
    opacity_low = opacity[opacity <= 0.1]
    plt.hist(opacity_low, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Opacity (0-0.1)')
    plt.ylabel('Frequency')
    plt.title(f'Opacity Distribution (0-0.1)\nCount: {len(opacity_low)}/{len(opacity)} ({len(opacity_low)/len(opacity)*100:.1f}%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 单独保存scale数量级分布图
    plt.figure(figsize=(12, 6))
    if len(scale_filtered) > 0:
        # 主图：数量级分布
        plt.subplot(1, 2, 1)
        plt.hist(scale_log_filtered, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Scale Magnitude (log10)')
        plt.ylabel('Point Count')
        plt.title(f'Scale Magnitude Distribution (1E-10 to 1E-3)\nTotal Points: {len(scale_filtered):,}')
        plt.grid(True, alpha=0.3)
        
        # 设置x轴刻度为数量级
        min_mag = int(np.floor(np.min(scale_log_filtered)))
        max_mag = int(np.ceil(np.max(scale_log_filtered)))
        plt.xticks(range(min_mag, max_mag + 1), [f'1E{mag}' for mag in range(min_mag, max_mag + 1)])
        
        # 对比图：原始值分布
        plt.subplot(1, 2, 2)
        plt.hist(scale_filtered, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.xscale('log')
        plt.xlabel('Scale Value (log scale)')
        plt.ylabel('Point Count')
        plt.title(f'Scale Value Distribution (1E-10 to 1E-3)\nTotal Points: {len(scale_filtered):,}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'No data in range 1E-10 to 1E-3', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Scale Distribution (No Data in Range)')
    
    plt.savefig(os.path.join(output_dir, "scale_magnitude_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 单独保存opacity分布（0-0.1范围）
    plt.figure(figsize=(10, 6))
    plt.hist(opacity_low, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Opacity (0-0.1)')
    plt.ylabel('Frequency')
    plt.title(f'Opacity Distribution (0-0.1)\nCount: {len(opacity_low)}/{len(opacity)} ({len(opacity_low)/len(opacity)*100:.1f}%)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "opacity_hist_0_0.1.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存统计信息到文本文件
    stats_file = os.path.join(output_dir, "statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=== Point Cloud Data Statistics ===\n\n")
        f.write(f"Total points: {len(points)}\n\n")
        
        f.write("=== Scale Statistics ===\n")
        f.write(f"Mean: {scale_mean:.6f}\n")
        f.write(f"Min: {scale_min:.6f}\n")
        f.write(f"Max: {scale_max:.6f}\n")
        f.write(f"Std: {scale_std:.6f}\n")
        f.write(f"Median: {np.median(scale):.6f}\n")
        f.write(f"25th percentile: {np.percentile(scale, 25):.6f}\n")
        f.write(f"75th percentile: {np.percentile(scale, 75):.6f}\n")
        f.write(f"Non-zero count: {np.sum(scale > 0)}\n")
        f.write(f"Zero count: {np.sum(scale == 0)}\n")
        if len(scale_filtered) > 0:
            f.write(f"1E-10 to 1E-3 range mean: {np.mean(scale_filtered):.6f}\n")
            f.write(f"1E-10 to 1E-3 range min: {np.min(scale_filtered):.6f}\n")
            f.write(f"1E-10 to 1E-3 range max: {np.max(scale_filtered):.6f}\n")
            f.write(f"1E-10 to 1E-3 range count: {len(scale_filtered)} ({len(scale_filtered)/len(scale_positive)*100:.2f}%)\n")
        f.write("\n")
        
        f.write("=== Opacity Statistics ===\n")
        f.write(f"Mean: {opacity_mean:.6f}\n")
        f.write(f"Min: {opacity_min:.6f}\n")
        f.write(f"Max: {opacity_max:.6f}\n")
        f.write(f"Std: {opacity_std:.6f}\n")
        f.write(f"Median: {np.median(opacity):.6f}\n")
        f.write(f"25th percentile: {np.percentile(opacity, 25):.6f}\n")
        f.write(f"75th percentile: {np.percentile(opacity, 75):.6f}\n\n")
        
        f.write("=== Opacity Distribution Details ===\n")
        f.write(f"Opacity <= 0.01: {np.sum(opacity <= 0.01)} points ({np.sum(opacity <= 0.01)/len(opacity)*100:.2f}%)\n")
        f.write(f"Opacity <= 0.05: {np.sum(opacity <= 0.05)} points ({np.sum(opacity <= 0.05)/len(opacity)*100:.2f}%)\n")
        f.write(f"Opacity <= 0.1: {np.sum(opacity <= 0.1)} points ({np.sum(opacity <= 0.1)/len(opacity)*100:.2f}%)\n")
        f.write(f"Opacity <= 0.5: {np.sum(opacity <= 0.5)} points ({np.sum(opacity <= 0.5)/len(opacity)*100:.2f}%)\n")
        f.write(f"Opacity > 0.5: {np.sum(opacity > 0.5)} points ({np.sum(opacity > 0.5)/len(opacity)*100:.2f}%)\n")
    
    # 额外分析：如果存在单独的scale字段，也进行分析
    if 'scale_0' in points.dtype.names and 'scale_1' in points.dtype.names and 'scale_2' in points.dtype.names:
        print(f"\n=== Individual Scale Field Analysis ===")
        scale_0 = points['scale_0']
        scale_1 = points['scale_1'] 
        scale_2 = points['scale_2']
        
        print(f"Scale_0 - Mean: {np.mean(scale_0):.6f}, Range: [{np.min(scale_0):.6f}, {np.max(scale_0):.6f}]")
        print(f"Scale_1 - Mean: {np.mean(scale_1):.6f}, Range: [{np.min(scale_1):.6f}, {np.max(scale_1):.6f}]")
        print(f"Scale_2 - Mean: {np.mean(scale_2):.6f}, Range: [{np.min(scale_2):.6f}, {np.max(scale_2):.6f}]")
        
        # 保存单独scale字段的统计信息
        with open(stats_file, 'a', encoding='utf-8') as f:
            f.write("=== Individual Scale Field Statistics ===\n")
            f.write(f"Scale_0 - Mean: {np.mean(scale_0):.6f}, Min: {np.min(scale_0):.6f}, Max: {np.max(scale_0):.6f}\n")
            f.write(f"Scale_1 - Mean: {np.mean(scale_1):.6f}, Min: {np.min(scale_1):.6f}, Max: {np.max(scale_1):.6f}\n")
            f.write(f"Scale_2 - Mean: {np.mean(scale_2):.6f}, Min: {np.min(scale_2):.6f}, Max: {np.max(scale_2):.6f}\n\n")
    
    # 统计scale在1E-3到1E-10范围内每个数量级的数量
    print(f"\n=== Scale Magnitude Distribution Analysis (1E-10 to 1E-3) ===")
    
    # 过滤掉零值和负值
    scale_positive = scale[scale > 0]
    
    if len(scale_positive) > 0:
        # 只分析1E-3到1E-10范围
        scale_filtered = scale_positive[(scale_positive >= 1e-10) & (scale_positive <= 1e-3)]
        
        if len(scale_filtered) > 0:
            # 计算数量级（以10为底的对数）
            scale_log = np.log10(scale_filtered)
            
            # 定义数量级范围（-10到-3）
            min_log = -10
            max_log = -3
            
            print(f"Analysis range: 1E-10 to 1E-3")
            print(f"Magnitude range: {min_log} to {max_log}")
            print(f"Total analysis points: {len(scale_filtered):,}")
            
            # 统计每个数量级的数量
            magnitude_counts = {}
            magnitude_ranges = {}
            
            for mag in range(min_log, max_log + 1):
                # 定义当前数量级的范围
                lower_bound = 10**mag
                upper_bound = 10**(mag + 1)
                
                # 统计在这个范围内的数量
                count = np.sum((scale_filtered >= lower_bound) & (scale_filtered < upper_bound))
                magnitude_counts[mag] = count
                magnitude_ranges[mag] = (lower_bound, upper_bound)
                
                print(f"1E{mag} to 1E{mag+1}: {count:,} points ({count/len(scale_filtered)*100:.2f}%)")
        else:
            print("No data found in range 1E-10 to 1E-3")
            magnitude_counts = {}
            magnitude_ranges = {}
        
        # 保存到txt文件
        scale_magnitude_file = os.path.join(output_dir, "scale_magnitude_analysis.txt")
        with open(scale_magnitude_file, 'w', encoding='utf-8') as f:
            f.write("=== Scale Magnitude Distribution Analysis (1E-10 to 1E-3) ===\n\n")
            f.write(f"Total points: {len(scale):,}\n")
            f.write(f"Positive points: {len(scale_positive):,}\n")
            f.write(f"Zero points: {len(scale) - len(scale_positive):,}\n")
            f.write(f"Analysis range: 1E-10 to 1E-3\n")
            f.write(f"Analysis points: {len(scale_filtered):,} ({len(scale_filtered)/len(scale_positive)*100:.2f}%)\n")
            f.write(f"Magnitude range: {min_log} to {max_log}\n\n")
            
            f.write("Magnitude distribution:\n")
            f.write("Magnitude\t\tRange\t\t\t\tCount\t\tPercentage\n")
            f.write("-" * 70 + "\n")
            
            for mag in sorted(magnitude_counts.keys()):
                lower, upper = magnitude_ranges[mag]
                count = magnitude_counts[mag]
                percentage = count / len(scale_filtered) * 100
                f.write(f"1E{mag}\t\t{lower:.2e} - {upper:.2e}\t\t{count:,}\t\t{percentage:.2f}%\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write(f"Total\t\t\t\t\t\t{len(scale_filtered):,}\t\t100.00%\n")
        
        print(f"Magnitude analysis saved to: {scale_magnitude_file}")
        
        # 可视化数量级分布
        plt.figure(figsize=(12, 6))
        
        # 柱状图
        plt.subplot(1, 2, 1)
        magnitudes = list(magnitude_counts.keys())
        counts = list(magnitude_counts.values())
        plt.bar(magnitudes, counts, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Magnitude (1E^x)')
        plt.ylabel('Point Count')
        plt.title('Scale Magnitude Distribution (1E-10 to 1E-3)')
        plt.grid(True, alpha=0.3)
        
        # 设置x轴刻度
        plt.xticks(magnitudes, [f'1E{mag}' for mag in magnitudes])
        
        # 百分比饼图
        plt.subplot(1, 2, 2)
        non_zero_counts = [count for count in counts if count > 0]
        non_zero_magnitudes = [f"1E{mag}" for mag, count in zip(magnitudes, counts) if count > 0]
        
        if len(non_zero_counts) > 0:
            plt.pie(non_zero_counts, labels=non_zero_magnitudes, autopct='%1.1f%%', startangle=90)
            plt.title('Scale Magnitude Distribution Percentage (1E-10 to 1E-3)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scale_magnitude_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Magnitude distribution chart saved to: {output_dir}/scale_magnitude_distribution.png")
    
    else:
        print("No positive scale values found")
        with open(stats_file, 'a', encoding='utf-8') as f:
            f.write("=== Scale Magnitude Distribution Analysis ===\n")
            f.write("No positive scale values found\n\n")
    
    print(f"Statistics saved to: {stats_file}")
    print(f"Visualization charts saved to: {output_dir}")




if __name__ == "__main__":
    dir = ROOT_DIR
    wh = [[100, 100], [616, 409], [1920, 1080]]
    wh_idx = 2
    analyze(
        data_dir=dir,
        output_dir=dir,
        num_points_to_keep=NUM_POINTS_TO_KEEP, # 10_000_000
        opacity_threshold=None,
        img_width=wh[wh_idx][0],
        img_height=wh[wh_idx][1],
        fps=FPS,
        delete_images=True,
        point_size=1,
        logger=logger)