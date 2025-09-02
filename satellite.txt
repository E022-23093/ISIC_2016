#!/usr/bin/env python3

import argparse, os, json, tqdm, multiprocessing
import tiledwebmaps as twm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import imageio.v2 as imageio
import tinypl as pl
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True, help="输入PNG文件目录")
parser.add_argument("--output_path", type=str, required=True, help="输出瓦片目录")
parser.add_argument("--geojson", type=str, required=True, help="GeoJSON文件路径")
parser.add_argument("--shape", type=int, default=None, help="瓦片像素尺寸（必须是图像尺寸的因数）")
parser.add_argument("--workers", type=int, default=8, help="并行处理线程数")
parser.add_argument("--image_field", type=str, default="id", help="GeoJSON中图像ID字段")
args = parser.parse_args()

# 创建输出目录
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

def load_geojson_data(geojson_path):
    """加载GeoJSON文件并创建ID到地理信息的映射"""
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    image_geo_map = {}
    
    for feature in geojson_data['features']:
        properties = feature.get('properties', {})
        image_id = str(properties.get(args.image_field, ''))
        
        if not image_id:
            continue
        
        geometry = feature.get('geometry', {})
        if geometry.get('type') == 'Point':
            lon, lat = geometry['coordinates']
            geo_info = {
                'lon': lon,
                'lat': lat,
                'properties': properties
            }
            image_geo_map[image_id] = geo_info
    
    return image_geo_map

def get_png_files(input_path):
    """获取所有PNG文件"""
    png_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    return png_files

def extract_id_from_filename(filename):
    """从文件名中提取ID"""
    import re
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    # 尝试提取数字ID
    numbers = re.findall(r'\d+', basename)
    if numbers:
        # 返回最长的数字字符串
        return max(numbers, key=len)
    
    # 如果没有数字，返回整个基础文件名
    return basename

def determine_tile_params(image_shape, target_shape):
    """根据图像尺寸和目标瓦片尺寸确定分割参数"""
    height, width = image_shape[:2]
    
    if target_shape is None:
        # 自动确定合适的瓦片尺寸
        # 目标是生成合理数量的瓦片（不超过100个）
        max_tiles = 100
        min_tile_size = 256
        
        # 计算最佳瓦片尺寸
        total_pixels = width * height
        target_tile_pixels = max(total_pixels / max_tiles, min_tile_size * min_tile_size)
        tile_size = int(np.sqrt(target_tile_pixels))
        
        # 确保瓦片尺寸是合理的
        tile_size = max(min_tile_size, min(tile_size, min(width, height)))
    else:
        tile_size = target_shape
    
    # 计算分割数量
    tiles_x = int(np.ceil(width / tile_size))
    tiles_y = int(np.ceil(height / tile_size))
    
    return tile_size, tiles_x, tiles_y

# 全局瓦片计数器和锁
tile_counter = multiprocessing.Value('i', 0)
lock = multiprocessing.Lock()

def process_single_image(args_tuple):
    """处理单个PNG文件，将其切分成瓦片并放入统一的瓦片结构中"""
    png_file, geo_map = args_tuple
    
    try:
        # 提取图像ID
        image_id = extract_id_from_filename(png_file)
        
        # 获取地理信息
        geo_info = geo_map.get(image_id, {})
        
        # 读取图像
        image = imageio.imread(png_file)
        
        # 处理透明通道
        if len(image.shape) == 4:  # RGBA
            alpha = image[:, :, 3]
            rgb = image[:, :, :3]
            # 将透明像素设为白色
            white_bg = np.ones_like(rgb) * 255
            alpha_mask = alpha[:, :, None] / 255.0
            image = (rgb * alpha_mask + white_bg * (1 - alpha_mask)).astype(np.uint8)
        elif len(image.shape) == 3:
            image = image[:, :, :3]
        
        # 确定瓦片参数
        tile_size, tiles_x, tiles_y = determine_tile_params(image.shape, args.shape)
        
        height, width = image.shape[:2]
        tiles_generated = 0
        
        # 切分图像为瓦片
        for y in range(tiles_y):
            for x in range(tiles_x):
                # 计算瓦片边界
                left = x * tile_size
                top = y * tile_size
                right = min(left + tile_size, width)
                bottom = min(top + tile_size, height)
                
                # 提取瓦片
                tile = image[top:bottom, left:right]
                
                # 如果瓦片小于标准尺寸，用白色填充
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded_tile = 255 * np.ones((tile_size, tile_size, 3), dtype=np.uint8)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                # 获取全局瓦片坐标
                with lock:
                    global_x = tile_counter.value // 10000  # 假设最多10000行
                    global_y = tile_counter.value % 10000
                    tile_counter.value += 1
                
                # 创建瓦片目录结构 (与原代码一致: zoom/x/y.jpg)
                zoom_dir = os.path.join(args.output_path, "0")  # 固定zoom级别为0
                if not os.path.exists(zoom_dir):
                    os.makedirs(zoom_dir)
                
                x_dir = os.path.join(zoom_dir, str(global_x))
                if not os.path.exists(x_dir):
                    os.makedirs(x_dir)
                
                # 保存瓦片
                tile_path = os.path.join(x_dir, f"{global_y}.jpg")
                imageio.imwrite(tile_path, tile, quality=100)
                
                tiles_generated += 1
        
        return {
            'image_id': image_id,
            'status': 'success',
            'tiles_count': tiles_generated,
            'tile_size': tile_size,
            'original_size': [width, height],
            'geo_info': geo_info
        }
    
    except Exception as e:
        print(f"处理 {png_file} 时出错: {e}")
        return {'image_id': extract_id_from_filename(png_file), 'status': 'error', 'error': str(e)}

def create_layout_config(processed_images):
    """创建与原代码一致的layout.yaml配置文件"""
    # 从处理结果中获取平均瓦片尺寸
    successful_images = [img for img in processed_images if img.get('status') == 'success']
    if not successful_images:
        return
    
    avg_tile_size = int(np.mean([img.get('tile_size', 256) for img in successful_images]))
    
    # 创建layout配置（模仿原代码）
    layout_yaml = {
        "crs": "epsg:4326",  # 使用WGS84坐标系
        "tile_shape_px": [avg_tile_size, avg_tile_size],
        "tile_shape_crs": [0.01, 0.01],  # 地理尺寸（度）
        "tile_axes": ["east", "north"],
        "path": "{zoom}/{x}/{y}.jpg",
        "min_zoom": 0,
        "max_zoom": 0,
    }
    
    # 保存layout配置
    with open(os.path.join(args.output_path, "layout.yaml"), "w") as f:
        yaml.dump(layout_yaml, f, default_flow_style=False)

def create_summary_report(processed_images):
    """创建处理摘要报告"""
    successful = [img for img in processed_images if img.get('status') == 'success']
    failed = [img for img in processed_images if img.get('status') == 'error']
    
    report = {
        'processing_summary': {
            'total_images': len(processed_images),
            'successful': len(successful),
            'failed': len(failed),
            'total_tiles': sum(img.get('tiles_count', 0) for img in successful)
        },
        'successful_images': {
            img['image_id']: {
                'tiles_count': img.get('tiles_count', 0),
                'tile_size': img.get('tile_size', 0),
                'original_size': img.get('original_size', [0, 0]),
                'has_geo_info': bool(img.get('geo_info'))
            } for img in successful
        },
        'failed_images': {
            img['image_id']: img.get('error', 'Unknown error') for img in failed
        }
    }
    
    # 保存报告
    with open(os.path.join(args.output_path, "processing_report.json"), "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report

def main():
    print("加载GeoJSON文件...")
    try:
        geo_map = load_geojson_data(args.geojson)
        print(f"从GeoJSON中加载了 {len(geo_map)} 个地理信息记录")
    except Exception as e:
        print(f"警告: 加载GeoJSON文件失败: {e}")
        geo_map = {}
    
    # 获取所有PNG文件
    png_files = get_png_files(args.input_path)
    
    if not png_files:
        print("在输入目录中未找到PNG文件")
        return
    
    print(f"找到 {len(png_files)} 个PNG文件")
    
    if args.shape:
        print(f"使用固定瓦片尺寸: {args.shape}x{args.shape}")
    else:
        print("将为每个图像自动确定最佳瓦片尺寸")
    
    # 准备处理参数
    process_args = [(png_file, geo_map) for png_file in png_files]
    
    # 并行处理所有PNG文件
    print("开始处理图像...")
    pipe = process_args
    pipe = pl.process.map(pipe, process_single_image, workers=args.workers)
    
    processed_images = []
    for result in tqdm.tqdm(pipe, total=len(png_files), desc="生成瓦片"):
        if result:
            processed_images.append(result)
    
    # 创建配置文件
    create_layout_config(processed_images)
    
    # 创建摘要报告
    report = create_summary_report(processed_images)
    
    # 生成多级缩放瓦片（与原代码一致）
    print("生成多级缩放瓦片...")
    try:
        twm.util.add_zooms(args.output_path, workers=args.workers)
    except Exception as e:
        print(f"生成多级缩放时出错: {e}")
    
    print(f"\n处理完成！")
    print(f"成功处理: {report['processing_summary']['successful']} 个图像")
    print(f"处理失败: {report['processing_summary']['failed']} 个图像")
    print(f"总瓦片数: {report['processing_summary']['total_tiles']}")
    print(f"输出目录结构:")
    print(f"  {args.output_path}/")
    print(f"  ├── layout.yaml")
    print(f"  ├── processing_report.json")
    print(f"  └── 0/")
    print(f"      ├── 0/")
    print(f"      │   ├── 0.jpg")
    print(f"      │   ├── 1.jpg")
    print(f"      │   └── ...")
    print(f"      ├── 1/")
    print(f"      └── ...")

if __name__ == "__main__":
    main()
