#!/usr/bin/env python3

import argparse, os, json, tqdm, multiprocessing
import tiledwebmaps as twm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import imageio.v2 as imageio
import tinypl as pl
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True, help="输入数据集路径")
parser.add_argument("--output_path", type=str, required=True, help="输出瓦片路径")
parser.add_argument("--geojson", type=str, required=True, help="GeoJSON文件路径")
parser.add_argument("--shape", type=int, default=256, help="瓦片像素尺寸")
parser.add_argument("--workers", type=int, default=8, help="并行处理线程数")
parser.add_argument("--image_field", type=str, default="id", help="GeoJSON中图像文件名字段")
parser.add_argument("--extent", type=float, default=0.0001, help="每个点图像的覆盖范围（度）")
args = parser.parse_args()

# 创建输出目录
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# 设置瓦片参数
tile_shape = (args.shape, args.shape)
# 瓦片的地理尺寸（度）- 根据需要调整
tile_size_degrees = 0.01  # 约1km（在赤道附近）
tile_shape_crs = [tile_size_degrees, tile_size_degrees]

# 创建布局配置
layout = twm.Layout(
    crs=twm.proj.CRS("epsg:4326"),
    tile_shape_px=tile_shape,
    tile_shape_crs=tile_shape_crs,
    tile_axes=twm.geo.CompassAxes("east", "north"),
)

# 保存布局配置
layout_yaml = {
    "crs": "epsg:4326",
    "tile_shape_px": [tile_shape[0], tile_shape[1]],
    "tile_shape_crs": tile_shape_crs,
    "tile_axes": ["east", "north"],
    "path": "{zoom}/{x}/{y}.jpg",
    "min_zoom": 0,
    "max_zoom": 0,
}
with open(os.path.join(args.output_path, "layout.yaml"), "w") as f:
    yaml.dump(layout_yaml, f, default_flow_style=False)

def load_geojson_data(geojson_path):
    """加载GeoJSON文件并创建文件名到地理信息的映射"""
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    image_geo_map = {}
    
    for feature in geojson_data['features']:
        # 获取图像文件名
        properties = feature.get('properties', {})
        image_name = properties.get(args.image_field)
        
        if not image_name:
            continue
        
        # 获取几何信息
        geometry = feature.get('geometry', {})
        if geometry.get('type') == 'Polygon':
            # 对于多边形，获取边界框
            coordinates = geometry['coordinates'][0]  # 外环
            lons = [coord[0] for coord in coordinates]
            lats = [coord[1] for coord in coordinates]
            
            bounds = {
                'min_lon': min(lons),
                'max_lon': max(lons),
                'min_lat': min(lats),
                'max_lat': max(lats),
                'center_lon': (min(lons) + max(lons)) / 2,
                'center_lat': (min(lats) + max(lats)) / 2
            }
            
        elif geometry.get('type') == 'Point':
            # 对于点，使用指定的覆盖范围
            lon, lat = geometry['coordinates']
            extent = args.extent / 2  # 半径
            bounds = {
                'min_lon': lon - extent,
                'max_lon': lon + extent,
                'min_lat': lat - extent,
                'max_lat': lat + extent,
                'center_lon': lon,
                'center_lat': lat
            }
        else:
            print(f"不支持的几何类型: {geometry.get('type')}")
            continue
        
        # 添加其他属性
        bounds['properties'] = properties
        image_geo_map[image_name] = bounds
    
    return image_geo_map

def get_png_files(input_path):
    """扫描输入目录中的所有PNG文件"""
    png_files = []
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    
    return png_files

def get_image_bounds_from_geojson(image_path, geo_map):
    """从GeoJSON映射中获取图像的地理边界"""
    image_name = os.path.basename(image_path)
    
    # 尝试完整文件名匹配（不带扩展名）
    name_without_ext = os.path.splitext(image_name)[0]
    if name_without_ext in geo_map:
        return geo_map[name_without_ext]
    
    # 尝试完整文件名匹配
    if image_name in geo_map:
        return geo_map[image_name]
    
    # 尝试只用数字ID匹配（如果文件名包含数字）
    import re
    numbers = re.findall(r'\d+', name_without_ext)
    for num in numbers:
        if num in geo_map:
            return geo_map[num]
    
    # 尝试模糊匹配
    for key in geo_map.keys():
        if name_without_ext in str(key) or str(key) in name_without_ext:
            return geo_map[key]
    
    return None

lock = multiprocessing.Lock()
processed_count = 0

def process_image(args_tuple):
    """处理单个PNG文件"""
    global processed_count
    image_file, geo_map = args_tuple
    
    try:
        # 获取地理边界
        bounds = get_image_bounds_from_geojson(image_file, geo_map)
        if bounds is None:
            print(f"无法找到 {os.path.basename(image_file)} 的地理信息，跳过")
            return
        
        # 读取PNG图像
        image = imageio.imread(image_file)
        if len(image.shape) == 4:  # RGBA
            # 处理透明通道，将透明像素设为白色
            alpha = image[:, :, 3]
            rgb = image[:, :, :3]
            # 创建白色背景
            white_bg = 255 * (alpha[:, :, None] == 0)
            image = rgb * (alpha[:, :, None] / 255.0) + white_bg * (1 - alpha[:, :, None] / 255.0)
            image = image.astype('uint8')
        elif len(image.shape) == 3:
            image = image[:, :, :3]  # 只保留RGB通道
        
        # 使用中心点进行瓦片生成
        center = [bounds['center_lon'], bounds['center_lat']]
        
        # 计算分割数量（根据图像尺寸和覆盖范围）
        img_height, img_width = image.shape[:2]
        lon_range = bounds['max_lon'] - bounds['min_lon']
        lat_range = bounds['max_lat'] - bounds['min_lat']
        
        # 估算分割数量（每个瓦片大约覆盖tile_size_degrees度）
        partition_x = max(1, int(lon_range / tile_size_degrees))
        partition_y = max(1, int(lat_range / tile_size_degrees))
        partition = max(partition_x, partition_y, 1)
        
        # 生成瓦片
        tile_count = 0
        for tile_image, tile_coord in twm.util.to_tiles(image, center, layout, partition):
            # 创建输出目录
            tile_dir = os.path.join(args.output_path, "0", str(tile_coord[0]))
            if not os.path.exists(tile_dir):
                with lock:
                    if not os.path.exists(tile_dir):
                        os.makedirs(tile_dir)
            
            # 保存瓦片
            tile_path = os.path.join(tile_dir, f"{tile_coord[1]}.jpg")
            imageio.imwrite(tile_path, tile_image, quality=95)
            tile_count += 1
        
        with lock:
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"已处理 {processed_count} 个文件...")
    
    except Exception as e:
        print(f"处理 {image_file} 时出错: {e}")

def main():
    print("加载GeoJSON文件...")
    try:
        geo_map = load_geojson_data(args.geojson)
        print(f"从GeoJSON中加载了 {len(geo_map)} 个地理信息记录")
    except Exception as e:
        print(f"加载GeoJSON文件失败: {e}")
        return
    
    # 获取所有PNG文件
    png_files = get_png_files(args.input_path)
    
    if not png_files:
        print("在输入目录中未找到PNG文件")
        return
    
    print(f"找到 {len(png_files)} 个PNG文件")
    
    # 检查有多少文件有对应的地理信息
    matched_count = 0
    for png_file in png_files:
        if get_image_bounds_from_geojson(png_file, geo_map) is not None:
            matched_count += 1
    
    print(f"其中 {matched_count} 个文件在GeoJSON中找到了对应的地理信息")
    
    if matched_count == 0:
        print("警告: 没有PNG文件找到对应的地理信息!")
        print("请检查:")
        print("1. GeoJSON文件中的文件名字段是否正确")
        print("2. 文件名是否匹配")
        print(f"PNG文件示例: {png_files[:3]}")
        print(f"GeoJSON键示例: {list(geo_map.keys())[:3]}")
        return
    
    # 准备处理参数
    process_args = [(png_file, geo_map) for png_file in png_files]
    
    # 并行处理所有PNG文件
    pipe = process_args
    pipe = pl.process.map(pipe, process_image, workers=args.workers)
    
    print("开始处理图像...")
    for _ in tqdm.tqdm(pipe, total=len(png_files), desc="处理PNG文件"):
        pass
    
    print(f"成功处理了 {processed_count} 个文件")
    
    # 生成多级缩放瓦片
    print("生成多级缩放瓦片...")
    try:
        twm.util.add_zooms(args.output_path, workers=args.workers)
        print("转换完成！")
    except Exception as e:
        print(f"生成多级缩放时出错: {e}")

if __name__ == "__main__":
    main()
