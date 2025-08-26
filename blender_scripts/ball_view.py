# A simple script that uses blender to render views of a single object by randomly positioning the camera within a radius range.
# Also produces depth map at the same time.

import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np
import random
from math import radians, cos, sin, pi
         
DEBUG = False
            
VIEWS = 360
RESOLUTION = 400
RESULTS_PATH = 'pic_gen3_ballview'
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'

# 相机半径范围 [r1, r2]
RADIUS_MIN = 3.0  # 最小半径
RADIUS_MAX = 10.0  # 最大半径

import os
print("当前工作目录:", os.getcwd())

fp = bpy.path.abspath(f"//{RESULTS_PATH}")

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def generate_random_spherical_position(radius_min, radius_max):
    """
    在指定半径范围内生成随机球面坐标
    返回 (x, y, z) 坐标
    """
    # 随机生成半径
    radius = random.uniform(radius_min, radius_max)
    
    # 随机生成球面坐标角度
    # theta: 0 到 2π (水平角度)
    # phi: 0 到 π (垂直角度，0为顶部，π为底部)
    theta = random.uniform(0, 2 * pi)
    phi = random.uniform(0, pi)
    
    # 转换为笛卡尔坐标
    x = radius * sin(phi) * cos(theta)
    y = radius * sin(phi) * sin(theta)
    z = radius * cos(phi)
    
    return (x, y, z)

if not os.path.exists(fp):
    os.makedirs(fp)

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}

# Render Optimizations
bpy.context.scene.world.use_nodes = False  # 禁用节点
bpy.context.scene.world.color = (0, 0, 0)  # 设为黑色

# Set up rendering of depth map.
bpy.context.scene.use_nodes = False

tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if 'Custom Outputs' not in tree.nodes:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.label = 'Custom Outputs'
    render_layers.name = 'Custom Outputs'

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    if FORMAT == 'OPEN_EXR':
      links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
      # Remap as other types can not represent the full range of depth.
      map = tree.nodes.new(type="CompositorNodeMapRange")
      # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
      map.inputs['From Min'].default_value = 0
      map.inputs['From Max'].default_value = 8
      map.inputs['To Min'].default_value = 1
      map.inputs['To Max'].default_value = 0
      links.new(render_layers.outputs['Depth'], map.inputs[0])

      links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.name = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = False

# Create collection for objects not to render with background

    
#objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
#bpy.ops.object.delete({"selected_objects": objs})

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

cam = scene.objects['Camera']
cam.location = (0, 4.0, 0.5)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

if not DEBUG:
    for output_node in [tree.nodes['Depth Output'], tree.nodes['Normal Output']]:
        output_node.base_path = ''

out_data['frames'] = []

# 设置随机种子以确保可重复性
random.seed(42)

for i in range(0, VIEWS):
    if DEBUG:
        # 在DEBUG模式下使用固定的测试位置
        cam_pos = (0, 4.0, 0.5)
    else:
        # 生成随机球面位置
        cam_pos = generate_random_spherical_position(RADIUS_MIN, RADIUS_MAX)
    
    # 设置相机位置
    cam.location = cam_pos
    
    print("View {}, Camera position: {}".format(i, cam_pos))
    scene.render.filepath = fp + '/r_' + str(i)

    tree.nodes['Depth Output'].file_slots[0].path = scene.render.filepath + "_depth_"
    tree.nodes['Normal Output'].file_slots[0].path = scene.render.filepath + "_normal_"

    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still

    frame_data = {
        'file_path': 'r_' + str(i),
        # 'camera_position': list(cam_pos),
        'transform_matrix': listify_matrix(cam.matrix_world)
    }
    out_data['frames'].append(frame_data)

if not DEBUG:
    with open(fp + '/' + 'transforms_train.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)
