
from maroon.cuda.cuda import load_kernel_from_cu
import pycuda.driver as cuda

import math

import matplotlib
import plotly.express as px
import open3d as o3d
import os
import json
import cv2
import numpy as np


def filter_outliers(pc, kernel_size=3, threshold=pow(2, 16)):
    mod = load_kernel_from_cu(os.path.join(
        os.path.dirname(__file__), "..", "cuda", "pointcloud.cu"))

    pc_ = pc.reshape(-1).astype(np.float32)
    num_points = int(pc_.shape[0] / 3)
    valid_mask = np.zeros(num_points).astype(np.uint8)

    pc_gpu = cuda.mem_alloc(pc_.nbytes)
    valid_mask_gpu = cuda.mem_alloc(valid_mask.nbytes)

    cuda.memcpy_htod(pc_gpu, pc_)
    cuda.memcpy_htod(valid_mask_gpu, valid_mask)

    func = mod.get_function("outlier_filtering")
    block_size = (32, 32, 1)
    grid_size = (int(math.ceil(
        pc.shape[1] / block_size[0])), int(math.ceil(pc.shape[0] / block_size[1])), 1)

    func(pc_gpu, np.int32(pc.shape[1]), np.int32(pc.shape[0]), np.int32(kernel_size), np.float32(threshold), valid_mask_gpu,
         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(valid_mask, valid_mask_gpu)
    valid_mask = valid_mask.reshape(pc.shape[:2])
    pc_gpu.free()
    valid_mask_gpu.free()

    return valid_mask


def triangulate_image_pointcloud(pc, threshold=pow(2, 16)):
    mod = load_kernel_from_cu(os.path.join(
        os.path.dirname(__file__), "..", "cuda", "pointcloud.cu"))

    pc_ = pc.reshape(-1).astype(np.float32)
    num_points = (pc_.shape[0] / 3)
    num_quads = (pc.shape[0]-1) * (pc.shape[1]-1)
    # maximum of 2 triangles per quad
    index_list = np.zeros((2*3*num_quads), dtype=np.int32)

    pc_gpu = cuda.mem_alloc(pc_.nbytes)
    list_gpu = cuda.mem_alloc(index_list.nbytes)

    cuda.memcpy_htod(pc_gpu, pc_)
    cuda.memcpy_htod(list_gpu, index_list)

    func = mod.get_function("triangulate")
    block_size = (32, 32, 1)
    grid_size = (int(math.ceil(
        pc.shape[1] / block_size[0])), int(math.ceil(pc.shape[0] / block_size[1])), 1)

    func(pc_gpu, np.int32(pc.shape[1]), np.int32(pc.shape[0]), np.float32(threshold), list_gpu,
         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(index_list, list_gpu)
    pc_gpu.free()
    list_gpu.free()

    index_list = index_list.reshape(num_quads*2, 3)
    # remove invalid index entries
    index_list = index_list[index_list[:, 0] != 0]
    index_list = index_list - 1  # shift entries by 1 to get true index

    return pc.reshape(-1, 3), index_list


def filter_depth(depth,  bf_stddev_space=-1, bf_stddev_range=-1):
    mod = load_kernel_from_cu(os.path.join(
        os.path.dirname(__file__), "..", "cuda", "pointcloud.cu"))

    depth_cpu = depth.reshape(-1, 3).astype(np.float32)
    depth_out_cpu = np.zeros_like(depth_cpu)

    depth_gpu = cuda.mem_alloc(depth_cpu.nbytes)
    depth_out_gpu = cuda.mem_alloc(depth_out_cpu.nbytes)

    cuda.memcpy_htod(depth_out_gpu, depth_out_cpu)
    cuda.memcpy_htod(depth_gpu, depth_cpu)

    func = mod.get_function("bilateral_filtering")

    block_size = (32, 32, 1)  # 1 warp per point, 4 warps pers block
    grid_size = (
        int(math.ceil(depth.shape[1] / block_size[0])), math.ceil(depth.shape[0] / block_size[1]), 1)

    func(depth_gpu, np.int32(depth.shape[1]), np.int32(depth.shape[0]), np.float32(bf_stddev_space),
         np.float32(bf_stddev_range), depth_out_gpu,
         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(depth_out_cpu, depth_out_gpu)
    depth_out_cpu = depth_out_cpu.reshape(
        (depth.shape[0], depth.shape[1])).astype(np.uint16)

    depth_out_gpu.free()
    depth_gpu.free()

    return depth_out_cpu


def project(p1, proj, height, width, color_attribute=None, mask=None):
    assert proj.shape[0] == 4 and proj.shape[1] == 4
    assert p1.shape[1] == 3
    if color_attribute is not None:
        assert color_attribute.shape[0] == p1.shape[0]

    mod = load_kernel_from_cu(os.path.join(
        os.path.dirname(__file__), "..", "cuda", "metrics.cu"))
    depth_out = np.zeros((height, width)).astype(np.float32).reshape(-1)

    points_cpu = p1.reshape(-1, 3)
    ones = np.ones((points_cpu.shape[0], 1))

    mask_cpu = np.ones((height, width)).astype(np.uint8).reshape(-1)
    if mask is not None:
        mask_cpu = mask.reshape(-1).astype(np.uint8).reshape(-1)
    print("MASK: ", np.unique(mask_cpu))

    points_cpu = np.concatenate(
        [points_cpu, ones], axis=-1).reshape(-1).astype(np.float32)
    points_gpu = cuda.mem_alloc(points_cpu.nbytes)
    depth_gpu = cuda.mem_alloc(depth_out.nbytes)
    mask_gpu = cuda.mem_alloc(mask_cpu.nbytes)

    cuda.memcpy_htod(points_gpu, points_cpu)
    cuda.memcpy_htod(depth_gpu, depth_out)
    cuda.memcpy_htod(mask_gpu, mask_cpu)
    addr_proj = mod.get_global("project")

    # move to __const__ memory
    proj_linearized = proj.reshape(-1).astype(np.float32).copy()
    cuda.memcpy_htod(addr_proj[0], proj_linearized)

    block_size = (32, 1, 1)  # 1 warp per point, 4 warps pers block
    grid_size = (
        int(math.ceil(p1.reshape(-1, 3).shape[0] / block_size[0])), 1, 1)

    if color_attribute is not None:
        color = color_attribute.reshape(-1).astype(np.float32)
        color_out = np.zeros((height, width, 3)).astype(np.float32).reshape(-1)
        color_gpu = cuda.mem_alloc(color.nbytes)
        color_out_gpu = cuda.mem_alloc(color_out.nbytes)
        cuda.memcpy_htod(color_gpu, color)

        func = mod.get_function("project_depth_color")
        func(points_gpu, np.int32(p1.reshape(-1, 3).shape[0]), color_gpu, np.int32(width), np.int32(height), mask_gpu, depth_gpu, color_out_gpu,
             block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(depth_out, depth_gpu)
        cuda.memcpy_dtoh(color_out, color_out_gpu)
        depth_out = depth_out.reshape(height, width)
        color_out = color_out.reshape(height, width, 3)

        points_gpu.free()
        depth_gpu.free()
        color_out_gpu.free()
        color_gpu.free()
        mask_gpu.free()

        return depth_out, color_out

    else:
        func = mod.get_function("project_depth")
        func(points_gpu, np.int32(p1.reshape(-1, 3).shape[0]), np.int32(width), np.int32(height), mask_gpu, depth_gpu,
             block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(depth_out, depth_gpu)
        depth_out = depth_out.reshape(height, width)

        points_gpu.free()
        depth_gpu.free()
        mask_gpu.free()

        return depth_out


def project_interpolate(src_points, src_indices, proj, height, width, src_color_attribute=None):
    assert proj.shape[0] == 4 and proj.shape[1] == 4
    assert src_points.shape[1] == 3
    if src_color_attribute is not None:
        assert src_points.shape[0] == src_color_attribute.shape[0]

    mod = load_kernel_from_cu(os.path.join(
        os.path.dirname(__file__), "..", "cuda", "metrics.cu"))

    pixels_out = np.zeros((src_points.shape[0], 2)).astype(
        np.float32).reshape(-1)

    # --------------------- COMPUTE PIXELS ------------------
    points_cpu = src_points.reshape(-1, 3)
    ones = np.ones((points_cpu.shape[0], 1))

    points_cpu = np.concatenate(
        [points_cpu, ones], axis=-1).reshape(-1).astype(np.float32)
    points_gpu = cuda.mem_alloc(points_cpu.nbytes)
    pixels_gpu = cuda.mem_alloc(pixels_out.nbytes)

    cuda.memcpy_htod(points_gpu, points_cpu)
    addr_proj = mod.get_global("project")

    # move to __const__ memory
    proj_linearized = proj.reshape(-1).astype(np.float32).copy()
    cuda.memcpy_htod(addr_proj[0], proj_linearized)

    func = mod.get_function("compute_pixels")

    block_size = (32, 1, 1)  # 1 warp per point, 4 warps pers block
    grid_size = (
        int(math.ceil(src_points.reshape(-1, 3).shape[0] / block_size[0])), 1, 1)

    func(points_gpu, np.int32(src_points.reshape(-1, 3).shape[0]), np.int32(width), np.int32(height), pixels_gpu,
         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(pixels_out, pixels_gpu)
    pixels_out = pixels_out.reshape(src_points.shape[0], 2)

    # --------------------- COMPUTE FRAGMENtS PER PIXEL ------------------

    fragments_per_face = np.zeros(
        (src_indices.shape[0],)).astype(np.uint32).reshape(-1)
    indices_cpu = src_indices.reshape(-1)
    fragments_gpu = cuda.mem_alloc(fragments_per_face.nbytes)
    indices_gpu = cuda.mem_alloc(indices_cpu.nbytes)
    cuda.memcpy_htod(indices_gpu, indices_cpu)

    func = mod.get_function("compute_fragments_per_triangle")

    block_size = (32, 1, 1)  # 1 warp per point, 4 warps pers block
    grid_size = (
        int(math.ceil((indices_cpu.shape[0] / 3) / block_size[0])), 1, 1)

    func(pixels_gpu, np.int32(src_points.reshape(-1, 3).shape[0]), indices_gpu,
         np.int32(indices_cpu.shape[0] / 3),
         np.int32(width), np.int32(height), fragments_gpu,
         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(fragments_per_face, fragments_gpu)
    fragments_gpu.free()
    # --------------------- INTERPOLATE ------------------

    cumulative_fragments = np.cumsum(
        fragments_per_face).astype(np.uint32).reshape(-1)
    cumulative_fragments = np.concatenate(
        [np.zeros(1, dtype=np.uint32), cumulative_fragments])

    fragments_gpu = cuda.mem_alloc(cumulative_fragments.nbytes)
    cuda.memcpy_htod(fragments_gpu, cumulative_fragments)

    interpolated_pixels = np.zeros(
        int(fragments_per_face.sum())*2).astype(np.float32).reshape(-1)
    interpolated_pixels_gpu = cuda.mem_alloc(interpolated_pixels.nbytes)
    cuda.memcpy_htod(interpolated_pixels_gpu, interpolated_pixels)

    interpolated_points = np.zeros(
        int(fragments_per_face.sum())*3).astype(np.float32).reshape(-1)
    interpolated_points_gpu = cuda.mem_alloc(interpolated_points.nbytes)
    cuda.memcpy_htod(interpolated_points_gpu, interpolated_points)

    block_size = (32, 1, 1)  # 1 warp per point, 4 warps pers block
    grid_size = (
        int(math.ceil((indices_cpu.shape[0] / 3) / block_size[0])), 1, 1)

    color_dimension = None
    if src_color_attribute is not None:
        color_dimension = src_color_attribute.shape[-1]
        assert color_dimension == 3 or color_dimension == 1, "color must be 3-dim or 1-dim"

        rgb = src_color_attribute.reshape(
            -1, color_dimension).copy().astype(np.float32).reshape(-1)
        rgb_gpu = cuda.mem_alloc(rgb.nbytes)
        cuda.memcpy_htod(rgb_gpu, rgb)
        interpolated_rgb = np.zeros(
            (int(fragments_per_face.sum()), color_dimension)).astype(np.float32).reshape(-1)
        interpolated_rgb_gpu = cuda.mem_alloc(interpolated_rgb.nbytes)
        cuda.memcpy_htod(interpolated_rgb_gpu, interpolated_rgb)

        func = mod.get_function("interpolate_attr{}".format(color_dimension))
        func(pixels_gpu, points_gpu,
             indices_gpu, np.int32(indices_cpu.shape[0] / 3),
             np.int32(width), np.int32(height),
             rgb_gpu,
             fragments_gpu, interpolated_pixels_gpu,
             interpolated_points_gpu, interpolated_rgb_gpu,
             block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(interpolated_rgb, interpolated_rgb_gpu)
    else:
        func = mod.get_function("interpolate_points")
        func(pixels_gpu, points_gpu,
             indices_gpu, np.int32(indices_cpu.shape[0] / 3),
             np.int32(width), np.int32(height),
             fragments_gpu, interpolated_pixels_gpu,
             interpolated_points_gpu,
             block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(interpolated_points, interpolated_points_gpu)
    cuda.memcpy_dtoh(interpolated_pixels, interpolated_pixels_gpu)

    interpolated_points_gpu.free()
    interpolated_pixels_gpu.free()

    fragments_gpu.free()
    indices_gpu.free()
    pixels_gpu.free()
    points_gpu.free()
    if src_color_attribute is not None:
        interpolated_rgb_gpu.free()
        rgb_gpu.free()

    interpolated_pixels = interpolated_pixels.reshape(
        int(fragments_per_face.sum()), 2)
    mask = interpolated_pixels[:, 0] != -1
    interpolated_pixels = interpolated_pixels[mask]

    interpolated_points = interpolated_points.reshape(-1, 3)
    interpolated_points = interpolated_points[mask]
    interpolated_depth = interpolated_points[:, 2]

    if color_dimension is not None:
        interpolated_rgb = interpolated_rgb.reshape(-1, color_dimension)
        interpolated_rgb = interpolated_rgb[mask]

    uvx = np.concatenate([(interpolated_pixels).astype(np.int32),
                         interpolated_depth[:, None]], axis=-1)

    # --------------------- Depth test ------------------

    sort_ind = np.lexsort((uvx[:, 2], uvx[:, 0], uvx[:, 1]))
    uvx_sorted = uvx[sort_ind]

    linear_uvx = np.stack(
        [(uvx_sorted[:, 1] * width + uvx_sorted[:, 0]).astype(np.int32), uvx_sorted[:, 2]], axis=-1)
    sorted_linear_uvx = np.sort(linear_uvx[:, 0])

    unique_sorted, counts = np.unique(linear_uvx[:, 0], return_counts=True)
    sorted_unique_sorted = np.sort(unique_sorted)

    non_duplicate_indices = np.concatenate(
        [np.zeros((1,)).astype(np.int64), np.cumsum(counts.astype(np.int64))], axis=-1).astype(np.int64)[:-1]

    passed_depth_test_indices = sort_ind[non_duplicate_indices]
    depth_test_uvx = uvx[passed_depth_test_indices][:, :2].astype(np.int32)

    filtered_points = interpolated_points[passed_depth_test_indices]
    if color_dimension is not None:
        filtered_rgb = interpolated_rgb[passed_depth_test_indices]

    point_map = np.zeros((height, width, 3))
    point_map[depth_test_uvx[:, 1], depth_test_uvx[:, 0]] = filtered_points
    rgb_map = None
    if color_dimension is not None:
        rgb_map = np.zeros((height, width, color_dimension))
        rgb_map[depth_test_uvx[:, 1], depth_test_uvx[:, 0]] = filtered_rgb

    # # --------------------- Depth test for points ------------------
    # uvx = np.concatenate([np.floor(pixels_out).astype(np.float32),
    #                      src_points[:, 2]], axis=-1)

    # sort_ind = np.lexsort((uvx[:, 2], uvx[:, 0], uvx[:, 1]))
    # uvx_sorted = uvx[sort_ind]

    # linear_uvx = np.stack(
    #     [(uvx_sorted[:, 1] * width + uvx_sorted[:, 0]).astype(np.uint32), uvx_sorted[:, 2]], axis=-1)
    # _, counts = np.unique(linear_uvx[:, 0], return_counts=True)
    # non_duplicate_indices = np.concatenate(
    #     [np.zeros((1,)), np.cumsum(counts)], axis=-1).astype(np.uint32)[:-1]

    # passed_depth_test_indices = sort_ind[non_duplicate_indices]
    # out_faces = np.all(
    #         np.in1d(src_indices, passed_depth_test_indices).reshape(src_indices.shape), axis=-1)
    # out_pixels = pixels_out[passed_depth_test_indices].astype(np.uint32)

    return interpolated_points, interpolated_pixels, point_map, rgb_map
