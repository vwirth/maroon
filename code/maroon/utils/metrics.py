
import math
import pycuda.driver as cuda
from maroon.cuda.cuda import load_kernel_from_cu
import numpy as np
import open3d as o3d
import os
import cv2


def nearest_neighbor_error(p1, p2, mask_src=None, mask_dest=None):

    mod = load_kernel_from_cu(os.path.join(
        os.path.dirname(__file__), "cuda", "metrics.cu"))
    dists = np.zeros(p1.shape[0], dtype=np.float32)

    p1_ = np.ascontiguousarray(
        p1.copy().reshape(-1).astype(np.float32))
    p2_ = np.ascontiguousarray(
        p2.copy().reshape(-1).astype(np.float32))
    if mask_src is None:
        mask_src = np.ones(p1.shape[0])
    if mask_dest is None:
        mask_dest = np.ones(p2.shape[0])
    mask_src_cpu = mask_src.reshape(-1).astype(np.uint8)
    mask_dest_cpu = mask_dest.reshape(-1).astype(np.uint8)
    assert mask_src_cpu.shape[0] == p1.shape[0]
    assert mask_dest_cpu.shape[0] == p2.shape[0], "mask: {}, p2: {}".format(
        mask_dest_cpu.shape, p2.shape)

    p1_gpu = cuda.mem_alloc(p1_.nbytes)
    p2_gpu = cuda.mem_alloc(p2_.nbytes)
    mask_src_gpu = cuda.mem_alloc(mask_src_cpu.nbytes)
    mask_dest_gpu = cuda.mem_alloc(mask_dest_cpu.nbytes)
    dists_gpu = cuda.mem_alloc(dists.nbytes)

    cuda.memcpy_htod(p1_gpu, p1_)
    cuda.memcpy_htod(p2_gpu, p2_)
    cuda.memcpy_htod(mask_dest_gpu, mask_dest_cpu)
    cuda.memcpy_htod(mask_src_gpu, mask_src_cpu)

    func = mod.get_function("one_sided_chamfer")
    block_size = (32*32, 1, 1)
    grid_size = (
        int(math.ceil((p1_.shape[0] / 3) / (block_size[0] / 32))), 1, 1)

    func(p1_gpu, np.int32(len(p1_) / 3), mask_src_gpu, p2_gpu, np.int32(len(p2_) / 3), mask_dest_gpu, dists_gpu,
         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(dists, dists_gpu)
    p1_gpu.free()
    p2_gpu.free()
    dists_gpu.free()
    mask_dest_gpu.free()
    mask_src_gpu.free()

    valid = dists < pow(2, 16)-1

    return dists, valid


def compute_view_dependence(p1, n1, proj, width, height):
    assert proj.shape[0] == 4 and proj.shape[1] == 4
    assert p1.shape[1] == 3

    mod = load_kernel_from_cu(os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "cuda", "metrics.cu"))
    dependence = np.zeros((p1.shape[0], 3)).astype(np.float32).reshape(-1)

    points_cpu = p1.reshape(-1, 3)
    ones = np.ones((points_cpu.shape[0], 1))
    points_cpu = np.concatenate(
        [points_cpu, ones], axis=-1).reshape(-1).astype(np.float32)

    normal_cpu = n1.reshape(-1).astype(np.float32)

    points_gpu = cuda.mem_alloc(points_cpu.nbytes)
    dependence_gpu = cuda.mem_alloc(dependence.nbytes)
    normal_gpu = cuda.mem_alloc(normal_cpu.nbytes)

    cuda.memcpy_htod(points_gpu, points_cpu)
    cuda.memcpy_htod(dependence_gpu, dependence)
    cuda.memcpy_htod(normal_gpu, normal_cpu)
    addr_proj = mod.get_global("project")
    addr_proj_inv = mod.get_global("project_inv")

    # move to __const__ memory
    proj_linearized = proj.reshape(-1).astype(np.float32).copy()

    proj_inv = proj.copy()
    proj_inv[3, :] = np.array([0, 0, 0, 1]).reshape(1, 4)
    proj_inv = np.linalg.inv(proj_inv)
    proj_inv[3, :] = proj[3, :]
    inv_proj_lineraized = proj_inv.reshape(-1).astype(np.float32).copy()
    cuda.memcpy_htod(addr_proj[0], proj_linearized)
    cuda.memcpy_htod(addr_proj_inv[0], inv_proj_lineraized)

    func = mod.get_function("view_dependence")

    block_size = (32, 1, 1)  # 1 warp per point, 4 warps pers block
    grid_size = (
        int(math.ceil(p1.reshape(-1, 3).shape[0] / block_size[0])), 1, 1)

    func(points_gpu, np.int32(p1.reshape(-1, 3).shape[0]), normal_gpu,
         np.int32(width), np.int32(height),
         dependence_gpu,
         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(dependence, dependence_gpu)
    dependence = dependence.reshape((p1.shape[0], 3))

    points_gpu.free()
    dependence_gpu.free()
    normal_gpu.free()

    return dependence


def projective_error_attribute(p1, a1, dest_attribute, proj, mask_src=None, mask_dest=None):
    assert proj.shape[0] == 4 and proj.shape[1] == 4
    assert p1.shape[1] == 3
    assert a1.shape[0] == p1.shape[0]

    mod = load_kernel_from_cu(os.path.join(
        os.path.dirname(__file__), "cuda", "metrics.cu"))
    error = np.zeros((p1.shape[0],)).astype(np.float32).reshape(-1)

    points_cpu = p1.reshape(-1, 3)
    ones = np.ones((points_cpu.shape[0], 1))
    points_cpu = np.concatenate(
        [points_cpu, ones], axis=-1).reshape(-1).astype(np.float32)
    if mask_src is None:
        mask_src = np.ones(p1.shape[0])
    if mask_dest is None:
        mask_dest = np.ones((dest_attribute.shape[0], dest_attribute.shape[1]))
    mask_src_cpu = mask_src.reshape(-1).astype(np.uint8)
    mask_dest_cpu = mask_dest.reshape(-1).astype(np.uint8)
    assert mask_src_cpu.shape[0] == p1.shape[0]
    assert mask_dest.shape[0] == dest_attribute.shape[0]
    assert mask_dest.shape[1] == dest_attribute.shape[1]

    attribute_cpu = a1.reshape(-1).astype(np.float32)
    dest_attribute_cpu = dest_attribute.reshape(-1).astype(np.float32)
    mask_src_gpu = cuda.mem_alloc(mask_src_cpu.nbytes)
    mask_dest_gpu = cuda.mem_alloc(mask_dest_cpu.nbytes)

    points_gpu = cuda.mem_alloc(points_cpu.nbytes)
    error_gpu = cuda.mem_alloc(error.nbytes)
    dest_attribute_gpu = cuda.mem_alloc(dest_attribute_cpu.nbytes)
    attribute_gpu = cuda.mem_alloc(attribute_cpu.nbytes)

    cuda.memcpy_htod(points_gpu, points_cpu)
    cuda.memcpy_htod(error_gpu, error)
    cuda.memcpy_htod(dest_attribute_gpu, dest_attribute_cpu)
    cuda.memcpy_htod(attribute_gpu, attribute_cpu)
    cuda.memcpy_htod(mask_dest_gpu, mask_dest_cpu)
    cuda.memcpy_htod(mask_src_gpu, mask_src_cpu)
    addr_proj = mod.get_global("project")

    # move to __const__ memory
    proj_linearized = proj.reshape(-1).astype(np.float32).copy()
    cuda.memcpy_htod(addr_proj[0], proj_linearized)

    func = mod.get_function("projective_error_attribute")

    block_size = (32, 1, 1)  # 1 warp per point, 4 warps pers block
    grid_size = (
        int(math.ceil(p1.reshape(-1, 3).shape[0] / block_size[0])), 1, 1)

    func(points_gpu, np.int32(p1.reshape(-1, 3).shape[0]), attribute_gpu, mask_src_gpu, dest_attribute_gpu, mask_dest_gpu,
         np.int32(dest_attribute.shape[1]), np.int32(
             dest_attribute.shape[0]),
         error_gpu,
         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(error, error_gpu)
    error = error.reshape((p1.shape[0],))

    points_gpu.free()
    error_gpu.free()
    attribute_gpu.free()
    dest_attribute_gpu.free()
    mask_dest_gpu.free()
    mask_src_gpu.free()

    valid = error < pow(2, 16)-1

    return error, valid


def projective_error_without_open3d(p1, dest_depth, proj, mask_src=None, mask_dest=None):
    assert proj.shape[0] == 4 and proj.shape[1] == 4
    assert p1.shape[1] == 3

    mod = load_kernel_from_cu(os.path.join(
        os.path.dirname(__file__), "cuda", "metrics.cu"))
    error = np.zeros((p1.shape[0],)).astype(np.float32).reshape(-1)

    points_cpu = p1.reshape(-1, 3)
    ones = np.ones((points_cpu.shape[0], 1))
    points_cpu = np.concatenate(
        [points_cpu, ones], axis=-1).reshape(-1).astype(np.float32)
    if mask_src is None:
        mask_src = np.ones(p1.shape[0])
    if mask_dest is None:
        mask_dest = np.ones((dest_depth.shape[0], dest_depth.shape[1]))
    mask_src_cpu = mask_src.reshape(-1).astype(np.uint8)
    mask_dest_cpu = mask_dest.reshape(-1).astype(np.uint8)

    depth_cpu = dest_depth.reshape(-1).astype(np.float32)

    points_gpu = cuda.mem_alloc(points_cpu.nbytes)
    error_gpu = cuda.mem_alloc(error.nbytes)
    depth_gpu = cuda.mem_alloc(depth_cpu.nbytes)
    mask_src_gpu = cuda.mem_alloc(mask_src_cpu.nbytes)
    mask_dest_gpu = cuda.mem_alloc(mask_dest_cpu.nbytes)

    cuda.memcpy_htod(points_gpu, points_cpu)
    cuda.memcpy_htod(error_gpu, error)
    cuda.memcpy_htod(depth_gpu, depth_cpu)
    cuda.memcpy_htod(mask_src_gpu, mask_src_cpu)
    cuda.memcpy_htod(mask_dest_gpu, mask_dest_cpu)
    addr_proj = mod.get_global("project")

    # move to __const__ memory
    proj_linearized = proj.reshape(-1).astype(np.float32).copy()
    cuda.memcpy_htod(addr_proj[0], proj_linearized)

    func = mod.get_function("projective_error")

    block_size = (32, 1, 1)  # 1 warp per point, 4 warps pers block
    grid_size = (
        int(math.ceil(p1.reshape(-1, 3).shape[0] / block_size[0])), 1, 1)

    func(points_gpu, np.int32(p1.reshape(-1, 3).shape[0]), mask_src_gpu, depth_gpu,
         np.int32(dest_depth.shape[1]), np.int32(
             dest_depth.shape[0]), mask_dest_gpu,
         error_gpu,
         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(error, error_gpu)
    error = error.reshape((p1.shape[0],))

    points_gpu.free()
    error_gpu.free()
    depth_gpu.free()
    mask_dest_gpu.free()
    mask_src_gpu.free()

    valid = error < pow(2, 16)-1

    return error, valid


# this metric does not consider perspective
def projective_one_sided_chamfer(p1, p2, thresh_2d: float = 0.05):

    mod = load_kernel_from_cu(os.path.join(
        os.path.dirname(__file__), "cuda", "metrics.cu"))
    dists = np.zeros(p1.shape[0], dtype=np.float32)

    p1_ = p1.reshape(-1).astype(np.float32)
    p2_ = p2.reshape(-1).astype(np.float32)

    p1_gpu = cuda.mem_alloc(p1_.nbytes)
    p2_gpu = cuda.mem_alloc(p2_.nbytes)
    dists_gpu = cuda.mem_alloc(dists.nbytes)

    cuda.memcpy_htod(p1_gpu, p1_)
    cuda.memcpy_htod(p2_gpu, p2_)

    func = mod.get_function("projective_one_sided_chamfer")
    block_size = (32, 32, 1)
    grid_size = (1, int(math.ceil(p1.shape[0] / block_size[1])), 1)

    func(p1_gpu, np.int32(len(p1)), p2_gpu, np.int32(len(p2)), dists_gpu, np.float32(thresh_2d),
         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(dists, dists_gpu)
    p1_gpu.free()
    p2_gpu.free()
    dists_gpu.free()

    dists_greater_zero = dists[dists > 0]
    chamfer_dist = dists_greater_zero.sum(
    ) / float(dists_greater_zero.shape[0])
    return dists, chamfer_dist


def projective_one_sided_chamfer_axis(p1, p2, axis=2):

    mod = load_kernel_from_cu(os.path.join(
        os.path.dirname(__file__), "cuda", "metrics.cu"))
    dists = np.zeros(p1.shape[0], dtype=np.float32)
    p1_ = p1.reshape(-1).astype(np.float32)
    p2_ = p2.reshape(-1).astype(np.float32)

    p1_gpu = cuda.mem_alloc(p1_.nbytes)
    p2_gpu = cuda.mem_alloc(p2_.nbytes)
    dists_gpu = cuda.mem_alloc(dists.nbytes)

    cuda.memcpy_htod(p1_gpu, p1_)
    cuda.memcpy_htod(p2_gpu, p2_)

    func = mod.get_function("projective_one_sided_chamfer_axis")
    block_size = (32, 32, 1)
    grid_size = (1, int(math.ceil(p1.shape[0] / block_size[1])), 1)

    func(p1_gpu, np.int32(len(p1)), p2_gpu, np.int32(len(p2)), np.int32(axis), dists_gpu,
         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(dists, dists_gpu)
    p1_gpu.free()
    p2_gpu.free()
    dists_gpu.free()

    dists_greater_zero = dists[dists > 0]
    chamfer_dist = dists_greater_zero.sum(
    ) / float(dists_greater_zero.shape[0])
    return dists, chamfer_dist


def projective_error(p1, depth, rec_config, filter_zeros=True):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(
        p1.reshape(-1, 3))
    pc.paint_uniform_color([0, 0, 0.8])

    # cv2.imshow("depth: ", depth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    def create_depthmap():
        renderer_pc = o3d.visualization.rendering.OffscreenRenderer(
            depth.shape[1], depth.shape[0])
        renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))
        renderer_pc.scene.add_geometry(
            "pcd", pc, o3d.visualization.rendering.MaterialRecord())
        renderer_pc.scene.camera.set_projection(
            o3d.visualization.rendering.Camera.Projection.Ortho, rec_config["xmin"], rec_config["xmax"], rec_config["ymin"], rec_config["ymax"], rec_config["zmin"], rec_config["zmax"])

        center = [0, 0, -1]  # look_at target
        eye = [0, 0, 0]  # camera position
        up = [0, 1, 0]  # camera orientation
        renderer_pc.scene.camera.look_at(center, eye, up)
        rgb_image = np.asarray(renderer_pc.render_to_image())
        depth_image = np.asarray(renderer_pc.render_to_depth_image())

        # cv2.imshow("rgb: ", rgb_image)
        # cv2.imshow("depth: ", depth_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return depth_image * (rec_config["zmax"] - rec_config["zmin"]) + rec_config["zmin"]

    depth_proj = create_depthmap()
    error_z = np.abs(depth - depth_proj)
    if filter_zeros:
        error_z[depth == 0] = np.nan
    return error_z
