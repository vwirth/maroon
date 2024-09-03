

import os

import numpy as np
import cv2
import open3d as o3d
import json
import datetime
import math

import pycuda.driver as cuda
from maroon.cuda.cuda import load_kernel_from_cu


import pykinect_azure as pykinect
pykinect.initialize_libraries()


# extracted from: https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/examples/undistort/main.cpp#L68
def compute_xy_range(calibration: pykinect.Calibration,
                     camera_type: pykinect.k4a._k4a.k4a_calibration_type_t,
                     width: int,
                     height: int):
    step_u = 0.25
    step_v = 0.25
    min_u = 0
    min_v = 0
    max_u = width-1
    max_v = height-1
    center_u = 0.5 * width
    center_v = 0.5 * height

    valid = 0
    p = pykinect.k4a._k4a.k4a_float2_t((0, 0))

    # search x_min
    uv = np.asarray([center_u, center_v])
    while uv[0] >= min_u:
        p.xy.x = uv[0]
        p.xy.y = uv[1]
        ray, valid = calibration.convert_2d_to_3d(
            p, 1.0, camera_type, camera_type)
        if (not valid):
            break
        x_min = ray.xyz.x
        uv[0] -= step_u

    # search x_max
    uv = np.asarray([center_u, center_v])
    while uv[0] <= max_u:
        p.xy.x = uv[0]
        p.xy.y = uv[1]
        ray, valid = calibration.convert_2d_to_3d(
            p, 1.0, camera_type, camera_type)
        if (not valid):
            break
        x_max = ray.xyz.x
        uv[0] += step_u

    # search y_min
    uv = np.asarray([center_u, center_v])
    while uv[1] >= min_v:
        p.xy.x = uv[0]
        p.xy.y = uv[1]
        ray, valid = calibration.convert_2d_to_3d(
            p, 1.0, camera_type, camera_type)
        if (not valid):
            break
        y_min = ray.xyz.x
        uv[1] -= step_v

    # search y_max
    uv = np.asarray([center_u, center_v])
    while uv[1] <= max_v:
        p.xy.x = uv[0]
        p.xy.y = uv[1]
        ray, valid = calibration.convert_2d_to_3d(
            p, 1.0, camera_type, camera_type)
        if (not valid):
            break
        y_max = ray.xyz.x
        uv[1] += step_v

    return x_min, x_max, y_min, y_max


def create_pinhole_from_xy_range(calibration: pykinect.Calibration,
                                 camera_type: pykinect.k4a._k4a.k4a_calibration_type_t):
    width = calibration.depth_calibration.resolution_width
    height = calibration.depth_calibration.resolution_height
    if (camera_type == pykinect.k4a._k4a.K4A_CALIBRATION_TYPE_COLOR):
        width = calibration.color_calibration.resolution_width
        height = calibration.color_calibration.resolution_height

    x_min, x_max, y_min, y_max = compute_xy_range(calibration, camera_type)

    fx = 1.0 / (x_max - x_min)
    fy = 1.0 / (y_max - y_min)
    cx = -x_min * fx
    cy = -y_min * fy

    return width, height, fx, fy, cx, cy


INTERPOLATION_TYPES = {
    "nearest_neighbor": 0,
    "bilinear": 1,
    "bilinear_depth": 2
}


def create_undistortion_lut(calibration: pykinect.Calibration,
                            camera_type: pykinect.k4a._k4a.k4a_calibration_type_t,
                            pinhole_params: tuple,
                            interpolation_type=INTERPOLATION_TYPES["nearest_neighbor"]):
    p_width, p_height, p_fx, p_fy, p_cx, p_cy = pinhole_params
    lut_px = np.zeros((p_height, p_width, 2), dtype=np.int32)  # (x,y)
    lut_weight = np.zeros((p_height, p_width, 4),
                          dtype=np.float32)  # 4 weights

    ray = pykinect.k4a._k4a.k4a_float3_t((0, 0, 1))

    src_width = calibration.depth_calibration.resolution_width
    src_height = calibration.depth_calibration.resolution_height
    if (camera_type == pykinect.k4a._k4a.K4A_CALIBRATION_TYPE_COLOR):
        src_width = calibration.color_calibration.resolution_width
        src_height = calibration.color_calibration.resolution_height

    idx = 0
    for y in range(0, p_height):
        ray.y = (float(y) - p_cy) / p_fy
        for x in range(0, p_width):
            ray.xyz.x = ((float(x) - p_cx)) / p_fx
            distorted, valid = calibration.convert_3d_to_2d(
                ray, camera_type, camera_type)

            src = np.zeros((2,))
            if (interpolation_type == INTERPOLATION_TYPES["nearest_neighbor"]):
                src[0] = np.floor(distorted.xy.x + 0.5)
                src[1] = np.floor(distorted.xy.y + 0.5)
            elif (interpolation_type == INTERPOLATION_TYPES["bilinear"] or interpolation_type == INTERPOLATION_TYPES["bilinear_depth"]):
                src[0] = np.floor(distorted.xy.x)
                src[1] = np.floor(distorted.xy.y)
            else:
                print("Unknown interpolation type")
                exit(0)

            if (valid and src[0] >= 0 and src[0] < src_width and src[1] >= 0 and src[1] < src_height):
                lut_px[y, x] = src
                if (interpolation_type == INTERPOLATION_TYPES["bilinear"] or interpolation_type == INTERPOLATION_TYPES["bilinear_depth"]):
                    w_x = distorted.xy.x - src[0]
                    w_y = distorted.xy.y - src[1]
                    w0 = (1.0 - w_x) * (1.0 - w_y)
                    w1 = w_x * (1.0 - w_y)
                    w2 = (1.0 - w_x) * w_y
                    w3 = w_x * w_y

                    lut_weight[y, x, 0] = w0
                    lut_weight[y, x, 1] = w1
                    lut_weight[y, x, 2] = w2
                    lut_weight[y, x, 3] = w3
            else:
                lut_px[y, x, 0] = -1000
                lut_px[y, x, 1] = -1000

    return lut_px, lut_weight


def remap(src: np.ndarray, pinhole_params,
          lut_px: np.ndarray, lut_weights: np.ndarray,
          interpolation_type=INTERPOLATION_TYPES["nearest_neighbor"]):
    dst_width, dst_height, _, _, _, _ = pinhole_params
    src_height, src_width = src.shape[:2]
    dst = np.zeros((dst_height, dst_width))

    for y in range(0, dst_height):
        for x in range(0, dst_width):
            lut_data_px = lut_px[y, x]
            lut_data_weights = lut_weights[y, x]

            if (lut_data_px[0] != -1000 and lut_data_px[1] != 1000):
                if (interpolation_type == INTERPOLATION_TYPES["nearest_neighbor"]):
                    dst[y, x] = src[lut_data_px[1], lut_data_px[0]]
                elif (interpolation_type == INTERPOLATION_TYPES["bilinear"] or interpolation_type == INTERPOLATION_TYPES["bilinear_depth"]):
                    n0 = src[lut_data_px[1], lut_data_px[0]]
                    n1 = src[lut_data_px[1], lut_data_px[0] + 1]
                    n2 = src[lut_data_px[1] + 1, lut_data_px[0]]
                    n3 = src[lut_data_px[1] + 1, lut_data_px[0] + 1]

                    if interpolation_type == INTERPOLATION_TYPES["bilinear_depth"]:
                        if (n0 <= 0 or n1 <= 0 or n2 <= 0 or n3 <= 0):
                            continue
                        skip_interpolation_ratio = 0.04693441759
                        depth_min = min(min(n0, n1), min(n2, n3))
                        depth_max = max(max(n0, n1), max(n2, n3))
                        depth_delta = depth_max - depth_min
                        skip_interpolation_threshold = skip_interpolation_ratio * depth_min
                        if (depth_delta > skip_interpolation_threshold):
                            continue

                    dst[y, x] = n0 * lut_data_weights[0] + n1 * lut_data_weights[1] + \
                        n2 * lut_data_weights[2] + n3 * \
                        lut_data_weights[3] + 0.5
                else:
                    print("Unknown interpolation type")
                    exit(0)

    return dst


class KinectDataLoader():
    def __init__(self, kinect_data_path: str, space="color"):
        self.data_path = kinect_data_path
        self.calibration = self.load_calibration()
        self.space = space

    def get_timestamps(self):
        # timestamps_kinect = [int(f.split(".")[0]) * 1e-9 for f in sorted(
        #     os.listdir(os.path.join(self.data_path, "rgb")))]
        timestamp_file = os.path.join(self.data_path, "timestamps.txt")
        assert os.path.exists(timestamp_file)

        device_timestamps = []
        system_timestamps = []
        system_absolute_timestamps = []
        with open(timestamp_file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                tokens = line.split(" ")

                device_timestamps.append(np.uint64(tokens[1]) * 1e-6)
                system_timestamps.append(np.uint64(tokens[2]) * 1e-9)
                system_absolute_timestamps.append(np.uint64(tokens[3]) * 1e-9)

        return device_timestamps, system_timestamps, system_absolute_timestamps

    def get_synced_timestamps(self, calibration_path=""):

        timestamp_file = os.path.join(self.data_path, "timestamps_synced.txt")
        if len(calibration_path) > 0:
            calibration_file = os.path.join(
                calibration_path, "temporal_alignment.json")
        else:
            calibration_file = ""

        if not os.path.exists(timestamp_file) and not os.path.exists(calibration_file):
            return []

        if os.path.exists(timestamp_file):
            absolute_timestamps = []
            with open(timestamp_file, 'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    tokens = line.split(" ")
                    absolute_timestamps.append(np.uint64(tokens[1]) * 1e-9)
        else:
            with open(calibration_file) as f:
                calib_data = json.load(f)
            if not "kinect2radar" in calib_data:
                return []
            _, _, unsynced_timestamps = self.get_timestamps()
            absolute_timestamps = []
            for t in unsynced_timestamps:
                absolute_timestamps.append(
                    t + calib_data["kinect2radar"]["shift_seconds"])
        return absolute_timestamps

    def load_calibration(self):
        assert os.path.exists(os.path.join(
            self.data_path, "calibration.json")), "Path does not exist: {}".format(os.path.join(
                self.data_path, "calibration.json"))
        return pykinect.Calibration.from_file(
            os.path.join(self.data_path, "calibration.json"))

    def num_frames(self):
        assert os.path.exists(os.path.join(self.data_path, "rgb"))

        num_frames = 0
        for frame in os.listdir(os.path.join(self.data_path, "rgb")):
            if ".png" in frame:
                num_frames = num_frames + 1
        return num_frames

    def get_intrinsics(self):
        # BEWARE: this intrinsic matrix will not help to
        # transform the point cloud back to the 2D image
        # since it does not consider distortion
        # (the point cloud was created from an undistorted depth
        # map using the kinect API)

        if self.space == "color":
            mat = self.calibration.get_matrix(
                pykinect.k4a._k4a.K4A_CALIBRATION_TYPE_COLOR)
        else:
            mat = self.calibration.get_matrix(
                pykinect.k4a._k4a.K4A_CALIBRATION_TYPE_DEPTH)
        K = np.eye(4)
        K[:3, :3] = np.array(mat)

        return K

    def get_extrinsics(self, fr, to):
        return np.array(self.calibration.get_extrinsics(fr, to))

    def create_undistorted_depth(self, pc, distorted_depth):
        height, width = distorted_depth.shape[:2]

        mod = load_kernel_from_cu(os.path.join(
            os.path.dirname(__file__), "cuda", "data_loader.cu"))
        depth_custom = np.zeros((height, width)).astype(np.float32).reshape(-1)

        points_cpu = pc.reshape(-1).astype(np.float32).copy()
        points_gpu = cuda.mem_alloc(points_cpu.nbytes)
        depth_gpu = cuda.mem_alloc(depth_custom.nbytes)

        cuda.memcpy_htod(points_gpu, points_cpu)
        cuda.memcpy_htod(depth_gpu, depth_custom)
        addr_proj = mod.get_global("project")

        proj = self.get_intrinsics()
        # move to __const__ memory
        proj_linearized = proj[:3, :4].reshape(-1).astype(np.float32).copy()
        cuda.memcpy_htod(addr_proj[0], proj_linearized)

        func = mod.get_function("project_depth")

        block_size = (32, 1, 1)  # 1 warp per point, 4 warps pers block
        grid_size = (
            int(math.ceil(pc.reshape(-1, 3).shape[0] / block_size[0])), 1, 1)

        func(points_gpu, np.int32(pc.reshape(-1, 3).shape[0]), np.int32(width), np.int32(height), depth_gpu,
             block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(depth_custom, depth_gpu)
        depth_custom = depth_custom.reshape(height, width)

        points_gpu.free()
        depth_gpu.free()

        return depth_custom

    def compute_normal_from_pc(self, pc, depth):

        wpos_lin_x = pc.reshape(-1, 3).astype(np.float32)
        wpos_lin_y = pc.transpose(1, 0, 2).reshape(-1, 3).astype(np.float32)
        normal_x = []
        for i in range(0, 3):
            normal_x.append(np.convolve(
                wpos_lin_x[:, i], np.array([-1, 0, 1]), mode='same'))
        normal_x = np.stack(
            normal_x, axis=-1).reshape(depth.shape[0], depth.shape[1], 3)
        normal_y = []
        for i in range(0, 3):
            normal_y.append(np.convolve(
                wpos_lin_y[:, i], np.array([1, 0, -1]), mode='same'))
        normal_y = np.stack(
            normal_y, axis=-1).reshape(depth.shape[1], depth.shape[0], 3).transpose(1, 0, 2)
        normal = np.cross(normal_x, normal_y) * \
            np.array([1, 1, 1]).reshape(1, 1, 3)

        normal_length = np.linalg.norm(normal, axis=-1)
        positive = normal_length > 0

        normal[positive] = normal[positive] / \
            normal_length[positive][:, None]

        return normal

    def get_mask(self, frame_index=0):
        if frame_index == -1:
            frame_index = self.num_frames() - 1

        mask_dir = os.path.join(self.data_path, "mask")
        filename = str(frame_index).zfill(6)

        mask_file = os.path.join(mask_dir, filename + ".png")
        mask = None

        if not os.path.exists(mask_file):
            print("Did not find mask: {}".format(mask_file))

        else:
            mask = ((cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                    > 0) * 1).astype(np.uint8)

        if self.space == "depth":
            mask_dir = os.path.join(self.data_path, "mask_depth")
            if mask is not None and not os.path.exists(os.path.join(mask_dir, filename+".png")):
                depth_dir = os.path.join(self.data_path, "depth")
                depth_file = os.path.join(depth_dir, filename + ".png")
                depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

                depth_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_DEPTH16,
                                                                depth.shape[1],
                                                                depth.shape[0],
                                                                depth.shape[1]*2, depth)

                transformation = pykinect.Transformation(self.calibration)

                mask_rgba = np.stack(
                    [mask, mask, mask, np.ones_like(mask)], axis=-1).astype(np.uint8)
                mask_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                               mask_rgba.shape[1],
                                                               mask_rgba.shape[0],
                                                               mask_rgba.shape[1]*4, mask_rgba)

                _, mask_depth = transformation.color_image_to_depth_camera(
                    depth_image, mask_image).to_numpy()
                mask_depth = mask_depth[:, :, 0]
                mask = mask_depth
                cv2.imwrite(os.path.join(mask_dir, filename+".png"), mask)

        return mask

    def get_frame(self, frame_index=0, use_mask=False, mask_idx=None, manual_mask=None, undistort=False,
                  averaging_factor=1):
        if frame_index == -1:
            frame_index = self.num_frames() - 1
        average = averaging_factor > 1
        averaging_factor = min(averaging_factor, (self.num_frames()))

        rgb_dir = os.path.join(self.data_path, "rgb")
        depth_dir = os.path.join(self.data_path, "depth")
        mask_dir = os.path.join(self.data_path, "mask")
        filename = str(frame_index).zfill(6)

        mask_filename = filename
        if mask_idx is not None:
            if mask_idx == -1:
                mask_idx = self.num_frames() - 1
            mask_filename = str(mask_idx).zfill(6)

        rgb_file = os.path.join(rgb_dir, filename + ".png")
        depth_file = os.path.join(depth_dir, filename + ".png")
        mask_file = os.path.join(mask_dir, mask_filename + ".png")

        assert os.path.exists(rgb_file)
        assert os.path.exists(depth_file)
        rgb_image = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)
        if not average:
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        else:
            num_frames = (self.num_frames())
            avg_depth = None
            num_avg = 0

            if frame_index == num_frames-1:
                r = np.arange(frame_index - averaging_factor +
                              2, frame_index+2, 1)
            else:
                r = np.arange(-averaging_factor//2, averaging_factor//2, 1)
            for i in r:
                index = frame_index + i
                if (index < 0):
                    index = num_frames + index
                if (index >= num_frames):
                    index = (index - num_frames)
                filename_avg = str(index).zfill(6)
                if not os.path.exists(os.path.join(
                        depth_dir, filename_avg + ".png")):
                    continue
                print("Averaging index: ", index)

                depth = cv2.imread(os.path.join(
                    depth_dir, filename_avg + ".png"), cv2.IMREAD_UNCHANGED)

                if num_avg == 0:
                    pixel_cnts = np.zeros_like(depth)
                    avg_depth = depth.astype(np.float32)
                    pixel_cnts[avg_depth > 0] = 1
                else:
                    pixel_cnts[depth > 0] += 1
                    avg_depth[depth > 0] = depth.astype(
                        np.float32)[depth > 0] + avg_depth[depth > 0]

                num_avg = num_avg + 1
            # depth_avg = (avg_depth / num_avg).astype(np.uint16)
            depth_avg = np.zeros_like(avg_depth)
            depth_avg[pixel_cnts > 0] = (
                avg_depth[pixel_cnts > 0] / pixel_cnts[pixel_cnts > 0]).astype(np.uint16)

        mask = None
        if use_mask:
            if not os.path.exists(mask_file):
                print("Did not find mask: {}".format(mask_file))

            else:
                mask = ((cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                        > 0) * 1).astype(np.uint8)

        # if mask is not None:
        #     rgb_overlay = rgb_image * mask[:, :, None]
        #     cv2.imshow("rgb_overlay: ", rgb_overlay.astype(np.uint8) * 255)
        #     cv2.waitKey(0)

        if self.space == "depth":
            depth_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_DEPTH16,
                                                            depth.shape[1],
                                                            depth.shape[0],
                                                            depth.shape[1]*2, depth)
            rgba_image = np.concatenate([rgb_image, np.ones(
                (rgb_image.shape[0], rgb_image.shape[1], 1))], axis=-1).astype(np.uint8)
            color_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                            rgb_image.shape[1],
                                                            rgb_image.shape[0],
                                                            rgb_image.shape[1]*4, rgba_image)

            transformation = pykinect.Transformation(self.calibration)
            _, rgb_image = transformation.color_image_to_depth_camera(
                depth_image, color_image).to_numpy()
            rgb_image = rgb_image[:, :, :3]

            mask_dir = os.path.join(self.data_path, "mask_depth")
            filename = str(frame_index).zfill(6)
            if use_mask and mask is not None and not os.path.exists(os.path.join(mask_dir, filename+".png")):
                mask_rgba = np.stack(
                    [mask, mask, mask, np.ones_like(mask)], axis=-1).astype(np.uint8)
                mask_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                               mask_rgba.shape[1],
                                                               mask_rgba.shape[0],
                                                               mask_rgba.shape[1]*4, mask_rgba)

                _, mask_depth = transformation.color_image_to_depth_camera(
                    depth_image, mask_image).to_numpy()
                mask_depth = mask_depth[:, :, 0]
                mask = mask_depth
                cv2.imwrite(os.path.join(mask_dir, filename+".png"), mask)
            elif use_mask:
                mask = cv2.imread(os.path.join(mask_dir, filename+".png"))

        if average:
            depth = depth_avg

        if use_mask and manual_mask is not None:
            mask = manual_mask

        # depth_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_DEPTH16,
        #                                                 depth.shape[1],
        #                                                 depth.shape[0],
        #                                                 depth.shape[1]*2, depth)

        # transformation = pykinect.Transformation(self.calibration)
        # transformed_depth_image = transformation.depth_image_to_color_camera(
        #     depth_image)
        # _, transformed_depth = transformed_depth_image.to_numpy()

        # rgbd_image = rgb_image * (transformed_depth > 0).astype(np.uint8)[:,:,None]
        # cv2.imshow("rgbd", rgbd_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return rgb_image, depth, *self.project_depth_to_3d(depth, mask, True, space=self.space)

    def get_ir_image(self, frame_index=0, use_mask=False, manual_mask=None, undistort=False):
        if frame_index == -1:
            frame_index = self.num_frames() - 1

        ir_dir = os.path.join(self.data_path, "ir")
        mask_dir = os.path.join(self.data_path, "mask")
        depth_dir = os.path.join(self.data_path, "depth")
        filename = str(frame_index).zfill(6)

        ir_file = os.path.join(ir_dir, filename + ".png")
        depth_file = os.path.join(depth_dir, filename + ".png")
        if not os.path.exists(ir_file):
            return None
        assert os.path.exists(
            depth_file), "Path does not exist: {}".format(ir_file)
        mask_file = os.path.join(mask_dir, filename + ".png")

        ir = cv2.imread(ir_file, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        mask = None
        if use_mask:
            if not os.path.exists(mask_file):
                print("Did not find mask: {}".format(mask_file))

            else:
                mask = ((cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                        > 0) * 1).astype(np.uint8)

        # print("mask at: ", mask_file)
        # cv2.imshow("mask: ", mask * 255)
        # cv2.waitKey(0)

        ir_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_CUSTOM16,
                                                     ir.shape[1],
                                                     ir.shape[0],
                                                     ir.shape[1]*2, ir)

        depth_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_DEPTH16,
                                                        depth.shape[1],
                                                        depth.shape[0],
                                                        depth.shape[1]*2, depth)

        transformation = pykinect.Transformation(self.calibration)

        if self.space == "depth" and use_mask and mask is not None:
            mask_dir = os.path.join(self.data_path, "mask_depth")
            filename = str(frame_index).zfill(6)
            if not os.path.exists(os.path.join(mask_dir, filename+".png")):
                mask_rgba = np.stack(
                    [mask, mask, mask, np.ones_like(mask)], axis=-1).astype(np.uint8)
                mask_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                               mask_rgba.shape[1],
                                                               mask_rgba.shape[0],
                                                               mask_rgba.shape[1]*4, mask_rgba)

                _, mask_depth = transformation.color_image_to_depth_camera(
                    depth_image, mask_image).to_numpy()
                mask_depth = mask_depth[:, :, 0]
                mask = mask_depth
                cv2.imwrite(os.path.join(mask_dir, filename+".png"), mask)
            else:
                mask = cv2.imread(os.path.join(mask_dir, filename+".png"))

        if use_mask and manual_mask is not None:
            mask = manual_mask

        if self.space == "color":
            _, transformed_ir_image = transformation.depth_image_to_color_camera_custom(depth_image,
                                                                                        ir_image)

            _, transformed_ir = transformed_ir_image.to_numpy()

            if mask is not None:
                transformed_ir = transformed_ir * mask

            return transformed_ir.astype(np.float32)
        else:
            if mask is not None:
                ir = ir * mask
            return ir.astype(np.float32)

    def get_depth_pc(self, frame_index=0, undistort=False):
        if frame_index == -1:
            frame_index = self.num_frames() - 1

        depth_dir = os.path.join(self.data_path, "depth")

        index = 0
        depth_file = str(frame_index).zfill(6) + ".png"
        # for f in sorted(os.listdir(depth_dir)):

        #     if os.path.isfile(os.path.join(depth_dir, f)) and f[-3:] == "png":
        #         if index == frame_index:
        #             depth_file = os.path.join(depth_dir, f)
        #             break

        #         index = index + 1

        assert os.path.exists(depth_file)
        depth = cv2.imread(os.path.join(
            depth_dir, depth_file), cv2.IMREAD_UNCHANGED)

        depth_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_DEPTH16,
                                                        depth.shape[1],
                                                        depth.shape[0],
                                                        depth.shape[1]*2, depth)

        transformation = pykinect.Transformation(self.calibration)
        transformed_depth_image = transformation.depth_image_to_color_camera(
            depth_image)
        _, transformed_depth = transformed_depth_image.to_numpy()

        xyz_transformed_image = transformation.depth_image_to_point_cloud(
            transformed_depth_image, pykinect.k4a._k4a.K4A_CALIBRATION_TYPE_COLOR)
        _, xyz_color_space = xyz_transformed_image.to_numpy()
        xyz_transformed = xyz_color_space.copy().reshape(-1, 3)
        xyz_color_space = xyz_color_space.reshape(
            transformed_depth.shape[0], transformed_depth.shape[1], 3)

        K = np.eye(4)
        K[:3, :3] = np.array(self.calibration.extrinsic.rotation).reshape(3, 3)
        K[:3, 3] = np.array(self.calibration.extrinsic.translation).reshape(3)
        ones = np.ones(xyz_transformed.shape[0]).reshape(-1, 1)
        xyz_depth_space = np.matmul(np.concatenate(
            [xyz_transformed, ones], axis=1), K.transpose())[:, :3]
        xyz_depth_space = xyz_depth_space.reshape(
            transformed_depth.shape[0], transformed_depth.shape[1], 3)

        if undistort:
            transformed_depth = self.create_undistorted_depth(
                xyz_color_space.reshape(-1, 3), transformed_depth)
            depth = self.create_undistorted_depth(
                xyz_depth_space.reshape(-1, 3), depth)

        return depth, transformed_depth, xyz_depth_space,

    def transform(self, pc, fr, to, normals=None):
        K = self.get_extrinsics(fr, to)
        if len(pc.shape) == 3:
            h, w, _ = pc.shape
        else:
            h = None
            w = None

        xyz_transformed = pc.copy().reshape(-1, 3)
        ones = np.ones(xyz_transformed.shape[0]).reshape(-1, 1)
        pc_transformed = np.matmul(np.concatenate(
            [xyz_transformed, ones], axis=1), K.transpose())[:, :3]
        if h is not None:
            pc_transformed = pc_transformed.reshape(
                h, w, 3)

        if normals is not None:
            N = np.transpose(np.linalg.inv(K))
            ones = np.ones((normals.shape[0], 1))
            transformed_normals = np.matmul(np.concatenate(
                [normals, ones], axis=1), N.transpose())[:, :3]
            length = np.linalg.norm(transformed_normals, axis=-1)
            transformed_normals[length >
                                0] = transformed_normals[length > 0] / length[length > 0].reshape(-1, 1)

            return pc_transformed, transformed_normals
        else:
            return pc_transformed

    def project_depth_to_3d(self, depth, mask=None, compute_normal=True, space="color", frame_index=0):
        depth = depth.astype(np.uint16)

        depth_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_DEPTH16,
                                                        depth.shape[1],
                                                        depth.shape[0],
                                                        depth.shape[1]*2, depth)

        transformation = pykinect.Transformation(self.calibration)
        transformed_depth_image = transformation.depth_image_to_color_camera(
            depth_image)
        _, transformed_depth = transformed_depth_image.to_numpy()
        if space == "color" and mask is not None:
            transformed_depth = transformed_depth * mask
            transformed_depth_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_DEPTH16,
                                                                        transformed_depth.shape[1],
                                                                        transformed_depth.shape[0],
                                                                        transformed_depth.shape[1]*2, transformed_depth)

            # cv2.imshow("transformed depth_: ", transformed_depth / transformed_depth.max())
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        elif space == "depth" and mask is not None:
            depth = depth * mask
            depth_image = pykinect.Image.create_from_buffer(pykinect.k4a._k4a.K4A_IMAGE_FORMAT_DEPTH16,
                                                            depth.shape[1],
                                                            depth.shape[0],
                                                            depth.shape[1]*2, depth)

        xyz_transformed_image = transformation.depth_image_to_point_cloud(
            depth_image, pykinect.k4a._k4a.K4A_CALIBRATION_TYPE_DEPTH)
        _, xyz_depth_space = xyz_transformed_image.to_numpy()
        xyz_depth_space = xyz_depth_space.astype(np.float32)
        xyz_depth_space = xyz_depth_space.reshape(
            depth.shape[0], depth.shape[1], 3)

        xyz_transformed_image = transformation.depth_image_to_point_cloud(
            transformed_depth_image, pykinect.k4a._k4a.K4A_CALIBRATION_TYPE_COLOR)
        _, xyz_color_space = xyz_transformed_image.to_numpy()
        xyz_color_space = xyz_color_space.astype(np.float32)
        xyz_transformed = xyz_color_space.copy().reshape(-1, 3)
        xyz_color_space = xyz_color_space.reshape(
            transformed_depth.shape[0], transformed_depth.shape[1], 3)

        # K = self.get_extrinsics("depth", "color")
        # xyz_transformed = xyz_depth_space.copy().reshape(-1, 3)
        # ones = np.ones(xyz_transformed.shape[0]).reshape(-1, 1)
        # xyz_color_space_tmp = np.matmul(np.concatenate(
        #     [xyz_transformed, ones], axis=1), K.transpose())[:, :3]
        # xyz_color_space_tmp = xyz_color_space_tmp.reshape(
        #     depth.shape[0], depth.shape[1], 3)

        # visualizer = o3d.visualization.VisualizerWithKeyCallback()
        # visualizer.create_window()

        # pc_color_from_depth = o3d.geometry.PointCloud()
        # pc_color_from_depth.points = o3d.utility.Vector3dVector(
        #     xyz_color_space_tmp[xyz_color_space_tmp[:,:,2] < 700].reshape(-1, 3))
        # pc_color_from_depth.paint_uniform_color([0,1,0])
        # visualizer.add_geometry(pc_color_from_depth)

        # pc_color_from_color = o3d.geometry.PointCloud()
        # pc_color_from_color.points = o3d.utility.Vector3dVector(xyz_color_space[xyz_color_space[:,:,2] < 700].reshape(-1, 3))
        # pc_color_from_color.paint_uniform_color([0,0,1])
        # visualizer.add_geometry(pc_color_from_color)

        # visualizer.run()
        # visualizer.destroy_window()

        # K = np.eye(4)
        # K[:3, :3] = np.array(
        #     self.calibration.extrinsic.rotation).reshape(3, 3)
        # K[:3, 3] = np.array(
        #     self.calibration.extrinsic.translation).reshape(3)
        # ones = np.ones(xyz_transformed.shape[0]).reshape(-1, 1)
        # xyz_depth_space = np.matmul(np.concatenate(
        #     [xyz_transformed, ones], axis=1), K.transpose())[:, :3]
        # xyz_depth_space = xyz_depth_space.reshape(
        #     transformed_depth.shape[0], transformed_depth.shape[1], 3)

        if space != "color":
            if compute_normal:
                normal_depth_space = self.compute_normal_from_pc(
                    xyz_depth_space, depth)
            else:
                normal_depth_space = None
        else:
            if compute_normal:
                # xyz_color_space = xyz_depth_space
                normal_color_space = self.compute_normal_from_pc(
                    xyz_color_space, transformed_depth)

            else:
                normal_color_space = None

        if space != "color":
            return depth.astype(np.float32), xyz_depth_space, normal_depth_space
        else:
            return transformed_depth.astype(np.float32), xyz_color_space, normal_color_space
