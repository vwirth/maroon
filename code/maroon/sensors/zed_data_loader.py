

import os

import numpy as np
import cv2
import open3d as o3d
import json
import datetime
import pyzed.sl as sl


class ZEDDataLoader():
    def __init__(self, zed_data_path: str):
        self.data_path = zed_data_path

    def process_svo(self, svo_path: str):
        assert os.path.exists(svo_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        zed = sl.Camera()

        input_type = sl.InputType()
        input_type.set_from_svo_file(svo_path)

        init = sl.InitParameters(input_t=input_type)
        init.camera_fps = 30
        init.camera_resolution = sl.RESOLUTION.HD1200
        # init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.MILLIMETER
        init.depth_stabilization = 100

        runtime = sl.RuntimeParameters()
        runtime.enable_depth = True
        # runtime.enable_fill_mode = True

        err = zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            zed.close()
            exit(1)

        if not os.path.exists(os.path.join(self.data_path, "rgb")):
            os.makedirs(os.path.join(self.data_path, "rgb"))
        if not os.path.exists(os.path.join(self.data_path, "rgb", "left")):
            os.makedirs(os.path.join(self.data_path, "rgb", "left"))
        if not os.path.exists(os.path.join(self.data_path, "rgb", "right")):
            os.makedirs(os.path.join(self.data_path, "rgb", "right"))
        if not os.path.exists(os.path.join(self.data_path, "depth")):
            os.makedirs(os.path.join(self.data_path, "depth"))
        if not os.path.exists(os.path.join(self.data_path, "depth", "left")):
            os.makedirs(os.path.join(self.data_path, "depth", "left"))
        if not os.path.exists(os.path.join(self.data_path, "depth", "right")):
            os.makedirs(os.path.join(self.data_path, "depth", "right"))

        camera_config = zed.get_camera_information().camera_configuration
        calib_parameters = camera_config.calibration_parameters

        json_data = {}
        if os.path.exists(os.path.join(self.data_path, "calibration.json")):
            with open(os.path.join(self.data_path, "calibration.json")) as f:
                json_data = json.load(f)
        json_data["capture"] = {
            "fps": 60,
            "resolution": "HD1200",
            # "depth_mode": "NEURAL",
            "depth_mode": "ULTRA",
            "units": "MILLIMETER",
            "stabilization": 100
        }
        json_data["left"] = {
            "fx": calib_parameters.left_cam.fx,
            "fy": calib_parameters.left_cam.fy,
            "cx": calib_parameters.left_cam.cx,
            "cy": calib_parameters.left_cam.cy,
        }
        json_data["right"] = {
            "fx": calib_parameters.right_cam.fx,
            "fy": calib_parameters.right_cam.fy,
            "cx": calib_parameters.right_cam.cx,
            "cy": calib_parameters.right_cam.cy,
        }
        with open(os.path.join(self.data_path, "calibration.json"), "w") as f:
            json.dump(json_data, f)

        timestamps = []
        err = zed.grab(runtime)
        frame = 0

        updated = False
        while err != sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            if err == sl.ERROR_CODE.SUCCESS:
                filename = str(frame).zfill(6)

                if not os.path.exists(os.path.join(self.data_path, "rgb", "left", filename+".jpg")):
                    rgb_left = sl.Mat()
                    depth_left = sl.Mat()
                    rgb_right = sl.Mat()
                    depth_right = sl.Mat()

                    timestamp = zed.get_timestamp(
                        sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                    zed.retrieve_image(rgb_left, sl.VIEW.LEFT, sl.MEM.CPU)
                    zed.retrieve_image(rgb_right, sl.VIEW.RIGHT, sl.MEM.CPU)
                    zed.retrieve_measure(
                        depth_left, sl.MEASURE.DEPTH, sl.MEM.CPU)
                    meas = zed.retrieve_measure(
                        depth_right, sl.MEASURE.DEPTH_RIGHT, sl.MEM.CPU)
                    # print("meas: ", meas)

                    depth_left_np = depth_left.numpy()
                    depth_right_np = depth_right.numpy()
                    # set invalid entries to zero
                    depth_left_np[np.logical_or(
                        np.isnan(depth_left_np), np.isinf(depth_left_np))] = 0
                    depth_right_np[np.logical_or(
                        np.isnan(depth_right_np), np.isinf(depth_right_np))] = 0
                    depth_left_np = depth_left_np.astype(np.uint16)
                    depth_right_np = depth_right_np.astype(np.uint16)

                    cv2.imwrite(os.path.join(self.data_path, "rgb",
                                "left", filename+".jpg"), rgb_left.numpy())
                    cv2.imwrite(os.path.join(self.data_path, "rgb",
                                "right", filename+".jpg"), rgb_right.numpy())
                    cv2.imwrite(os.path.join(self.data_path, "depth",
                                "left", filename+".png"), depth_left_np)
                    if (depth_right_np.shape[0] > 0):
                        cv2.imwrite(os.path.join(
                            self.data_path, "depth", "right", filename+".png"), depth_right_np)

                    timestamps.append(timestamp)
                    updated = True

                frame = frame + 1
                err = zed.grab(runtime)

        if updated:
            with open(os.path.join(self.data_path, "timestamps.txt"), "w") as f:
                f.write("filename absolute_system[ns]\n")
                for i, t in enumerate(timestamps):
                    filename = str(i).zfill(6)
                    f.write("{} {}\n".format(filename, str(t)))

    def get_timestamps(self):
        if not os.path.exists(os.path.join(self.data_path, "timestamps.txt")):
            for f in os.listdir(os.path.dirname(self.data_path)):
                if f.endswith(".svo"):
                    self.process_svo(os.path.join(
                        os.path.dirname(self.data_path), f))

        timestamp_file = os.path.join(self.data_path, "timestamps.txt")
        assert os.path.exists(timestamp_file)

        system_absolute_timestamps = []
        with open(timestamp_file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                tokens = line.split(" ")
                system_absolute_timestamps.append(np.uint64(tokens[1]) * 1e-9)

        return system_absolute_timestamps

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
            if not "zed2radar" in calib_data:
                return []
            unsynced_timestamps = self.get_timestamps()
            absolute_timestamps = []
            for t in unsynced_timestamps:
                absolute_timestamps.append(
                    t + calib_data["zed2radar"]["shift_seconds"])
        return absolute_timestamps

    def load_calibration(self):
        assert os.path.exists(os.path.join(
            self.data_path, "calibration.json")), "Path does not exist: {}".format(os.path.join(
                self.data_path, "calibration.json"))

        calib = {}
        with open(os.path.join(self.data_path, "calibration.json")) as f:
            calib = json.load(f)

        return calib

    def num_frames(self):
        if (not os.path.exists(os.path.join(self.data_path, "rgb"))
                or not os.path.exists(os.path.join(self.data_path, "depth"))):
            for f in os.listdir(os.path.dirname(self.data_path)):
                if f.endswith(".svo"):
                    self.process_svo(os.path.join(
                        os.path.dirname(self.data_path), f))

        num_frames = 0
        for frame in os.listdir(os.path.join(self.data_path, "rgb", "left")):
            if ".jpg" in frame:
                num_frames = num_frames + 1
        return num_frames

    def get_intrinsics(self):
        calib = self.load_calibration()

        K_l = np.eye(4)
        K_l[0, 0] = calib["left"]["fx"]
        K_l[1, 1] = calib["left"]["fy"]
        K_l[0, 2] = calib["left"]["cx"]
        K_l[1, 2] = calib["left"]["cy"]
        return K_l

    def get_mask(self, frame_index=0):
        if frame_index == -1:
            frame_index = self.num_frames() - 1

        mask_dir = os.path.join(self.data_path, "mask")
        filename = str(frame_index).zfill(6)

        mask_file = os.path.join(mask_dir, "left", filename + ".png")
        mask_left = None

        if not os.path.exists(mask_file):
            print("Did not find mask: {}".format(mask_file))

        else:
            mask_left = (cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                         > 0).astype(np.uint8)
        return mask_left

    def get_frame(self, frame_index=0, use_mask=False, mask_idx=None, averaging_factor=1, manual_mask=None):
        if frame_index == -1:
            frame_index = self.num_frames() - 1
        average = averaging_factor > 1
        averaging_factor = min(averaging_factor, (self.num_frames()))

        if (not os.path.exists(os.path.join(self.data_path, "rgb"))
                or not os.path.exists(os.path.join(self.data_path, "depth"))):
            for f in os.listdir(os.path.dirname(self.data_path)):
                if f.endswith(".svo"):
                    self.process_svo(os.path.join(
                        os.path.dirname(self.data_path), f))

        rgb_dir = os.path.join(self.data_path, "rgb")
        depth_dir = os.path.join(self.data_path, "depth")
        mask_dir = os.path.join(self.data_path, "mask")
        filename = str(frame_index).zfill(6)

        assert os.path.exists(rgb_dir)

        rgb_left = cv2.imread(os.path.join(
            rgb_dir, "left", filename + ".jpg"), cv2.IMREAD_UNCHANGED)[:, :, :3]
        rgb_right = cv2.imread(os.path.join(
            rgb_dir, "right", filename + ".jpg"), cv2.IMREAD_UNCHANGED)

        if not average:
            depth_left = cv2.imread(os.path.join(
                depth_dir, "left", filename + ".png"), cv2.IMREAD_UNCHANGED)

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
                        depth_dir, "left", filename_avg + ".png")):
                    continue
                print("Averaging index: ", index)

                depth = cv2.imread(os.path.join(
                    depth_dir, "left", filename_avg + ".png"), cv2.IMREAD_UNCHANGED)
                if num_avg == 0:
                    pixel_cnts = np.zeros_like(depth)
                    avg_depth = depth.astype(np.float32)
                    pixel_cnts[avg_depth > 0] = 1
                else:
                    pixel_cnts[depth > 0] += 1
                    avg_depth[depth > 0] = depth.astype(
                        np.float32)[depth > 0] + avg_depth[depth > 0]

                num_avg = num_avg + 1
            depth_left = np.zeros_like(avg_depth)
            depth_left[pixel_cnts > 0] = (
                avg_depth[pixel_cnts > 0] / pixel_cnts[pixel_cnts > 0]).astype(np.uint16)

            # depth_left = (avg_depth / num_avg).astype(np.uint16)

        # depth_left = depth_left / depth_left.max()
        # depth_left = ((1 - depth_left) * 2000) + 200

        mask_filename = filename
        if mask_idx is not None:
            if mask_idx == -1:
                mask_idx = self.num_frames() - 1
            mask_filename = str(mask_idx).zfill(6)

        mask_file = os.path.join(mask_dir, "left", mask_filename + ".png")
        mask_left = None
        if use_mask:
            if not os.path.exists(mask_file):
                print("Did not find mask: {}".format(mask_file))

            else:
                mask_left = (cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                             > 0).astype(np.uint8)

        if use_mask and manual_mask is not None:
            mask_left = manual_mask

        if mask_left is not None:
            depth_left = depth_left * mask_left

        calib = self.load_calibration()

        K_l = np.eye(3)
        K_l[0, 0] = calib["left"]["fx"]
        K_l[1, 1] = calib["left"]["fy"]
        K_l[0, 2] = calib["left"]["cx"]
        K_l[1, 2] = calib["left"]["cy"]

        pixel_x, pixel_y = np.meshgrid(np.linspace(
            0, depth_left.shape[1]-1, depth_left.shape[1]), np.linspace(0, depth_left.shape[0]-1, depth_left.shape[0]))
        pixels_left = np.stack(
            [pixel_x * depth_left, pixel_y * depth_left, depth_left], axis=-1)
        xyz_left = np.matmul(pixels_left, np.linalg.inv(K_l).transpose())

        def compute_pixel_normals(xyz):
            wpos_lin_x = xyz.reshape(-1, 3)
            wpos_lin_y = xyz.transpose(1, 0, 2).reshape(-1, 3)
            normal_x = []
            for i in range(0, 3):
                normal_x.append(np.convolve(
                    wpos_lin_x[:, i], np.array([-1, 0, 1]), mode='same'))
            normal_x = np.stack(
                normal_x, axis=-1).reshape(xyz.shape[0], xyz.shape[1], 3)
            normal_y = []
            for i in range(0, 3):
                normal_y.append(np.convolve(
                    wpos_lin_y[:, i], np.array([1, 0, -1]), mode='same'))
            normal_y = np.stack(
                normal_y, axis=-1).reshape(xyz.shape[1], xyz.shape[0], 3).transpose(1, 0, 2)
            normal = np.cross(normal_x, normal_y)
            normal_length = np.linalg.norm(normal, axis=-1)
            normal[normal_length > 0] = normal[normal_length > 0] / \
                normal_length[normal_length > 0][:, None]
            normal = normal * np.array([1, 1, 1])[None, :]

            return normal

        normal_left = compute_pixel_normals(xyz_left)

        if (os.path.exists(os.path.join(depth_dir, "right", filename + ".png"))):
            K_r = np.eye(3)
            K_r[0, 0] = calib["right"]["fx"]
            K_r[1, 1] = calib["right"]["fy"]
            K_r[0, 2] = calib["right"]["cx"]
            K_r[1, 2] = calib["right"]["cy"]

            depth_right = cv2.imread(os.path.join(
                depth_dir, "right", filename + ".png"), cv2.IMREAD_UNCHANGED)
            pixels_right = np.stack(
                [pixel_x * depth_right, pixel_y * depth_right, depth_right], axis=-1)
            xyz_right = np.matmul(pixels_right, np.linalg.inv(K_r).transpose())
            normal_right = compute_pixel_normals(xyz_right)
        else:
            depth_right = None
            pixels_right = None
            xyz_right = None
            normal_right = None

        return (rgb_left, depth_left, xyz_left, normal_left), (rgb_right, depth_right, xyz_right, normal_right)

    def get_depth_pc(self, frame_index=0):
        if frame_index == -1:
            frame_index = self.num_frames() - 1

        if (not os.path.exists(os.path.join(self.data_path, "rgb"))
                or not os.path.exists(os.path.join(self.data_path, "depth"))):
            for f in os.listdir(os.path.dirname(self.data_path)):
                if f.endswith(".svo"):
                    self.process_svo(os.path.join(
                        os.path.dirname(self.data_path), f))

        depth_dir = os.path.join(self.data_path, "depth")
        filename = str(frame_index).zfill(6)

        depth_left = cv2.imread(os.path.join(
            depth_dir, "left", filename + ".png"), cv2.IMREAD_UNCHANGED)

        calib = self.load_calibration()

        K_l = np.eye(3)
        K_l[0, 0] = calib["left"]["fx"]
        K_l[1, 1] = calib["left"]["fy"]
        K_l[0, 2] = calib["left"]["cx"]
        K_l[1, 2] = calib["left"]["cy"]

        pixel_x, pixel_y = np.meshgrid(np.linspace(
            0, depth_left.shape[1]-1, depth_left.shape[1]), np.linspace(0, depth_left.shape[0]-1, depth_left.shape[0]))
        pixels_left = np.stack(
            [pixel_x * depth_left, pixel_y * depth_left, depth_left], axis=-1)
        xyz_left = np.matmul(pixels_left, np.linalg.inv(K_l).transpose())

        return depth_left, xyz_left
