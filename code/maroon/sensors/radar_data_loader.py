import json
import os
import numpy as np
import cv2

from maroon.sensors.radar_control import RadarControllerFSCW
import time
import math
import open3d as o3d


class RadarDataLoader:
    def __init__(self, radar_data_path: str,
                 frequency_low=72,
                 frequency_high=82,
                 frequency_points=128,
                 xmin=-0.1,
                 xmax=0.1,
                 xsteps=100,
                 ymin=-0.1,
                 ymax=0.1,
                 ysteps=100,
                 zmin=0.2,
                 zmax=0.4,
                 zsteps=50,
                 use_empty_space_measurements=False,
                 averaging=1):
        self.data_path = radar_data_path

        self.xmin = xmin
        self.xmax = xmax
        self.xsteps = xsteps
        self.ymin = ymin
        self.ymax = ymax
        self.ysteps = ysteps
        self.zmin = zmin
        self.zmax = zmax
        self.zsteps = zsteps
        self.num_frames = 0

        self.radar_client = RadarControllerFSCW(frequency_low_ghz=frequency_low,
                                                frequency_high_ghz=frequency_high,
                                                frequency_points=frequency_points,
                                                averaging=averaging)

        self.use_empty_space_measurements = use_empty_space_measurements
        # cache last reconstruction results
        self.last_reco = {
            "maxproj": None,
            "reco": None,
            "depth": None,
            "volume": None,
            "pointcloud": None
        }

    def get_last_reco(self):
        return self.last_reco["reco"]

    def get_last_depth(self):
        return self.last_reco["depth"]

    def get_last_maxproj(self):
        return self.last_reco["maxproj"]

    def get_last_volume(self):
        return self.last_reco["volume"]

    def get_last_pointcloud(self):
        return self.last_reco["pointcloud"]

    def set_config(self, xmin, xmax, xsteps, ymin, ymax, ysteps, zmin, zmax, zsteps):
        self.xmin = xmin
        self.xmax = xmax
        self.xsteps = xsteps
        self.ymin = ymin
        self.ymax = ymax
        self.ysteps = ysteps
        self.zmin = zmin
        self.zmax = zmax
        self.zsteps = zsteps

    def get_config(self, frame_index=0, use_calibration=True):
        intr_data = None
        if (os.path.exists(os.path.join(self.data_path,  "calibration.json"))):
            with open(os.path.join(self.data_path,  "calibration.json"), 'r') as f:
                intr_data = json.load(f)

        frame_index_str = str(frame_index).zfill(6)
        if (use_calibration and frame_index_str in intr_data):
            rec_config = {
                "xmin": intr_data[frame_index_str]["volume"]["bb_min"][0],
                "xmax": intr_data[frame_index_str]["volume"]["bb_max"][0],
                "xsteps": intr_data[frame_index_str]["volume"]["grid_size"][0],
                "ymin": intr_data[frame_index_str]["volume"]["bb_min"][1],
                "ymax": intr_data[frame_index_str]["volume"]["bb_max"][1],
                "ysteps": intr_data[frame_index_str]["volume"]["grid_size"][1],
                "zmin": intr_data[frame_index_str]["volume"]["bb_min"][2],
                "zmax": intr_data[frame_index_str]["volume"]["bb_max"][2],
                "zsteps": intr_data[frame_index_str]["volume"]["grid_size"][2],
            }
        else:
            rec_config = {
                "xmin": self.xmin,
                "xmax": self.xmax,
                "xsteps": self.xsteps,
                "ymin": self.ymin,
                "ymax": self.ymax,
                "ysteps": self.ysteps,
                "zmin": self.zmin,
                "zmax": self.zmax,
                "zsteps": self.zsteps,
            }

        return rec_config

    def get_intrinsics(self, frame_index):
        assert os.path.exists(os.path.join(
            self.data_path, "calibration.json")), "Path not found: {}".format(os.path.join(
                self.data_path, "calibration.json"))

        if frame_index == -1:
            frame_index = len(self.radar_frames) - 1

        intrinsics = None
        frame_index_str = str(frame_index).zfill(6)
        if (os.path.exists(os.path.join(self.data_path,  "calibration.json"))):
            with open(os.path.join(self.data_path,  "calibration.json"), 'r') as f:
                intr_data = json.load(f)

            if "depth" in intr_data[frame_index_str]:
                if "transform" in intr_data[frame_index_str]["depth"]:
                    del intr_data[frame_index_str]["depth"]["transform"]
                    bb_min = intr_data[frame_index_str]["volume"]["bb_min"]
                    bb_max = intr_data[frame_index_str]["volume"]["bb_max"]
                    grid_size = intr_data[frame_index_str]["volume"]["grid_size"]
                    grid_size = [float(g-1) for g in grid_size]

                    intrinsics = np.eye(4)
                    intrinsics[0, 0] = grid_size[0] / (bb_max[0] - bb_min[0])
                    intrinsics[0, 3] = bb_min[0] * \
                        grid_size[0] / (bb_min[0] - bb_max[0])
                    intrinsics[1, 1] = grid_size[1] / (bb_min[1] - bb_max[1])
                    intrinsics[1, 3] = bb_min[1]*grid_size[1] / \
                        (bb_max[1] - bb_min[1]) + grid_size[1]
                    # from meter to millieter
                    intrinsics[2, 2] = 1000.0

                    intr_data[frame_index_str]["depth"]["intrinsics"] = intrinsics.tolist()

                    if os.path.exists(os.path.join(self.data_path, "depth", frame_index_str + ".png")):
                        depth = cv2.imread(os.path.join(self.data_path, "depth",
                                                        frame_index_str + ".png"), cv2.IMREAD_UNCHANGED)
                        depth = np.flipud(depth)
                        cv2.imwrite(os.path.join(self.data_path, "depth",
                                                 frame_index_str + ".png"), depth)

                    with open(os.path.join(self.data_path,  "calibration.json"), 'w') as f:
                        json.dump(intr_data, f)

                elif "intrinsics" in intr_data[frame_index_str]["depth"]:
                    intrinsics = np.array(
                        intr_data[frame_index_str]["depth"]["intrinsics"]).reshape(4, 4)

        return intrinsics

    def load_calibrated_frame(self, frame_index):
        if self.use_empty_space_measurements:
            tmp = np.load(os.path.join(self.data_path, "calibrated_data", '{}_emptyfiltered.npy'.format(str(frame_index).zfill(6))),
                          allow_pickle=True)
            raw_data_cal = {
                "fvec": tmp.item().get("fvec"),
                "number_of_measurements": tmp.item().get("number_of_measurements"),
                "xr": tmp.item().get("xr"),
                "timestamp_us": tmp.item().get("timestamp_us")
            }
        else:
            tmp = np.load(os.path.join(self.data_path, "calibrated_data", '{}.npy'.format(str(frame_index).zfill(6))),
                          allow_pickle=True)
            raw_data_cal = {
                "fvec": tmp.item().get("fvec"),
                "number_of_measurements": tmp.item().get("number_of_measurements"),
                "xr": tmp.item().get("xr"),
                "timestamp_us": tmp.item().get("timestamp_us")
            }

        return raw_data_cal

    def read_radar_frames(self, filename=""):

        if not os.path.exists(os.path.join(self.data_path, "calibrated_data")):
            os.makedirs(os.path.join(self.data_path, "calibrated_data"))

        raw_data_cal = self.load_calibrated_frame(0)
        num_frames = int(raw_data_cal["number_of_measurements"])

        raw_data_cal = [raw_data_cal]
        for i in range(0, num_frames-1):
            data_cal_ = self.load_calibrated_frame(i+1)
            raw_data_cal.append(data_cal_)

        self.num_frames = num_frames

        # self.binary_file = binary_paths[0]
        self.radar_frames = raw_data_cal

    def get_timestamps(self):
        assert hasattr(self,
                       'radar_frames'), "You must call read_radar_frames() first"
        assert len(
            self.radar_frames) > 0, "You must call read_radar_frames() first"

        timestamps = [x["timestamp_us"][0].item()
                      * 1.0e-6 for x in self.radar_frames]

        return timestamps

    def reconstruct_positions(self, reco_positions, frame_index=0, reco_normals=None, use_normal_weights=False):

        raw_data_cal = self.radar_frames[frame_index]
        tx_positions = self.radar_client.Tx[1:95]
        rx_positions = self.radar_client.Rx[1:95]
        reco_data = raw_data_cal['xr'][1:95, 1:95, :]

        self.radar_client.initialize_reconstruction(
            tx_positions, rx_positions)
        reco = self.radar_client.reconstruct_custom_positions(
            reco_positions, reco_data, raw_data_cal['fvec'], reco_normals=reco_normals, use_normal_weights=use_normal_weights).reshape(-1, 1)
        rec_data_max = np.abs(reco)

        return rec_data_max.reshape(-1, 1)

    def reconstruct_positions_noaggregate(self, reco_positions, frame_index=0):
        raw_data_cal = self.radar_frames[frame_index]
        tx_positions = self.radar_client.Tx[1:95]
        rx_positions = self.radar_client.Rx[1:95]
        reco_data = raw_data_cal['xr'][1:95, 1:95, :]
        reco_freq = raw_data_cal['fvec'][:]

        zero_indices = np.stack(np.nonzero(np.all(reco_data == 0, axis=-1)), axis=0).reshape(
            2, -1).transpose(1, 0)
        valid_antennas = np.logical_not(np.all(reco_data == 0, axis=-1))
        # print("zero_indices: ")
        # for p in zero_indices:
        #     print(p)

        valid = reco_positions[:, 2] != 0
        num_tx = 94
        view_in = tx_positions.reshape(
            1, -1, 3) - reco_positions.copy()[:, None, :]
        in_length = np.zeros((view_in.shape[0], view_in.shape[1]))
        in_length[valid] = np.linalg.norm(
            view_in[valid], axis=-1).reshape(-1, num_tx)
        # view_in[valid] = view_in[valid] / \
        #     np.linalg.norm(view_in[valid], axis=-1)[:, :, None]
        # # view_in = view_in.reshape(-1, 3)
        # view_in = np.matmul(
        #     view_in[:, :, None, :], to_local.transpose(0, 2, 1)[:, None, :, :])[:, :, 0, :]
        # _, polar_in = view_to_spherical_polar(view_in.reshape(-1, 3))
        # polar_in = polar_in.reshape(-1, num_tx, 2)

        view_out = rx_positions.reshape(
            1, -1, 3) - reco_positions.copy()[:, None, :]
        out_length = np.zeros((view_out.shape[0], view_out.shape[1]))
        out_length[valid] = np.linalg.norm(
            view_out[valid], axis=-1).reshape(-1, num_tx)
        # view_out[valid] = view_out[valid] / \
        #     np.linalg.norm(view_out[valid], axis=-1)[:, :, None]
        # view_out = np.matmul(
        #     view_out[:, :, None, :], to_local.transpose(0, 2, 1)[:, None, :, :])[:, :, 0, :]
        # _, polar_out = view_to_spherical_polar(view_out.reshape(-1, 3))
        # polar_out = polar_out.reshape(-1, num_tx, 2)

        polar_index_mesh = np.stack(
            list(np.meshgrid(np.arange(0, num_tx, 1), np.arange(0, num_tx, 1))), axis=-1).reshape(-1, 2)

        length_mesh = np.stack(
            [out_length[:, polar_index_mesh[:, 1]], in_length[:, polar_index_mesh[:, 0]]], axis=-1)
        length_mesh = np.sum(length_mesh, axis=-1)

        max_length = np.max(length_mesh, axis=-1)
        min_length = np.min(length_mesh, axis=-1)

        length_mesh[max_length > 0] = ((length_mesh[max_length > 0] - min_length[max_length > 0, None]) /
                                       (max_length[max_length > 0, None] - min_length[max_length > 0, None]))
        length_mesh[max_length > 0] = 1 - length_mesh[max_length > 0]

        length_mesh = length_mesh.reshape(-1, 94, 94)
        length_mesh[:, np.logical_not(valid_antennas)] = 0

        self.radar_client.initialize_reconstruction(
            tx_positions, rx_positions)

        num_pts = 20000
        reco = []

        for start in np.arange(0, reco_positions.shape[0], num_pts):
            r = self.radar_client.reconstruct_custom_positions_noaggregate(
                reco_positions[start:(start+num_pts)], reco_data, reco_freq, use_normal_weights=False)
            reco.append(r)
        # num_pts x 94 x 94

        reco = np.concatenate(reco, axis=0)
        return reco, valid_antennas, length_mesh
        # reco_sum = np.abs(np.sum((reco), axis=(-1, -2)) / (94 * 94))
        # return reco_sum[:, None]

       # return np.abs(reco_sum)[:, None]

    def reconstruct_positions_hypotheses(self, reco_positions, frame_index=0):
        raw_data_cal = self.radar_frames[frame_index]
        tx_positions = self.radar_client.Tx[1:95]
        rx_positions = self.radar_client.Rx[1:95]
        reco_data = raw_data_cal['xr'][1:95, 1:95, :]
        reco_freq = raw_data_cal['fvec'][:]

        zero_indices = np.stack(np.nonzero(np.all(reco_data == 0, axis=-1)), axis=0).reshape(
            2, -1).transpose(1, 0)
        valid_antennas = np.logical_not(np.all(reco_data == 0, axis=-1))
        # print("zero_indices: ")
        # for p in zero_indices:
        #     print(p)

        self.radar_client.initialize_reconstruction(
            tx_positions, rx_positions)

        num_pts = 20000
        reco = []

        for start in np.arange(0, reco_positions.shape[0], num_pts):
            r = self.radar_client.reconstruct_custom_positions_hypotheses(
                reco_positions[start:(start+num_pts)], reco_data, reco_freq, use_normal_weights=False)
            reco.append(r)

        reco = np.concatenate(reco, axis=0)
        return np.abs(reco), valid_antennas
        # reco_sum = np.abs(np.sum((reco), axis=(-1, -2)) / (94 * 94))
        # return reco_sum[:, None]

       # return np.abs(reco_sum)[:, None]

    def reconstruct_positions_signal(self, reco_positions, frame_index=0):
        raw_data_cal = self.radar_frames[frame_index]
        tx_positions = self.radar_client.Tx[1:95]
        rx_positions = self.radar_client.Rx[1:95]
        reco_data = raw_data_cal['xr'][1:95, 1:95, :]
        reco_freq = raw_data_cal['fvec'][:]

        zero_indices = np.stack(np.nonzero(np.all(reco_data == 0, axis=-1)), axis=0).reshape(
            2, -1).transpose(1, 0)
        valid_antennas = np.logical_not(np.all(reco_data == 0, axis=-1))
        # print("zero_indices: ")
        # for p in zero_indices:
        #     print(p)

        self.radar_client.initialize_reconstruction(
            tx_positions, rx_positions)

        num_pts = 20000
        reco = []

        for start in np.arange(0, reco_positions.shape[0], num_pts):
            r = self.radar_client.reconstruct_custom_positions_signal(
                reco_positions[start:(start+num_pts)], reco_data, reco_freq, use_normal_weights=False)
            reco.append(r)

        reco = np.concatenate(reco, axis=0)
        return np.abs(reco), valid_antennas
        # reco_sum = np.abs(np.sum((reco), axis=(-1, -2)) / (94 * 94))
        # return reco_sum[:, None]

       # return np.abs(reco_sum)[:, None]

    def get_frame_coarse_to_fine(self,
                                 frame_index=0,
                                 xmin_c=-0.25,
                                 xmax_c=0.25,
                                 xsteps_c=100,
                                 ymin_c=-0.25,
                                 ymax_c=0.25,
                                 ysteps_c=100,
                                 zmin_c=0.25,
                                 zmax_c=0.5,
                                 zsteps_c=50,
                                 num_frames_coarse=1,
                                 amplitude_filter_threshold_dB=15,
                                 depth_filter_kernel_size=1,
                                 force_redo=False,
                                 save_depth=True,
                                 save_pc=True,
                                 save_amplitude=True,
                                 save_volume=False,
                                 precise=True):
        raw_data_cal = self.radar_frames[frame_index]
        tx_positions = self.radar_client.Tx[1:95]
        rx_positions = self.radar_client.Rx[1:95]

        frame_index_str = str(frame_index).zfill(6)
        intr_data = None
        if os.path.exists(os.path.join(self.data_path,  "calibration.json")):
            with open(os.path.join(self.data_path,  "calibration.json")) as f:
                intr_data = json.load(f)

        if (intr_data is None or (not frame_index_str in intr_data) or force_redo):
            print("Reconstruct coarse to fine....")

            holographic_x = np.linspace(xmin_c,
                                        xmax_c, xsteps_c)
            holographic_y = np.linspace(ymin_c,
                                        ymax_c, ysteps_c)
            holographic_z = np.linspace(zmin_c,
                                        zmax_c, zsteps_c)

            gap_x = holographic_x[1] - holographic_x[0]
            gap_y = holographic_y[1] - holographic_y[0]
            gap_z = holographic_z[1] - holographic_z[0]

            rec_config = {
                "xmin": xmin_c,
                "xmax": xmax_c,
                "xsteps": xsteps_c,
                "ymin": ymin_c,
                "ymax": ymax_c,
                "ysteps": ysteps_c,
                "zmin": zmin_c,
                "zmax": zmax_c,
                "zsteps": zsteps_c
            }
            self.set_config(**rec_config)

            if num_frames_coarse <= 1:
                radar_points, radar_depth, radar_intensity = self.get_frame(frame_index, force_redo=force_redo,
                                                                            use_intrinsic_parameters=not force_redo,
                                                                            amplitude_filter_threshold_dB=20,
                                                                            save_depth=False,
                                                                            save_pc=False,
                                                                            save_amplitude=False,
                                                                            save_volume=False)

                min_x = np.round(
                    radar_points[radar_points[:, :, 2] > 0, 0].min(), 2)
                max_x = np.round(
                    radar_points[radar_points[:, :, 2] > 0, 0].max(), 2)
                min_y = np.round(
                    radar_points[radar_points[:, :, 2] > 0, 1].min(), 2)
                max_y = np.round(
                    radar_points[radar_points[:, :, 2] > 0, 1].max(), 2)
                min_z = np.round(
                    radar_points[radar_points[:, :, 2] > 0, 2].min(), 2)
                max_z = np.round(
                    radar_points[radar_points[:, :, 2] > 0, 2].max(), 2)

            else:
                min_x = math.inf
                min_y = math.inf
                min_z = math.inf

                max_x = 0
                max_y = 0
                max_z = 0

                for i in range(frame_index, min(frame_index + num_frames_coarse, self.num_frames)):
                    radar_points, _, _ = self.get_frame(i, force_redo=force_redo,
                                                        use_intrinsic_parameters=not force_redo,
                                                        amplitude_filter_threshold_dB=20,
                                                        save_depth=False,
                                                        save_pc=False,
                                                        save_amplitude=False,
                                                        save_volume=False)
                    min_x = min(min_x, np.round(
                        radar_points[radar_points[:, :, 2] > 0, 0].min(), 2))
                    max_x = max(max_x, np.round(
                        radar_points[radar_points[:, :, 2] > 0, 0].max(), 2))
                    min_y = min(min_y, np.round(
                        radar_points[radar_points[:, :, 2] > 0, 1].min(), 2))
                    max_y = max(max_y, np.round(
                        radar_points[radar_points[:, :, 2] > 0, 1].max(), 2))
                    min_z = min(min_z, np.round(
                        radar_points[radar_points[:, :, 2] > 0, 2].min(), 2))
                    max_z = max(max_z, np.round(
                        radar_points[radar_points[:, :, 2] > 0, 2].max(), 2))

            min_x = min_x - gap_x
            max_x = max_x + gap_x
            min_y = min_y - gap_y
            max_y = max_y + gap_y
            min_z = min_z - gap_z
            max_z = max_z + gap_z

            if (precise):
                steps_x = min(int(np.ceil((max_x - min_x) / 0.001)), 200)
                steps_y = min(int(np.ceil((max_y - min_y) / 0.001)), 200)
                steps_z = min(int(np.ceil((max_z - min_z) / 0.001)), 200)
            else:
                steps_x = min(int(np.ceil((max_x - min_x) / 0.005)), 200)
                steps_y = min(int(np.ceil((max_y - min_y) / 0.005)), 200)
                steps_z = min(int(np.ceil((max_z - min_z) / 0.005)), 200)

            print("reconstructing in X: {}, {}, {}".format(min_x, max_x, steps_x))
            print("reconstructing in Y: {}, {}, {}".format(min_y, max_y, steps_y))
            print("reconstructing in Z: {}, {}, {}".format(min_z, max_z, steps_z))

            holographic_x = np.linspace(min_x, max_x, steps_x)
            holographic_y = np.linspace(min_y, max_y, steps_y)
            holographic_z = np.linspace(min_z, max_z, steps_z)

            rec_config = {
                "xmin": min_x,
                "xmax": max_x,
                "xsteps": steps_x,
                "ymin": min_y,
                "ymax": max_y,
                "ysteps": steps_y,
                "zmin": min_z,
                "zmax": max_z,
                "zsteps": steps_z
            }
            self.set_config(**rec_config)

            # self.radar_client.initialize_reconstruction(
            #     tx_positions, rx_positions, holographic_x, holographic_y, holographic_z)
            # self.radar_client.initialize_visualization()

            return self.get_frame(frame_index, force_redo=False,
                                  amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
                                  depth_filter_kernel_size=depth_filter_kernel_size,
                                  save_depth=save_depth,
                                  save_pc=save_pc,
                                  save_amplitude=save_amplitude,
                                  save_volume=save_volume)
        else:
            return self.get_frame(frame_index, force_redo=force_redo,
                                  use_intrinsic_parameters=True,
                                  amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
                                  depth_filter_kernel_size=depth_filter_kernel_size,
                                  save_depth=save_depth,
                                  save_pc=save_pc,
                                  save_amplitude=save_amplitude,
                                  save_volume=save_volume)

    def check_cached_information(self, frame_index, intr_data,
                                 amplitude_filter_threshold_dB=15,
                                 depth_filter_kernel_size=1):

        frame_index_str = str(frame_index).zfill(6)
        filename = str(frame_index).zfill(6)

        redo_reconstruction = False
        redo_filtering = False
        redo_reconstruction = intr_data is None or redo_reconstruction

        # compare stored metadata with current dataloader configurations
        if intr_data is not None and frame_index_str in intr_data and not redo_reconstruction and "volume" in intr_data[frame_index_str]:
            x_vals = np.linspace(intr_data[frame_index_str]["volume"]["bb_min"][0], intr_data[frame_index_str]
                                 ["volume"]["bb_max"][0], intr_data[frame_index_str]["volume"]["grid_size"][0])
            y_vals = np.linspace(intr_data[frame_index_str]["volume"]["bb_min"][1], intr_data[frame_index_str]
                                 ["volume"]["bb_max"][1], intr_data[frame_index_str]["volume"]["grid_size"][1])
            z_vals = np.linspace(intr_data[frame_index_str]["volume"]["bb_min"][2], intr_data[frame_index_str]
                                 ["volume"]["bb_max"][2], intr_data[frame_index_str]["volume"]["grid_size"][2])

            if abs(x_vals.min() - self.radar_client.grid_x_range.min()) > 1e-16:
                redo_reconstruction = True
            elif abs(x_vals.max() - self.radar_client.grid_x_range.max()) > 1e-16:
                redo_reconstruction = True
            elif abs(y_vals.min() - self.radar_client.grid_y_range.min()) > 1e-16:
                redo_reconstruction = True
            elif abs(y_vals.max() - self.radar_client.grid_y_range.max()) > 1e-16:
                redo_reconstruction = True
            elif abs(z_vals.min() - self.radar_client.grid_z_range.min()) > 1e-16:
                redo_reconstruction = True
            elif abs(z_vals.max() - self.radar_client.grid_z_range.max()) > 1e-16:
                redo_reconstruction = True
            elif x_vals.shape[0] != self.radar_client.grid_x_range.shape[0]:
                redo_reconstruction = True
            elif y_vals.shape[0] != self.radar_client.grid_y_range.shape[0]:
                redo_reconstruction = True
            elif z_vals.shape[0] != self.radar_client.grid_z_range.shape[0]:
                redo_reconstruction = True

            if frame_index_str in intr_data and "amplitude" in intr_data[frame_index_str] and "amplitude_filter_threshold_dB" in intr_data[frame_index_str]["amplitude"]:
                if intr_data[frame_index_str]["amplitude"]["amplitude_filter_threshold_dB"] != amplitude_filter_threshold_dB:
                    redo_filtering = True
                elif intr_data[frame_index_str]["amplitude"]["amplitude_filter_kernel_size"] != depth_filter_kernel_size:
                    redo_filtering = True

        # compare current dataloader configuration with stored depth
        if not redo_reconstruction and os.path.exists(os.path.join(self.data_path,  "depth", filename + ".png")):
            d = cv2.imread(os.path.join(self.data_path, "depth",
                                        filename + ".png"), cv2.IMREAD_UNCHANGED)
            min_d = d[d > 0].min() * 1e-3
            max_d = d.max() * 1e-3
            if (d.shape[0] != len(self.radar_client.grid_y_range)):
                redo_reconstruction = True
            if (d.shape[1] != len(self.radar_client.grid_x_range)):
                redo_reconstruction = True
            if ((min_d - self.radar_client.grid_z_range.min()) < -1e-2):
                redo_reconstruction = True
            if ((max_d - self.radar_client.grid_z_range.max()) > 1e-2):
                redo_reconstruction = True

        # XXX: This check takes a lot of time
        # compare current dataloader configuration with stored volume
        # if not redo_reconstruction and os.path.exists(os.path.join(self.data_path,  "volume", filename + ".ply")):
        #     reco = self.radar_client.load_volume(os.path.join(self.data_path, "volume", filename + ".ply"),
        #                                          meta_input_directory=self.data_path)

        #     if (reco.shape[0] != len(self.radar_client.grid_x_range)):
        #         redo_reconstruction = True
        #     if (reco.shape[1] != len(self.radar_client.grid_y_range)):
        #         redo_reconstruction = True
        #     if (reco.shape[2] != len(self.radar_client.grid_z_range)):
        #         redo_reconstruction = True

        return redo_reconstruction, redo_filtering

    def load_cache_depth(self, frame_index,
                         reco=None,
                         redo_reconstruction=False,
                         save_depth=False,
                         amplitude_filter_threshold_dB=15,
                         depth_filter_kernel_size=1):

        if frame_index == -1:
            frame_index = len(self.radar_frames) - 1

        assert hasattr(self,
                       'radar_frames'), "You must call read_radar_frames() first"
        assert len(
            self.radar_frames) > 0, "You must call read_radar_frames() first"

        frame_index_str = str(frame_index).zfill(6)
        filename = str(frame_index).zfill(6)
        raw_data_cal = self.radar_frames[frame_index]

        if not os.path.exists(os.path.join(self.data_path, "depth")) and save_depth:
            os.makedirs(os.path.join(self.data_path, "depth"))

        # print("Depth: redo: {}, save: {}".format(redo_reconstruction, save_depth))

        if (not os.path.exists(os.path.join(self.data_path, "depth", filename + ".png")) or redo_reconstruction):
            if reco is None:
                reco = self.radar_client.reconstruct(
                    raw_data_cal['xr'][1:95, 1:95, :], raw_data_cal['fvec'])
            if save_depth:
                z_coord = self.radar_client.save_depth(os.path.join(self.data_path, "depth", filename + ".png"),
                                                       reco,
                                                       depth_filter_kernel_size=depth_filter_kernel_size,
                                                       amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
                                                       visualize=False,
                                                       meta_output_directory=self.data_path,
                                                       force_save=True,
                                                       frequencies=raw_data_cal['fvec'])
            else:
                z_coord = self.radar_client.extract_depth(reco, frequencies=raw_data_cal['fvec'],
                                                          depth_filter_kernel_size=depth_filter_kernel_size,
                                                          amplitude_filter_threshold_dB=amplitude_filter_threshold_dB)

            depth = z_coord * 1000  # meter -> millimeter
            # depth = np.flipud(depth)
        else:

            # we call this explicitely to overwrite changes in the intrinsic matrix
            intrinsics = self.get_intrinsics(frame_index)
            depth = cv2.imread(os.path.join(self.data_path, "depth",
                                            filename + ".png"), cv2.IMREAD_UNCHANGED)
            # cv2.imshow("depth", depth / depth.max())
            # cv2.waitKey(0)

        self.last_reco.update({
            "reco": reco,
            "depth": depth
        })

        return reco, depth

    def load_cache_volume(self, frame_index,
                          reco=None,
                          redo_reconstruction=False,
                          save_volume=False):
        if frame_index == -1:
            frame_index = len(self.radar_frames) - 1

        assert hasattr(self,
                       'radar_frames'), "You must call read_radar_frames() first"
        assert len(
            self.radar_frames) > 0, "You must call read_radar_frames() first"

        frame_index_str = str(frame_index).zfill(6)
        filename = str(frame_index).zfill(6)
        raw_data_cal = self.radar_frames[frame_index]

        if not os.path.exists(os.path.join(self.data_path, "volume")) and save_volume:
            os.makedirs(os.path.join(self.data_path, "volume"))
        # print("Volume: redo: {}, save: {}".format(redo_reconstruction, save_volume))

        if (not os.path.exists(os.path.join(self.data_path, "volume", filename + ".ply")) or redo_reconstruction):
            if reco is None:
                reco = self.radar_client.reconstruct(
                    raw_data_cal['xr'][1:95, 1:95, :], raw_data_cal['fvec'])
            if save_volume:
                self.radar_client.save_volume(reco, os.path.join(self.data_path, "volume", filename + ".ply"), force_save=True,
                                              meta_output_directory=self.data_path)
        else:
            reco = self.radar_client.load_volume(os.path.join(self.data_path, "volume", filename + ".ply"),
                                                 meta_input_directory=self.data_path)

        self.last_reco.update({
            "reco": reco
        })

        return reco

    def load_cache_maxproj(self, frame_index,
                           reco=None,
                           redo_reconstruction=False,
                           save_amplitude=False):

        if frame_index == -1:
            frame_index = len(self.radar_frames) - 1

        assert hasattr(self,
                       'radar_frames'), "You must call read_radar_frames() first"
        assert len(
            self.radar_frames) > 0, "You must call read_radar_frames() first"

        frame_index_str = str(frame_index).zfill(6)
        filename = str(frame_index).zfill(6)
        raw_data_cal = self.radar_frames[frame_index]

        if not os.path.exists(os.path.join(self.data_path, "maxproj")) and save_amplitude:
            os.makedirs(os.path.join(self.data_path, "maxproj"))
        # print("MAXPROJ: redo: {}, save: {}".format(redo_reconstruction, save_amplitude))

        if (not os.path.exists(os.path.join(self.data_path, "maxproj", filename + ".png")) or redo_reconstruction):
            if reco is None:
                reco = self.radar_client.reconstruct(
                    raw_data_cal['xr'][1:95, 1:95, :], raw_data_cal['fvec'])

            if save_amplitude:
                max_proj = self.radar_client.save_maximum_proj(os.path.join(
                    self.data_path, "maxproj", filename + ".png"), reco)
            else:
                _, max_proj = self.radar_client.maximum_projection(reco)
        else:
            max_proj = cv2.imread(os.path.join(
                self.data_path, "maxproj",  filename + ".png"), cv2.IMREAD_UNCHANGED)
            max_proj = max_proj.astype(np.float32)

        self.last_reco.update({
            "maxproj": max_proj
        })

        return reco, max_proj

    def load_cache_pc(self, frame_index,
                      reco=None,
                      depth=None,
                      max_proj=None,
                      redo_reconstruction=False,
                      save_pc=False,
                      amplitude_filter_threshold_dB=15,
                      depth_filter_kernel_size=1):

        if frame_index == -1:
            frame_index = len(self.radar_frames) - 1

        assert hasattr(self,
                       'radar_frames'), "You must call read_radar_frames() first"
        assert len(
            self.radar_frames) > 0, "You must call read_radar_frames() first"

        frame_index_str = str(frame_index).zfill(6)
        filename = str(frame_index).zfill(6)
        raw_data_cal = self.radar_frames[frame_index]

        if not os.path.exists(os.path.join(self.data_path, "xyz")) and save_pc:
            os.makedirs(os.path.join(self.data_path, "xyz"))
        # print("PC: redo: {}, save: {}".format(redo_reconstruction, save_pc))

        # if (not os.path.exists(os.path.join(self.data_path, "xyz", filename + ".ply")) or redo_reconstruction):

        if depth is None:
            reco, depth = self.load_cache_depth(frame_index, reco=reco, redo_reconstruction=redo_reconstruction, save_depth=False,
                                                depth_filter_kernel_size=depth_filter_kernel_size,
                                                amplitude_filter_threshold_dB=amplitude_filter_threshold_dB)
        if max_proj is None:
            reco, max_proj = self.load_cache_maxproj(
                frame_index, reco=reco, redo_reconstruction=redo_reconstruction, save_amplitude=False)

        if save_pc:
            z_coord = depth * 1e-3
            # z_coord = np.flipud(z_coord)
            pc = self.radar_client.save_pointcloud_from_depth(os.path.join(
                self.data_path, "xyz", filename + ".ply"), z_coord, maxproj=max_proj)
        else:
            intrinsics = self.get_intrinsics(frame_index)
            pc = self.radar_client.pointcloud_from_depth_image(
                depth, intrinsics, visualize=False)
            # pc = pc[depth > 0]
        # else:
        #    pc = self.radar_client.load_pointcloud(os.path.join(self.data_path, "xyz", filename + ".ply"))

        self.last_reco.update({
            "reco": reco,
            "depth": depth,
            "maxproj": max_proj,
            "pointcloud": pc
        })

        return reco, depth, max_proj, pc

    def load_cache_pc_nomaxproj(self, frame_index,
                                reco=None,
                                redo_reconstruction=False,
                                save_pc=False,
                                amplitude_filter_threshold_dB=15,
                                depth_filter_kernel_size=1):

        if frame_index == -1:
            frame_index = len(self.radar_frames) - 1

        assert hasattr(self,
                       'radar_frames'), "You must call read_radar_frames() first"
        assert len(
            self.radar_frames) > 0, "You must call read_radar_frames() first"

        frame_index_str = str(frame_index).zfill(6)
        filename = str(frame_index).zfill(6)
        raw_data_cal = self.radar_frames[frame_index]

        if not redo_reconstruction:
            intr_data = None
            if (os.path.exists(os.path.join(self.data_path,  "calibration.json"))):
                with open(os.path.join(self.data_path,  "calibration.json"), 'r') as f:
                    intr_data = json.load(f)
            redo_reconstruction, redo_filtering = self.check_cached_information(frame_index, intr_data,
                                                                                amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
                                                                                depth_filter_kernel_size=depth_filter_kernel_size)
            print("---- Redo PC reco: {}, filtering: {}".format(
                redo_reconstruction, redo_filtering))

        if not os.path.exists(os.path.join(self.data_path, "xyz_nomaxproj")) and save_pc:
            os.makedirs(os.path.join(self.data_path, "xyz_nomaxproj"))
        # print("PC: redo: {}, save: {}".format(redo_reconstruction, save_pc))

        # if (not os.path.exists(os.path.join(self.data_path, "xyz", filename + ".ply")) or redo_reconstruction):

        if (not os.path.exists(os.path.join(self.data_path, "xyz_nomaxproj", filename + ".ply")) or redo_reconstruction):
            if reco is None:
                reco = self.radar_client.reconstruct(
                    raw_data_cal['xr'][1:95, 1:95, :], raw_data_cal['fvec'])
            pc, pc_amplitude = self.radar_client.extract_pointcloud_parabola_fit(reco, amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
                                                                                 depth_filter_kernel_size=depth_filter_kernel_size)
            if save_pc:
                self.radar_client.save_pointcloud(os.path.join(
                    self.data_path, "xyz_nomaxproj", filename + ".ply"), pc, pc_amplitude=pc_amplitude)
        else:
            pc, pc_amplitude = self.radar_client.load_pointcloud(
                os.path.join(self.data_path, "xyz_nomaxproj", filename + ".ply"))

        return pc, pc_amplitude

    def get_frame(self, frame_index=0,
                  force_redo=False,
                  amplitude_filter_threshold_dB=15,
                  depth_filter_kernel_size=1,
                  use_intrinsic_parameters=False,
                  save_depth=True,
                  save_pc=True,
                  save_amplitude=True,
                  save_volume=False,
                  averaging_factor=1,
                  use_mask=False,
                  manual_mask=None):
        if frame_index == -1:
            frame_index = len(self.radar_frames) - 1

        assert hasattr(self,
                       'radar_frames'), "You must call read_radar_frames() first"
        assert len(
            self.radar_frames) > 0, "You must call read_radar_frames() first"

        average = averaging_factor > 1
        averaging_factor = min(averaging_factor, len(self.radar_frames))
        frame_index_str = str(frame_index).zfill(6)
        raw_data_cal = self.radar_frames[frame_index]
        tx_positions = self.radar_client.Tx[1:95]
        rx_positions = self.radar_client.Rx[1:95]

        # filename = str(raw_data_cal["timestamp_us"][0].item())
        filename = str(frame_index).zfill(6)

        intr_data = None
        if (os.path.exists(os.path.join(self.data_path,  "calibration.json"))):
            with open(os.path.join(self.data_path,  "calibration.json"), 'r') as f:
                intr_data = json.load(f)

        if use_intrinsic_parameters and intr_data is None:
            print("Warning: The flag 'use_intrinsic_parameters' requires a previous reconstruction with a calibration.json file")
        if use_intrinsic_parameters and intr_data is not None and frame_index_str in intr_data and "volume" in intr_data[frame_index_str]:
            assert intr_data is not None, "The flag 'use_intrinsic_parameters' requires a previous reconstruction with a calibration.json file"
            holographic_x = np.linspace(intr_data[frame_index_str]["volume"]["bb_min"][0], intr_data[frame_index_str]
                                        ["volume"]["bb_max"][0], intr_data[frame_index_str]["volume"]["grid_size"][0])
            holographic_y = np.linspace(intr_data[frame_index_str]["volume"]["bb_min"][1], intr_data[frame_index_str]
                                        ["volume"]["bb_max"][1], intr_data[frame_index_str]["volume"]["grid_size"][1])
            holographic_z = np.linspace(intr_data[frame_index_str]["volume"]["bb_min"][2], intr_data[frame_index_str]
                                        ["volume"]["bb_max"][2], intr_data[frame_index_str]["volume"]["grid_size"][2])
            # amplitude_filter_threshold_dB = intr_data["amplitude_filter_threshold_dB"]
            # depth_filter_kernel_size = intr_data["amplitude_filter_kernel_size"]
        else:

            holographic_x = np.linspace(self.xmin,
                                        self.xmax, self.xsteps,)
            holographic_y = np.linspace(self.ymin,
                                        self.ymax, self.ysteps,)
            holographic_z = np.linspace(self.zmin,
                                        self.zmax, self.zsteps,)

        self.radar_client.initialize_reconstruction(
            tx_positions, rx_positions, holographic_x, holographic_y, holographic_z)

        redo_reconstruction = force_redo
        redo_filtering = force_redo
        if not force_redo:
            redo_reconstruction, redo_filtering = self.check_cached_information(frame_index, intr_data,
                                                                                amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
                                                                                depth_filter_kernel_size=depth_filter_kernel_size)
        print("---- Redo reco: {}, filtering: {}".format(
            redo_reconstruction, redo_filtering))

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        reco = None
        pc = None
        depth = None
        max_proj = None

        if save_volume or average:
            print("Load volume...")
            if not average:
                reco = self.load_cache_volume(
                    frame_index, reco=reco, redo_reconstruction=redo_reconstruction, save_volume=save_volume)
            else:
                num_frames = len(self.radar_frames)
                avg_reco = None
                num_avg = 0
                for i in np.arange(-averaging_factor//2, averaging_factor//2, 1):
                    index = frame_index + i
                    if (index < 0):
                        index = num_frames + index
                    if (index >= num_frames):
                        index = (index - num_frames)
                    print("Averaging, reconstructing index: ", index)
                    if i == frame_index:
                        reco = self.load_cache_volume(
                            index, reco=None, redo_reconstruction=redo_reconstruction, save_volume=save_volume)
                    else:
                        reco = self.load_cache_volume(
                            index, reco=None, redo_reconstruction=redo_reconstruction, save_volume=False)
                    if num_avg == 0:
                        avg_reco = reco
                    else:
                        avg_reco = avg_reco + reco
                    num_avg = num_avg + 1
                reco = avg_reco / num_avg
                redo_reconstruction = True

        print("load maxproj....")
        reco, max_proj = self.load_cache_maxproj(
            frame_index, reco=reco, redo_reconstruction=redo_reconstruction, save_amplitude=save_amplitude)

        print("load depth....")
        reco, depth = self.load_cache_depth(
            frame_index, reco=reco, redo_reconstruction=redo_reconstruction or redo_filtering,
            save_depth=save_depth, depth_filter_kernel_size=depth_filter_kernel_size,
            amplitude_filter_threshold_dB=amplitude_filter_threshold_dB)

        print("load pc....")
        reco, depth, max_proj, pc = self.load_cache_pc(
            frame_index, reco=reco, depth=depth, max_proj=max_proj,
            redo_reconstruction=redo_reconstruction or redo_filtering,
            save_pc=save_pc, depth_filter_kernel_size=depth_filter_kernel_size,
            amplitude_filter_threshold_dB=amplitude_filter_threshold_dB)

        visualize = False
        if visualize:
            # print("max_proj: ", max_proj.shape)
            # print("depth min: {}, max: {}".format(depth[depth > 0].min(), depth.max()))
            # print("max_proj min: {}, max: {}".format(max_proj.min(), max_proj.max()))
            # print("pc min: ({},{},{}), max: ({},{},{})".format(pc[:,0].min(), pc[:,1].min(), pc[:,2].min(),
            #                                                    pc[:,0].max(),pc[:,1].max(),pc[:,2].max()))
            cv2.imshow("Depth: ", depth / depth.max())
            cv2.imshow("max_proj: ", max_proj / max_proj.max())
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            pc_vis = o3d.geometry.PointCloud()
            pc_vis.points = o3d.utility.Vector3dVector(
                pc.reshape(-1, 3) * np.array([1, 1, -1]).reshape(1, 3))
            col = ((max_proj / max_proj.max())[depth > 0])
            pc_vis.colors = o3d.utility.Vector3dVector(
                np.stack([np.zeros_like(col), col, np.zeros_like(col)], axis=-1))

            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.add_geometry(pc_vis)
            opt = visualizer.get_render_option()
            opt.show_coordinate_frame = True
            visualizer.run()
            visualizer.destroy_window()

        return pc, depth, max_proj

    def compute_normal_from_pc(self, pc, depth):

        wpos_lin_x = pc.reshape(-1, 3).astype(np.float32)
        wpos_lin_y = pc.transpose(1, 0, 2).reshape(-1, 3).astype(np.float32)
        normal_x = []
        for i in range(0, 3):
            normal_x.append(np.convolve(
                wpos_lin_x[:, i], np.array([1, 0, -1]), mode='same'))
        normal_x = np.stack(
            normal_x, axis=-1).reshape(depth.shape[0], depth.shape[1], 3)
        normal_y = []
        for i in range(0, 3):
            normal_y.append(np.convolve(
                wpos_lin_y[:, i], np.array([1, 0, -1]), mode='same'))
        normal_y = np.stack(
            normal_y, axis=-1).reshape(depth.shape[1], depth.shape[0], 3).transpose(1, 0, 2)
        normal = np.cross(normal_x, normal_y) * \
            np.array([-1, -1, 1]).reshape(-1, 1, 3)

        normal_length = np.linalg.norm(normal, axis=-1)
        positive = normal_length > 0

        normal[positive] = normal[positive] / \
            normal_length[positive][:, None]

        return normal
