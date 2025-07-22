import json
import os
import numpy as np
import cv2

from maroon.sensors.radar_control import RadarControllerFSCW, RadarControllerFSK
import time
import math
import open3d as o3d
import matplotlib


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
                 reconstruction_method="fscw",
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

        if (reconstruction_method == "fscw"):
            self.radar_client = RadarControllerFSCW(frequency_low_ghz=frequency_low,
                                                    frequency_high_ghz=frequency_high,
                                                    frequency_points=frequency_points,
                                                    averaging=averaging)
        elif (reconstruction_method == "fsk"):
            self.radar_client = RadarControllerFSK(frequency_low_ghz=frequency_low,
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

            if "volume" in intr_data[frame_index_str]:
                bb_min = intr_data[frame_index_str]["volume"]["bb_min"]
                bb_max = intr_data[frame_index_str]["volume"]["bb_max"]
                grid_size = intr_data[frame_index_str]["volume"]["grid_size"]
                grid_size = [float(g-1) for g in grid_size]
                intrinsics = self.radar_client.get_intrinsics(grid_size=grid_size,
                                                              bb_min=bb_min,
                                                              bb_max=bb_max)
            elif "depth" in intr_data[frame_index_str] and "bb_min" in intr_data[frame_index_str]["depth"]:
                bb_min = intr_data[frame_index_str]["depth"]["bb_min"]
                bb_max = intr_data[frame_index_str]["depth"]["bb_max"]
                grid_size = intr_data[frame_index_str]["depth"]["grid_size"]
                grid_size = [float(g-1) for g in grid_size]
                intrinsics = self.radar_client.get_intrinsics(grid_size=grid_size,
                                                              bb_min=bb_min,
                                                              bb_max=bb_max)

            if "depth" in intr_data[frame_index_str]:
                if "transform" in intr_data[frame_index_str]["depth"]:
                    del intr_data[frame_index_str]["depth"]["transform"]
                    intr_data[frame_index_str]["depth"]["intrinsics"] = intrinsics.tolist()

                    if os.path.exists(os.path.join(self.data_path, "depth", frame_index_str + ".png")):
                        depth = cv2.imread(os.path.join(self.data_path, "depth",
                                                        frame_index_str + ".png"), cv2.IMREAD_UNCHANGED)
                        depth = np.flipud(depth)
                        cv2.imwrite(os.path.join(self.data_path, "depth",
                                                 frame_index_str + ".png"), depth)

                    with open(os.path.join(self.data_path,  "calibration.json"), 'w') as f:
                        json.dump(intr_data, f)

        if intrinsics is None:
            print("Warning: no intrinsics found for frame: {}, using config parameters".format(
                frame_index_str))

            intrinsics = self.radar_client.get_intrinsics(grid_size=(self.xsteps-1, self.ysteps-1),
                                                          bb_min=(
                self.xmin, self.ymin),
                bb_max=(self.xmax, self.ymax))

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
                         depth_filter_kernel_size=1,
                         frequency_indices=None,
                         rx_antenna_indices=None,
                         tx_antenna_indices=None,
                         last_z_guess=None,
                         ):

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
                                                       frequencies=raw_data_cal['fvec'],
                                                       frequency_indices=frequency_indices,
                                                       rx_antenna_indices=rx_antenna_indices,
                                                       tx_antenna_indices=tx_antenna_indices
                                                       )
            else:
                z_coord = self.radar_client.extract_depth(reco, _frequencies=raw_data_cal['fvec'],
                                                          depth_filter_kernel_size=depth_filter_kernel_size,
                                                          amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
                                                          frequency_indices=frequency_indices,
                                                          rx_antenna_indices=rx_antenna_indices,
                                                          tx_antenna_indices=tx_antenna_indices)

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
                          save_volume=False,
                          frequency_indices=None,
                          rx_antenna_indices=None,
                          tx_antenna_indices=None,
                          last_z_guess=None):
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
                    raw_data_cal['xr'][1:95, 1:95, :], raw_data_cal['fvec'],
                    frequency_indices=frequency_indices,
                    rx_antenna_indices=rx_antenna_indices,
                    tx_antenna_indices=tx_antenna_indices, last_z_guess=last_z_guess)
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
                           save_amplitude=False,
                           frequency_indices=None,
                           rx_antenna_indices=None,
                           tx_antenna_indices=None,
                           last_z_guess=None):

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
                    raw_data_cal['xr'][1:95, 1:95, :], raw_data_cal['fvec'],
                    frequency_indices=frequency_indices, rx_antenna_indices=rx_antenna_indices,
                    tx_antenna_indices=tx_antenna_indices, last_z_guess=last_z_guess)

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
                      depth_filter_kernel_size=1,
                      frequency_indices=None,
                      rx_antenna_indices=None,
                      tx_antenna_indices=None,
                      use_linear_pc=False,
                      last_z_guess=None
                      ):

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

        if (not os.path.exists(os.path.join(self.data_path, "xyz", filename + ".ply")) or redo_reconstruction or not use_linear_pc):
            if depth is None:
                reco, depth = self.load_cache_depth(frame_index, reco=reco, redo_reconstruction=redo_reconstruction, save_depth=False,
                                                    depth_filter_kernel_size=depth_filter_kernel_size,
                                                    amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
                                                    frequency_indices=frequency_indices,
                                                    rx_antenna_indices=rx_antenna_indices,
                                                    tx_antenna_indices=tx_antenna_indices,
                                                    last_z_guess=last_z_guess)
            if max_proj is None:
                reco, max_proj = self.load_cache_maxproj(
                    frame_index, reco=reco, redo_reconstruction=redo_reconstruction, save_amplitude=False,
                    frequency_indices=frequency_indices, rx_antenna_indices=rx_antenna_indices,
                    tx_antenna_indices=tx_antenna_indices, last_z_guess=last_z_guess)

            if save_pc:
                z_coord = depth * 1e-3
                # z_coord = np.flipud(z_coord)
                pc = self.radar_client.save_pointcloud_from_depth(os.path.join(
                    self.data_path, "xyz", filename + ".ply"), z_coord, maxproj=max_proj)

            else:

                if (self.xmin == self.xmax and self.xsteps == 1) or (self.ymin == self.ymax and self.ysteps == 1):
                    self.log(
                        "Warning: using one voxel creates a singularity in intrinsic matrix. Using internal voxel parameters for backprojection instead.")

                    x_coord = np.array([self.xmin])
                    y_coord = np.array([self.ymin])
                    z_coord = depth[0]
                    pc = np.stack([x_coord, y_coord, z_coord / 1000.0],
                                  axis=-1).reshape(1, 1, 3)
                else:
                    intrinsics = self.get_intrinsics(frame_index)
                    pc = self.radar_client.pointcloud_from_depth_image(
                        depth, intrinsics, visualize=False)

            if use_linear_pc:
                pc = pc[depth > 0]

                src_rgb = max_proj[depth > 0].reshape(-1)
                normalize = matplotlib.cm.colors.Normalize(
                    vmin=src_rgb.min(), vmax=src_rgb.max())
                # http://www.kennethmoreland.com/color-advice/
                # 'inferno', 'plasma', 'coolwarm'
                s_map = matplotlib.cm.ScalarMappable(
                    cmap=matplotlib.colormaps.get_cmap("viridis"), norm=normalize)
                src_rgb = s_map.to_rgba(src_rgb)[:, :3]
                max_proj = src_rgb
        else:
            pc, max_proj_colorized = self.radar_client.load_pointcloud(
                os.path.join(self.data_path, "xyz", filename + ".ply"))
            max_proj = max_proj_colorized

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
                                depth_filter_kernel_size=1,
                                frequency_indices=None,
                                rx_antenna_indices=None,
                                tx_antenna_indices=None,
                                last_z_guess=None):

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
                    raw_data_cal['xr'][1:95, 1:95, :], raw_data_cal['fvec'],
                    frequency_indices=frequency_indices,
                    rx_antenna_indices=rx_antenna_indices,
                    tx_antenna_indices=tx_antenna_indices, last_z_guess=last_z_guess)
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
                  manual_mask=None,
                  last_z_guess=None,
                  frequency_indices=None,
                  rx_antenna_indices=None,
                  tx_antenna_indices=None,
                  use_linear_pc=False
                  ):
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
                    frame_index, reco=reco, redo_reconstruction=redo_reconstruction, save_volume=save_volume,
                    frequency_indices=frequency_indices,
                    rx_antenna_indices=rx_antenna_indices,
                    tx_antenna_indices=tx_antenna_indices,
                    last_z_guess=last_z_guess
                )
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
                            index, reco=None, redo_reconstruction=redo_reconstruction, save_volume=save_volume,
                            frequency_indices=frequency_indices,
                            rx_antenna_indices=rx_antenna_indices,
                            tx_antenna_indices=tx_antenna_indices,
                            last_z_guess=last_z_guess
                        )
                    else:
                        reco = self.load_cache_volume(
                            index, reco=None, redo_reconstruction=redo_reconstruction, save_volume=False,
                            frequency_indices=frequency_indices,
                            rx_antenna_indices=rx_antenna_indices,
                            tx_antenna_indices=tx_antenna_indices,
                            last_z_guess=last_z_guess
                        )
                    if num_avg == 0:
                        avg_reco = reco
                    else:
                        avg_reco = avg_reco + reco
                    num_avg = num_avg + 1
                reco = avg_reco / num_avg
                redo_reconstruction = True

        print("load maxproj....")
        reco, max_proj = self.load_cache_maxproj(
            frame_index, reco=reco, redo_reconstruction=redo_reconstruction, save_amplitude=save_amplitude,
            frequency_indices=frequency_indices,
            rx_antenna_indices=rx_antenna_indices,
            tx_antenna_indices=tx_antenna_indices,
            last_z_guess=last_z_guess
        )

        print("load depth....")
        reco, depth = self.load_cache_depth(
            frame_index, reco=reco, redo_reconstruction=redo_reconstruction or redo_filtering,
            save_depth=save_depth, depth_filter_kernel_size=depth_filter_kernel_size,
            amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
            frequency_indices=frequency_indices,
            rx_antenna_indices=rx_antenna_indices,
            tx_antenna_indices=tx_antenna_indices,
            last_z_guess=last_z_guess
        )

        print("load pc....")
        reco, depth, max_proj, pc = self.load_cache_pc(
            frame_index, reco=reco, depth=depth, max_proj=max_proj,
            redo_reconstruction=redo_reconstruction or redo_filtering,
            save_pc=save_pc, depth_filter_kernel_size=depth_filter_kernel_size,
            amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
            frequency_indices=frequency_indices,
            rx_antenna_indices=rx_antenna_indices,
            tx_antenna_indices=tx_antenna_indices,
            use_linear_pc=use_linear_pc,
            last_z_guess=last_z_guess
        )

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
