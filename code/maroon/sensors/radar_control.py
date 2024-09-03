# from qarsc.reconstruct_data import holo_reconstruction_sfcw_cuda
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import serial
import os
import cv2
import json
import math
import time

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import convolution
import open3d as o3d
import shutil


from maroon.cuda.cuda import load_kernel_from_cu


class RadarControllerBase:
    def __init__(self,
                 frequency_low_ghz=72,
                 frequency_high_ghz=82,
                 frequency_points=128,
                 # directory where Rx.dat and Rx.dat are stored
                 antenna_root=os.path.join(
                     os.path.dirname(os.path.dirname(
                         os.path.dirname(os.path.dirname(__file__)))),
                     "data", "qar50sc"),
                 averaging=1):

        self.antenna_root = antenna_root
        self.frequency_low_ghz = frequency_low_ghz
        self.frequency_high_ghz = frequency_high_ghz
        self.frequency_points = frequency_points
        self.averaging = averaging

        self.connected = False
        self.capture_initialized = False
        self.reconstruction_initialized = False
        self.visualization_initialized = False
        self.modified_config = False

        assert os.path.exists(
            antenna_root), "Path does not exist: {}".format(antenna_root)

        self.Rx = np.loadtxt(os.path.join(antenna_root, "Rx.dat"),
                             skiprows=1, delimiter=",")
        self.Tx = np.loadtxt(os.path.join(antenna_root, "Tx.dat"),
                             skiprows=1, delimiter=",")

        # cuda.init()
        # pci_bus_id = 0  # 1st GPU
        # self.cuda_device = cuda.Device(pci_bus_id)
        # self.cuda_context = self.cuda_device.make_context()

    def set_frequency_range(self, frequency_low_ghz, frequency_high_ghz, frequency_points):
        """
            Modifies the frequency range
        """
        self.frequency_low_ghz = frequency_low_ghz
        self.frequency_low_ghz = frequency_high_ghz
        self.frequency_points = frequency_points

    def initialize_reconstruction(self, tx_positions,
                                  rx_positions,
                                  grid_x_range=np.linspace(-0.1, 0.1, 201),
                                  grid_y_range=np.linspace(-0.1, 0.1, 201),
                                  grid_z_range=np.linspace(0.23, 0.28, 5),
                                  ):
        """
            Initializes the signal reconstruction variables
            (Antenna positions, volume dimensions)
        """

        self.grid_x_range = grid_x_range.astype(np.float32)
        self.grid_y_range = grid_y_range.astype(np.float32)
        self.grid_z_range = grid_z_range.astype(np.float32)

        # we define the radar coordinate system like this:
        #    y
        #    ^
        #    |  z
        #    | /
        # ---/-------> x
        #    |
        # however, the reconstruction coordinate system looks like this:
        #          y
        #          ^
        #          |   z
        #          | /
        # x <------/-----
        #          |
        # so we reverse the antenna positions along the x-axis
        # to flip the x-axis
        rx_positions = rx_positions * np.array([-1, 1, 1]).reshape(1, 3)
        tx_positions = tx_positions * np.array([-1, 1, 1]).reshape(1, 3)

        self.tx_positions = np.ravel(tx_positions.astype(np.float32))
        self.rx_positions = np.ravel(rx_positions.astype(np.float32))

        self.num_tx_antennas = np.int32(len(tx_positions))
        self.num_rx_antennas = np.int32(len(rx_positions))

        self.reconstruction_initialized = True

    def reconstruct(self, meas_data_cal, frequencies, antenna_weights=None):
        assert False, "This is a Base class that does not have this method implemented. Try to use a specialized class"

    def reconstruct_custom_positions(self,  reco_positions, meas_data_cal,
                                     frequencies, antenna_weights=None, reco_normals=None,
                                     use_normal_weights=False):
        assert False, "This is a Base class that does not have this method implemented. Try to use a specialized class"

    def load_volume(self, filepath, meta_input_directory="", visualize=False):
        """
            Loads a volume that was stored as .ply before
            As the holography correlation factor is stored as RGB attribute in range [0,1],
            additional information about the maximum and minimum amplitude needs to be provided
            by a metadata file, which is usually stored along with the .ply file
        """
        volume = o3d.io.read_point_cloud(filepath)
        frame_index = os.path.basename(filepath).split(".")[0]

        if (len(meta_input_directory) != 0 and os.path.exists(meta_input_directory)):
            meta_dir = meta_input_directory
        else:
            meta_dir = os.path.dirname(filepath)

        range_data = None
        with open(os.path.join(meta_dir, "calibration.json"), 'r') as f:
            range_data = json.load(f)
        self.grid_x_range = np.linspace(range_data[frame_index]["depth"]["bb_min"][0], range_data[frame_index]
                                        ["depth"]["bb_max"][0], range_data[frame_index]["depth"]["grid_size"][0])
        self.grid_y_range = np.linspace(range_data[frame_index]["depth"]["bb_min"][1], range_data[frame_index]
                                        ["depth"]["bb_max"][1], range_data[frame_index]["depth"]["grid_size"][1])
        self.grid_z_range = np.linspace(range_data[frame_index]["depth"]["bb_min"][2], range_data[frame_index]
                                        ["depth"]["bb_max"][2], range_data[frame_index]["depth"]["grid_size"][2])
        reco_data = np.asarray(volume.colors).reshape(
            self.grid_x_range.shape[0], self.grid_y_range.shape[0], self.grid_z_range.shape[0], 3)
        reco_data = reco_data * \
            (range_data[frame_index]["depth"]["max_amplitude"] - range_data[frame_index]["depth"]["min_amplitude"]
             ) + range_data[frame_index]["depth"]["min_amplitude"]

        if visualize:
            pixel_x, pixel_y, pixel_z = np.meshgrid(
                self.grid_x_range, self.grid_y_range, self.grid_z_range)

            xyz = np.stack(
                [pixel_x, pixel_y, pixel_z], axis=-1)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(
                xyz.reshape(-1, 3))
            pc.colors = o3d.utility.Vector3dVector(
                reco_data.reshape(-1, 3))

        if visualize:
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.add_geometry(pc)
            opt = visualizer.get_render_option()
            opt.show_coordinate_frame = True
            visualizer.run()
            visualizer.destroy_window()

        return reco_data[:, :, :, 0]

    def save_volume(self, reco_image, filepath, visualize=False, meta_output_directory="", force_save=False):
        """
            Stores a reconstructed volume as .ply file
            As the holography correlation factor is stored as RGB attribute in range [0,1],
            additional information about the maximum and minimum amplitude needs to be stored
            in a metadata file 'calibration.json'
        """

        # reco_image shape: [x,y,z]
        assert os.path.exists(os.path.dirname(filepath))
        frame_index = os.path.basename(filepath).split(".")[0]

        if len(meta_output_directory) == 0 or not os.path.exists(meta_output_directory):
            intrinsics_file = os.path.join(os.path.dirname(
                filepath), "calibration.json")
        else:
            intrinsics_file = os.path.join(
                meta_output_directory, "calibration.json")
        if os.path.exists(intrinsics_file):
            with open(intrinsics_file, 'r') as f:
                try:
                    json_data = json.load(f)
                except Exception as e:
                    print(e)
                    json_data = {}
        else:
            json_data = {}

        pixel_x, pixel_y, pixel_z = np.meshgrid(
            self.grid_x_range, self.grid_y_range, self.grid_z_range)
        xyz = np.stack(
            [pixel_x, pixel_y, pixel_z], axis=-1)
        intensity = np.abs(reco_image)
        max_intensity = np.max(intensity).item()
        min_intensity = np.min(intensity).item()
        intensity = (np.abs(reco_image).reshape(-1) -
                     min_intensity) / (max_intensity - min_intensity)

        colors = np.stack([intensity, np.zeros_like(
            intensity), np.zeros_like(intensity)], axis=-1)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(
            xyz.reshape(-1, 3))

        pc.colors = o3d.utility.Vector3dVector(
            colors)

        if visualize:
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.add_geometry(pc)
            opt = visualizer.get_render_option()
            opt.show_coordinate_frame = True
            visualizer.run()
            visualizer.destroy_window()

        o3d.io.write_point_cloud(filepath, pc)
        with open(intrinsics_file, "w") as f:
            output = {}
            output[frame_index] = {}
            output[frame_index]["depth"] = {}
            output[frame_index]["depth"]["min_amplitude"] = min_intensity
            output[frame_index]["depth"]["max_amplitude"] = max_intensity
            output[frame_index]["depth"]["bb_min"] = [self.grid_x_range.min().item(
            ), self.grid_y_range.min().item(), self.grid_z_range.min().item()]
            output[frame_index]["depth"]["bb_max"] = [self.grid_x_range.max().item(
            ), self.grid_y_range.max().item(), self.grid_z_range.max().item()]
            output[frame_index]["depth"]["grid_size"] = [
                self.grid_x_range.shape[0], self.grid_y_range.shape[0], self.grid_z_range.shape[0]]

            output, ret = self.merge_meta_data(
                json_data, output, force=force_save)
            if ret == False:
                exit(0)

            json.dump(output, f, sort_keys=True)

    def merge_meta_data(self, meta_old, meta_new, force=False):
        meta_merged = {**meta_new}
        cont = force
        for k, v in meta_old.items():
            if k in meta_new:
                if type(v) is dict:
                    meta_merged[k], ret = self.merge_meta_data(
                        meta_old[k], meta_new[k], force=force)
                    if ret == False:
                        return meta_merged[k], ret
                elif type(v) is list:
                    for l1, l2 in zip(meta_old[k], meta_new[k]):
                        if l1 != l2 and not cont:
                            print("Found inconsistencies in updated metadata")
                            print("Old: {}".format(meta_old))
                            print("----------------------------------------")
                            print("New: {}".format(meta_new))
                            answer = input("****** Continue? (y/n)")
                            if answer.lower().strip() != "y":
                                return meta_merged, False
                            else:
                                cont = True
                else:
                    if meta_old[k] != meta_new[k] and not cont:
                        print("Found inconsistencies in updated metadata")
                        print("Old: {}".format(meta_old))
                        print("----------------------------------------")
                        print("New: {}".format(meta_new))
                        answer = input("****** Continue? (y/n)")
                        if answer.lower().strip() != "y":
                            return meta_merged, False
                        else:
                            cont = True
            else:
                meta_merged[k] = v
        return meta_merged, True

    def maximum_projection(self, reco_image, amplitude_filter_threshold_dB=0, visualize=False):
        """
            Extracts an amplitude map from the reconstructed volume using 'maximum projection'
            For each xy-coordinate the maximum amplitude along the z-axis is determined
        """
        if len(reco_image.shape) == 2:
            # only has x and y dimension, no maximum projection necessary
            rec_data_max = reco_image
        else:
            # np.abs returns the length of the vector defined by complex number
            rec_data_max = np.amax(np.abs(reco_image), axis=2)

        rec_data_max, intensity = self.filter_db(
            rec_data_max, filter_threshold_dB=amplitude_filter_threshold_dB)

        if visualize:
            fig = plt.figure("Show image after maximum projection")
            ax = plt.gca()
            im = plt.imshow(rec_data_max, vmin=np.nanmin(rec_data_max), vmax=np.nanmax(rec_data_max), extent=[self.grid_x_range.min(),
                                                                                                              self.grid_x_range.max(), self.grid_y_range.min(), self.grid_y_range.max()])
            plt.xlabel('$x$ (m)')
            plt.ylabel('$y$ (m)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label="Norm. Magnitude (dB)")
            plt.draw()

        return rec_data_max, intensity

    def filter_db(self, intensity_img, filter_threshold_dB=15, visualize=False):
        """
            Applies filtering using a dB threshold that is relative to the highest value
        """

        # normalize amplitude to [0, 20log10(max)] -> maximum is 0
        intensity = np.abs(intensity_img)
        rec_data_max = intensity.copy()
        rec_data_max /= np.max(rec_data_max)
        rec_data_max = 20 * np.log10(rec_data_max)

        if filter_threshold_dB > 0:
            rec_data_max[rec_data_max < -
                         filter_threshold_dB] = np.nan
            intensity[rec_data_max < -
                      filter_threshold_dB] = np.nan

        if visualize:
            plt.figure("Show image after setting threshold")
            plt.imshow(rec_data_max,
                       vmax=np.nanmax(rec_data_max), extent=[self.grid_x_range.min(), self.grid_x_range.max(), self.grid_y_range.min(), self.grid_y_range.max()])
            plt.xlabel('$x$ (m)')
            plt.ylabel('$y$ (m)')
            plt.colorbar(label="Norm. Magnitude (dB) Log Transformed")
            plt.draw()

        return rec_data_max, intensity

    def extract_depth(self, reco_image, frequencies=False, amplitude_filter_threshold_dB=15,
                      depth_filter_kernel_size=15, visualize=False):
        assert False, "This is a Base class that does not have this method implemented. Try to use a specialized class"

    def save_maximum_proj(self, filepath, reco_image):
        """
            Saves maximum projection as an image
        """

        if (len(os.path.dirname(filepath)) > 0):
            assert os.path.exists(os.path.dirname(
                filepath)), "Invalid path: {}".format(filepath)

        maximum_proj, intensity = self.maximum_projection(reco_image)
        maximum_proj_output = intensity
        maximum_proj_img = intensity.copy()  # save decimals
        max_val = np.max(maximum_proj_img)
        if max_val > pow(2, 16)-1:
            raise ValueError(
                "Error, maximum projection value exceeds 16-bit image: {}".format(max_val))
        maximum_proj = (maximum_proj_img).astype(np.uint16)
        cv2.imwrite(filepath, maximum_proj)

        return maximum_proj_output

    def save_depth(self, filepath, reco_image,
                   frequencies=None,
                   meta_output_directory="",
                   amplitude_filter_threshold_dB=15,
                   depth_filter_kernel_size=15, visualize=False,
                   force_save=False):
        """
            Saves depth map as an image. 
            To be able to reconstruct a point cloud from the depth image afterwards, additional
            information is stored inside am metadata file 'calibration.json'
            To project the depth image back into 3D, an 'intrinsic' matrix is provided, which is
            a concept usually applied in optics.

        """

        if (len(os.path.dirname(filepath)) > 0):
            assert os.path.exists(os.path.dirname(
                filepath)), "Invalid path: {}".format(filepath)
        frame_index = os.path.basename(filepath).split(".")[0]

        if len(meta_output_directory) == 0 or not os.path.exists(meta_output_directory):
            intrinsics_file = os.path.join(os.path.dirname(
                filepath), "calibration.json")
        else:
            intrinsics_file = os.path.join(
                meta_output_directory, "calibration.json")
        if os.path.exists(intrinsics_file):
            with open(intrinsics_file, 'r') as f:
                json_data = json.load(f)
        else:
            json_data = {}

        assert len(reco_image.shape) == 3
        z_coord = self.extract_depth(reco_image, frequencies=frequencies, depth_filter_kernel_size=depth_filter_kernel_size,
                                     amplitude_filter_threshold_dB=amplitude_filter_threshold_dB)

        # flip the y-coordinate such that it corresponds to the direction of
        # the storage layout of an image
        # our coordinate system:
        #          z
        #      | /
        # -----------> x
        #    / |
        #      |
        #      v
        #      y
        # depth = np.flipud(z_coord.copy())
        depth = z_coord.copy()

        depth = depth * 1000  # meter -> millimeter
        max_val = np.max(depth)
        if max_val > pow(2, 16)-1:
            raise ValueError(
                "Error, depth value exceeds 16-bit image: {}".format(max_val))
        depth = depth.astype(np.uint16)
        cv2.imwrite(filepath, depth)

        grid_size = [float(self.grid_x_range.shape[0]-1),
                     float(self.grid_y_range.shape[0]-1)]
        bb_min = [self.grid_x_range.min(), self.grid_y_range.min()]
        bb_max = [self.grid_x_range.max(), self.grid_y_range.max()]
        intrinsics = np.eye(4)
        intrinsics[0, 0] = grid_size[0] / (bb_max[0] - bb_min[0])
        intrinsics[0, 3] = bb_min[0]*grid_size[0] / (bb_min[0] - bb_max[0])
        intrinsics[1, 1] = grid_size[1] / (bb_min[1] - bb_max[1])
        intrinsics[1, 3] = bb_min[1]*grid_size[1] / \
            (bb_max[1] - bb_min[1]) + grid_size[1]
        # from millimeter to meter space
        intrinsics[2, 2] = 1000.0

        json_data_new = {
            frame_index: {
                "volume": {
                    "bb_min": [self.grid_x_range.min().item(), self.grid_y_range.min().item(), self.grid_z_range.min().item()],
                    "bb_max": [self.grid_x_range.max().item(), self.grid_y_range.max().item(), self.grid_z_range.max().item()],
                    "grid_size": [self.grid_x_range.shape[0], self.grid_y_range.shape[0], self.grid_z_range.shape[0]],
                    "max_abs": np.abs(reco_image).max().item(),
                    "min_abs": np.abs(reco_image).min().item(),
                    "max_real": np.real(reco_image).max().item(),
                    "min_real": np.real(reco_image).min().item(),
                    "max_imag": np.imag(reco_image).max().item(),
                    "min_imag": np.imag(reco_image).min().item()
                },
                "amplitude": {
                    "amplitude_filter_kernel_size": depth_filter_kernel_size,
                    "amplitude_filter_threshold_dB": amplitude_filter_threshold_dB,
                },
                "depth": {
                    "intrinsics": intrinsics.tolist(),
                },

            },
            "frequency_low_ghz": self.frequency_low_ghz,
            "frequency_high_ghz": self.frequency_high_ghz,
            "frequency_points": self.frequency_points,
        }
        json_data, ret = self.merge_meta_data(
            json_data, json_data_new, force=force_save)
        if ret == False:
            exit(0)

        if visualize:
            # debug reconstruction code to
            # reconstruct the pointcloud from depth map
            pixel_x, pixel_y = np.meshgrid(np.linspace(
                0, depth.shape[1]-1, depth.shape[1]), np.linspace(0, depth.shape[0]-1, depth.shape[0]))
            ones = np.ones_like(pixel_x)
            pixels = np.stack([pixel_x, pixel_y, depth, ones], axis=-1)
            intrinsics_inv = np.linalg.inv(intrinsics)
            xyz = np.matmul(pixels, intrinsics_inv.transpose())

            nonzero_mask = xyz[:, :, 2] > 0
            # shape: [X, 3]
            xyz_nonzero = xyz[nonzero_mask, :3]

            # matplotlib coordsys has z pointing forward -> reverse coordinates
            # for better visualization

            xyz_nonzero_z_rev = xyz_nonzero.copy()
            xyz_nonzero_z_rev[:, 2] = xyz_nonzero_z_rev[:, 2] * -1
            pointcloud_z_rev = o3d.geometry.PointCloud()
            pointcloud_z_rev.points = o3d.utility.Vector3dVector(
                xyz_nonzero_z_rev)

            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.add_geometry(pointcloud_z_rev)
            opt = visualizer.get_render_option()
            opt.show_coordinate_frame = True
            visualizer.run()
            visualizer.destroy_window()

        with open(intrinsics_file, "w") as f:
            json.dump(json_data, f, sort_keys=True)

        return z_coord

    def pointcloud_from_depth_image(self, depth, intrinsics=None, visualize=False):
        """
            Reconstructs a pointcloud from a previously stored depth image.
            To project the depth image back into 3D, an 'intrinsic' matrix is provided, which is
            a concept usually applied in optics.

            IMPORTANT: The intrinsic matrix assumes that the y-axis of the 3D coordinate system is pointing downwards
            As the RADAR reconstruction coordinate system has an y-axis pointing upwards, 
            the depth map needs to be flipped along the y-axis before projecting back into 3D.            
        """

        # cv2.imshow("depth", depth / depth.max())
        # cv2.waitKey(0)
        # flip the depth image
        # depth = np.flipud(depth)

        pixel_x, pixel_y = np.meshgrid(np.linspace(
            0, depth.shape[1]-1, depth.shape[1]), np.linspace(0, depth.shape[0]-1, depth.shape[0]))
        ones = np.ones_like(pixel_x)
        pixels = np.stack([pixel_x, pixel_y, depth, ones], axis=-1)

        if intrinsics is None:
            # intrinsics = np.eye(4)
            # # scale from pixels to world space coordinates
            # intrinsics[0, 0] = (self.grid_x_range.max(
            # ) - self.grid_x_range.min()) / float(self.grid_x_range.shape[0]-1)
            # intrinsics[1, 1] = (self.grid_y_range.max(
            # ) - self.grid_y_range.min()) / float(self.grid_y_range.shape[0]-1)
            # # from millimeter to meter space
            # intrinsics[2, 2] = 1.0 / 1000.0

            # # translate from pixels to world space coordinates
            # intrinsics[0, 3] = self.grid_x_range.min()
            # intrinsics[1, 3] = self.grid_y_range.min()

            grid_size = [float(self.grid_x_range.shape[0]-1),
                         float(self.grid_y_range.shape[0]-1)]
            bb_min = [self.grid_x_range.min(), self.grid_y_range.min()]
            bb_max = [self.grid_x_range.max(), self.grid_y_range.max()]
            intrinsics = np.eye(4)
            intrinsics[0, 0] = grid_size[0] / (bb_max[0] - bb_min[0])
            intrinsics[0, 3] = bb_min[0]*grid_size[0] / (bb_min[0] - bb_max[0])
            intrinsics[1, 1] = grid_size[1] / (bb_min[1] - bb_max[1])
            intrinsics[1, 3] = bb_min[1]*grid_size[1] / \
                (bb_max[1] - bb_min[1]) + grid_size[1]
            # from millimeter to meter space
            intrinsics[2, 2] = 1000.0

        intrinsics_inv = np.linalg.inv(intrinsics)
        xyz = np.matmul(pixels, intrinsics_inv.transpose())

        # xyz = np.flipud(xyz)
        # nonzero_mask = xyz[:, :, 2] > 0
        # shape: [X, 3]
        # xyz_nonzero = xyz[nonzero_mask, :3]

        if visualize:
            nonzero_mask = xyz[:, :, 2] > 0
            xyz_nonzero_z_rev = xyz[nonzero_mask, :3]
            xyz_nonzero_z_rev[:, 2] = xyz_nonzero_z_rev[:, 2] * -1
            pointcloud_z_rev = o3d.geometry.PointCloud()
            pointcloud_z_rev.points = o3d.utility.Vector3dVector(
                xyz_nonzero_z_rev)

            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.add_geometry(pointcloud_z_rev)
            opt = visualizer.get_render_option()
            opt.show_coordinate_frame = True
            visualizer.run()
            visualizer.destroy_window()

        return xyz[:, :, :3]

    def extract_pointcloud(self, z_coordinate, visualize=False):
        """
            Extracts a pointcloud from a depth map.
            In contrast to 'pointcloud_from_depth_image' this does not assume a stored depth
            image but instead directly takes the computed depth map
            that still lies within the RADAR coordinate system (y-axis pointing upwards)     
        """

        x_coords, y_coords = np.meshgrid(
            self.grid_x_range, self.grid_y_range[::-1])

        xyz = np.stack([x_coords, y_coords, z_coordinate], axis=-1)

        nonzero_mask = xyz[:, :, 2] > 0
        # shape: [X, 3]
        xyz_nonzero = xyz[nonzero_mask, :]
        # our coordinate system:
        #      y
        #      ^
        #      |   z
        #      | /
        # -----------> x
        #    / |
        #
        #

        if visualize:
            # matplotlib coordsys has z pointing forward -> reverse coordinates
            # for better visualization

            xyz_nonzero_z_rev = xyz_nonzero.copy()
            xyz_nonzero_z_rev[:, 2] = xyz_nonzero_z_rev[:, 2] * -1
            pointcloud_z_rev = o3d.geometry.PointCloud()
            pointcloud_z_rev.points = o3d.utility.Vector3dVector(
                xyz_nonzero_z_rev)

            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.add_geometry(pointcloud_z_rev)
            opt = visualizer.get_render_option()
            opt.show_coordinate_frame = True
            visualizer.run()
            visualizer.destroy_window()

        return xyz_nonzero

    def save_pointcloud_from_depth(self, filepath, z_coordinate, maxproj=None, visualize=False):
        """
            Store pointcloud from depth as .ply file
            In contrast to 'pointcloud_from_depth_image' this does not assume a stored depth
            image but instead directly takes the computed depth map
            that still lies within the RADAR coordinate system (y-axis pointing upwards)   
        """

        if (len(os.path.dirname(filepath)) > 0):
            assert os.path.exists(os.path.dirname(
                filepath)), "Invalid path: {}".format(filepath)

        x_coords, y_coords = np.meshgrid(
            self.grid_x_range, self.grid_y_range[::-1])

        xyz = np.stack([x_coords, y_coords, z_coordinate], axis=-1)

        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(xyz[xyz[:, :, 2] > 0])
        if maxproj is not None:
            m = maxproj.max()
            cols = maxproj[xyz[:, :, 2] > 0] / m
            pointcloud.colors = o3d.utility.Vector3dVector(
                np.stack([np.zeros_like(cols), cols, np.zeros_like(cols)], axis=-1))
        o3d.io.write_point_cloud(filepath, pointcloud)

        return xyz

    def load_pointcloud(self, filepath):
        """
            Loads a pointcloud that was stored as .ply before
        """
        pc = o3d.io.read_point_cloud(filepath)
        return np.asarray(pc.points).reshape(-1, 3), np.asarray(pc.colors).reshape(-1, 3)[:, 1]

    def save_pointcloud(self, filepath, pc, pc_amplitude=None, visualize=False):
        """
            Store pointcloud directly as .ply file
        """

        if (len(os.path.dirname(filepath)) > 0):
            assert os.path.exists(os.path.dirname(
                filepath)), "Invalid path: {}".format(filepath)

        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3))
        if pc_amplitude is not None:
            m = pc_amplitude.max()
            cols = (pc_amplitude / m).reshape(-1)
            pointcloud.colors = o3d.utility.Vector3dVector(
                np.stack([np.zeros_like(cols), cols, np.zeros_like(cols)], axis=-1))
        o3d.io.write_point_cloud(filepath, pointcloud)


class RadarControllerFSCW(RadarControllerBase):
    def __init__(self,
                 frequency_low_ghz=72,
                 frequency_high_ghz=82,
                 frequency_points=128,
                 # directory where Rx.dat and Rx.dat are stored
                 antenna_root=os.path.join(
                     os.path.dirname(os.path.dirname(
                         os.path.dirname(os.path.dirname(__file__)))),
                     "data", "qar50sc"),
                 averaging=1):

        RadarControllerBase.__init__(self, frequency_low_ghz=frequency_low_ghz,
                                     frequency_high_ghz=frequency_high_ghz, frequency_points=frequency_points,
                                     antenna_root=antenna_root,
                                     averaging=averaging)

    # overwrites method of 'RadarControllerBase'
    def reconstruct(self, meas_data_cal, frequencies, antenna_weights=None):
        """
            Performs SFCW reconstruction by generating hypotheses within a voxel grid
        """
        assert self.reconstruction_initialized, "You need to call initialize_reconstruction() before this method"

        # start = time.time()
        # res = self.holo_reconstruction_sfcw_cuda_slow(
        #     meas_data_cal, frequencies, antenna_weights=antenna_weights)
        # end = time.time()
        # print("Elapsed time (naive): ", end-start)
        start = time.time()
        res = self.holo_reconstruction_sfcw_cuda_fast(
            meas_data_cal, frequencies, antenna_weights=antenna_weights)
        end = time.time()
        print("Elapsed time (optimized): ", end-start)
        return res

    def holo_reconstruction_sfcw_cuda_slow(self,  meas_data_cal, frequencies, antenna_weights=None):
        assert self.reconstruction_initialized, "You need to call initialize_reconstruction() before this method"

        # print("Reconstruction in x-range: {}".format(self.grid_x_range))
        # print("Reconstruction in y-range: {}".format(self.grid_y_range))
        # print("Reconstruction in z-range: {}".format(self.grid_z_range))

        mod = load_kernel_from_cu(os.path.join(
            os.path.dirname(__file__), "cuda", "holo_reco_sfcw.cu"))

        reco_image = np.zeros(
            (self.grid_x_range.shape[0], self.grid_y_range.shape[0], self.grid_z_range.shape[0]), dtype=np.complex64)
        reco_image = np.ravel(reco_image)
        reco_image_pcf = np.zeros(
            (self.grid_x_range.shape[0], self.grid_y_range.shape[0], self.grid_z_range.shape[0], self.num_rx_antennas), dtype=np.complex64)
        reco_image_pcf = np.ravel(reco_image_pcf)

        time_signal = np.complex64(meas_data_cal)
        time_signal_length = np.int32(len(frequencies))
        frequencies = np.float32(frequencies)

        # copy stuff to gpu
        grid_x_range_gpu = cuda.mem_alloc(self.grid_x_range.nbytes)
        grid_y_range_gpu = cuda.mem_alloc(self.grid_y_range.nbytes)
        grid_z_range_gpu = cuda.mem_alloc(self.grid_z_range.nbytes)

        frequencies_gpu = cuda.mem_alloc(frequencies.nbytes)
        tx_positions_gpu = cuda.mem_alloc(self.tx_positions.nbytes)
        rx_positions_gpu = cuda.mem_alloc(self.rx_positions.nbytes)
        reco_image_gpu = cuda.mem_alloc(reco_image.nbytes)
        reco_image_pcf_gpu = cuda.mem_alloc(reco_image_pcf.nbytes)

        time_signal = np.ravel(time_signal)
        if antenna_weights is not None:
            time_signal = np.complex64(
                time_signal*np.ravel(np.repeat(antenna_weights, len(frequencies))))
            # time_signal = time_signal

        time_signal_gpu = cuda.mem_alloc(time_signal.nbytes)

        cuda.memcpy_htod(grid_x_range_gpu, self.grid_x_range)

        cuda.memcpy_htod(tx_positions_gpu, self.tx_positions)
        cuda.memcpy_htod(rx_positions_gpu, self.rx_positions)
        cuda.memcpy_htod(reco_image_gpu, reco_image)
        cuda.memcpy_htod(reco_image_pcf_gpu, reco_image_pcf)
        cuda.memcpy_htod(time_signal_gpu, time_signal)
        cuda.memcpy_htod(grid_y_range_gpu, self.grid_y_range)
        cuda.memcpy_htod(grid_z_range_gpu, self.grid_z_range)
        cuda.memcpy_htod(frequencies_gpu, frequencies)

        func = mod.get_function("holo_reco_sfcw_time_domain")

        x_steps = np.int32(len(self.grid_x_range))
        y_steps = np.int32(len(self.grid_y_range))
        z_steps = np.int32(len(self.grid_z_range))

        block_size = (8, 8, 8)
        grid_size = (int(math.ceil(len(self.grid_x_range) / block_size[0])),
                     int(math.ceil(len(self.grid_y_range) / block_size[1])),
                     int(math.ceil(len(self.grid_z_range) / block_size[2])))

        func(
            tx_positions_gpu,
            self.num_tx_antennas,
            rx_positions_gpu,
            self.num_rx_antennas,
            grid_x_range_gpu,
            x_steps,
            grid_y_range_gpu,
            y_steps,
            grid_z_range_gpu,
            z_steps,
            reco_image_gpu,
            reco_image_pcf_gpu,
            time_signal_gpu,
            time_signal_length,
            frequencies_gpu,
            block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(reco_image, reco_image_gpu)
        cuda.memcpy_dtoh(reco_image_pcf, reco_image_pcf_gpu)

        reco_image = reco_image.reshape(
            (self.grid_x_range.shape[0], self.grid_y_range.shape[0], self.grid_z_range.shape[0]))
        reco_image = reco_image / (self.num_rx_antennas*self.num_tx_antennas)

        # as the X-Axis is the first component here
        # the coordinate system currently looks like this:
        #       z
        #      /
        # ---/-------> y
        #    |
        #    |
        #    v
        #    x
        # so we have to rotate the volume to get a coordinate system
        # like this
        #    y
        #    ^
        #    |  z
        #    | /
        # ---/-------> x
        #    |
        reco_image = np.rot90(reco_image, k=1, axes=(0, 1))

        # Vanessa: Free variables
        tx_positions_gpu.free()
        rx_positions_gpu.free()
        grid_x_range_gpu.free()
        grid_y_range_gpu.free()
        grid_z_range_gpu.free()
        reco_image_gpu.free()
        reco_image_pcf_gpu.free()
        time_signal_gpu.free()
        frequencies_gpu.free()

        return reco_image

    def holo_reconstruction_sfcw_cuda_fast(self,  meas_data_cal, frequencies, antenna_weights=None):
        assert self.reconstruction_initialized, "You need to call initialize_reconstruction() before this method"

        # print("Reconstruction in x-range: {}".format(self.grid_x_range))
        # print("Reconstruction in y-range: {}".format(self.grid_y_range))
        # print("Reconstruction in z-range: {}".format(self.grid_z_range))

        mod = load_kernel_from_cu(os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "cuda", "holo_reco_sfcw_optimized.cu"))

        reco_image = np.zeros(
            (self.grid_x_range.shape[0], self.grid_y_range.shape[0], self.grid_z_range.shape[0]), dtype=np.complex64)
        reco_image = np.ravel(reco_image)
        reco_image_pcf = np.zeros(
            (self.grid_x_range.shape[0], self.grid_y_range.shape[0], self.grid_z_range.shape[0], self.num_rx_antennas), dtype=np.complex64)
        reco_image_pcf = np.ravel(reco_image_pcf)

        time_signal = np.complex64(meas_data_cal)
        time_signal_length = np.int32(len(frequencies))
        frequencies = np.float32(frequencies)

        # copy stuff to gpu
        grid_x_range_gpu = cuda.mem_alloc(self.grid_x_range.nbytes)
        grid_y_range_gpu = cuda.mem_alloc(self.grid_y_range.nbytes)
        grid_z_range_gpu = cuda.mem_alloc(self.grid_z_range.nbytes)

        frequencies_gpu = cuda.mem_alloc(frequencies.nbytes)
        tx_positions_gpu = cuda.mem_alloc(self.tx_positions.nbytes)
        rx_positions_gpu = cuda.mem_alloc(self.rx_positions.nbytes)
        reco_image_gpu = cuda.mem_alloc(reco_image.nbytes)

        # transpose foe efficient data order on GPU
        time_signal = np.ravel(np.transpose(time_signal, (2, 0, 1)))
        if antenna_weights is not None:
            time_signal = np.complex64(
                time_signal*np.ravel(np.repeat(antenna_weights, len(frequencies))))
            # time_signal = time_signal

        time_signal_gpu = cuda.mem_alloc(time_signal.nbytes)

        c = 299792458.0

        cuda.memcpy_htod(grid_x_range_gpu, self.grid_x_range / c)
        cuda.memcpy_htod(tx_positions_gpu, self.tx_positions / c)
        cuda.memcpy_htod(rx_positions_gpu, self.rx_positions / c)
        cuda.memcpy_htod(reco_image_gpu, reco_image)
        cuda.memcpy_htod(time_signal_gpu, time_signal)
        cuda.memcpy_htod(grid_y_range_gpu, self.grid_y_range / c)
        cuda.memcpy_htod(grid_z_range_gpu, self.grid_z_range / c)
        addr = mod.get_global("frequencies")
        # move frequencies to __const__ memory
        cuda.memcpy_htod(addr[0], frequencies * 2 * math.pi)

        func = mod.get_function("holo_reco_sfcw_time_domain")

        x_steps = np.int32(len(self.grid_x_range))
        y_steps = np.int32(len(self.grid_y_range))
        z_steps = np.int32(len(self.grid_z_range))

        block_size = (32, 8, 1)  # 1 warp per point, 4 warps pers block
        grid_size = (int(x_steps),
                     int(math.ceil(y_steps / block_size[1])),
                     int(z_steps))

        func(
            tx_positions_gpu,
            self.num_tx_antennas,
            rx_positions_gpu,
            self.num_rx_antennas,
            grid_x_range_gpu,
            x_steps,
            grid_y_range_gpu,
            y_steps,
            grid_z_range_gpu,
            z_steps,
            reco_image_gpu,
            time_signal_gpu,
            time_signal_length,
            block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(reco_image, reco_image_gpu)

        reco_image = reco_image.reshape(
            (self.grid_x_range.shape[0], self.grid_y_range.shape[0], self.grid_z_range.shape[0]))
        reco_image = reco_image / (self.num_rx_antennas*self.num_tx_antennas)

        # as the X-Axis is the first component here
        # the coordinate system currently looks like this:
        #       z
        #      /
        # ---/-------> y
        #    |
        #    |
        #    v
        #    x
        # so we have to rotate the volume to get a coordinate system
        # like this
        #    y
        #    ^
        #    |  z
        #    | /
        # ---/-------> x
        #    |
        reco_image = np.rot90(reco_image, k=1, axes=(0, 1))

        tx_positions_gpu.free()
        rx_positions_gpu.free()
        grid_x_range_gpu.free()
        grid_y_range_gpu.free()
        grid_z_range_gpu.free()
        reco_image_gpu.free()
        time_signal_gpu.free()

        return reco_image

    def reconstruct_custom_positions(self,  reco_positions, meas_data_cal,
                                     frequencies, antenna_weights=None, reco_normals=None,
                                     use_normal_weights=False):
        """
            In contrast to the regular SFCW reconstruction, this method does not use 
            a voxel grid but instead takes 'reco_positions' (array of size Nx3) directly as
            input for generating hypotheses
        """
        assert self.reconstruction_initialized, "You need to call initialize_reconstruction() before this method"

        # print("Reconstruction for points: {} ({})".format(
        #     reco_positions, reco_positions.shape))

        mod = load_kernel_from_cu(os.path.join(
            os.path.dirname(__file__), "cuda", "holo_reco_sfcw_optimized.cu"))

        time_signal = np.complex64(meas_data_cal)
        time_signal_length = np.int32(len(frequencies))
        frequencies = np.float32(frequencies)
        reco_image = np.zeros((reco_positions.shape[0],), dtype=np.complex64)
        if reco_normals is not None:
            reco_positions = np.concatenate(
                [reco_positions, reco_normals], axis=0).reshape(-1).astype(np.float32)
        else:
            reco_positions = reco_positions.reshape(-1).astype(np.float32)

        # reco_normals = reco_normals.reshape(-1).astype(np.float32)
        # reco_normals_gpu = cuda.mem_alloc(reco_normals.nbytes)
        # cuda.memcpy_htod(reco_normals_gpu, reco_normals)

        # copy stuff to gpu
        reco_positions_gpu = cuda.mem_alloc(reco_positions.nbytes)
        frequencies_gpu = cuda.mem_alloc(frequencies.nbytes)
        tx_positions_gpu = cuda.mem_alloc(self.tx_positions.nbytes)
        rx_positions_gpu = cuda.mem_alloc(self.rx_positions.nbytes)
        reco_image_gpu = cuda.mem_alloc(reco_image.nbytes)

        time_signal = np.ravel(time_signal)
        if antenna_weights is not None:
            time_signal = np.complex64(
                time_signal*np.ravel(np.repeat(antenna_weights, len(frequencies))))
            # time_signal = time_signal

        time_signal_gpu = cuda.mem_alloc(time_signal.nbytes)

        cuda.memcpy_htod(tx_positions_gpu, self.tx_positions)
        cuda.memcpy_htod(rx_positions_gpu, self.rx_positions)
        # cuda.memcpy_htod(reco_image_gpu, reco_image)
        cuda.memcpy_htod(reco_positions_gpu, reco_positions)
        cuda.memcpy_htod(time_signal_gpu, time_signal)
        cuda.memcpy_htod(frequencies_gpu, frequencies)

        if (use_normal_weights):
            assert reco_normals is not None
            func = mod.get_function(
                "holo_reco_sfcw_time_domain_custom_positions_weighted")
        else:
            func = mod.get_function(
                "holo_reco_sfcw_time_domain_custom_positions")

        reco_positions_size = np.int32(len(reco_image))

        block_size = (32, 1, 1)
        grid_size = (int(math.ceil(reco_positions_size / block_size[0])),
                     1, 1)

        func(
            tx_positions_gpu,
            self.num_tx_antennas,
            rx_positions_gpu,
            self.num_rx_antennas,
            reco_positions_gpu,
            reco_positions_size,
            reco_image_gpu,
            time_signal_gpu,
            time_signal_length,
            frequencies_gpu,
            block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(reco_image, reco_image_gpu)

        reco_image = reco_image / (self.num_rx_antennas*self.num_tx_antennas)

        tx_positions_gpu.free()
        rx_positions_gpu.free()
        reco_positions_gpu.free()
        reco_image_gpu.free()
        time_signal_gpu.free()
        frequencies_gpu.free()
        # reco_normals_gpu.free()

        return reco_image

    def reconstruct_custom_positions_noaggregate(self,  reco_positions, meas_data_cal,
                                                 frequencies, antenna_weights=None, reco_normals=None,
                                                 use_normal_weights=False):
        """
            In contrast to the regular SFCW reconstruction, this method does not use 
            a voxel grid but instead takes 'reco_positions' (array of size Nx3) directly as
            input for generating hypotheses
        """
        assert self.reconstruction_initialized, "You need to call initialize_reconstruction() before this method"

        # print("Reconstruction for points: {} ({})".format(
        #     reco_positions, reco_positions.shape))

        mod = load_kernel_from_cu(os.path.join(
            os.path.dirname(__file__), "cuda", "holo_reco_sfcw_optimized.cu"))

        # time_signal = np.complex64(meas_data_cal.transpose(2, 0, 1))
        time_signal = np.complex64(meas_data_cal)
        time_signal_length = np.int32(len(frequencies))
        frequencies = np.float32(frequencies)
        reco_image = np.zeros(
            (reco_positions.shape[0], self.rx_positions.shape[0] // 3, self.tx_positions.shape[0] // 3), dtype=np.complex64).reshape(-1)

        reco_positions_size = np.int32((reco_positions.shape[0]))
        if reco_normals is not None:
            reco_positions = np.concatenate(
                [reco_positions, reco_normals], axis=0).reshape(-1).astype(np.float32)
        else:
            reco_positions = reco_positions.reshape(-1).astype(np.float32)

        c = 299792458.0
        # reco_normals = reco_normals.reshape(-1).astype(np.float32)
        # reco_normals_gpu = cuda.mem_alloc(reco_normals.nbytes)
        # cuda.memcpy_htod(reco_normals_gpu, reco_normals)

        # copy stuff to gpu
        reco_positions_gpu = cuda.mem_alloc(reco_positions.nbytes)
        frequencies_gpu = cuda.mem_alloc(frequencies.nbytes)
        tx_positions_gpu = cuda.mem_alloc(self.tx_positions.nbytes)
        rx_positions_gpu = cuda.mem_alloc(self.rx_positions.nbytes)
        reco_image_gpu = cuda.mem_alloc(reco_image.nbytes)

        time_signal = np.ravel(time_signal)
        if antenna_weights is not None:
            time_signal = np.complex64(
                time_signal*np.ravel(np.repeat(antenna_weights, len(frequencies))).transpose(2, 0, 1))
            # time_signal = time_signal

        time_signal_gpu = cuda.mem_alloc(time_signal.nbytes)

        cuda.memcpy_htod(tx_positions_gpu, self.tx_positions / c)
        cuda.memcpy_htod(rx_positions_gpu, self.rx_positions / c)
        cuda.memcpy_htod(reco_positions_gpu, reco_positions / c)
        cuda.memcpy_htod(time_signal_gpu, time_signal)
        addr = mod.get_global("frequencies")
        # move frequencies to __const__ memory
        cuda.memcpy_htod(addr[0], frequencies * 2 * math.pi)

        func = mod.get_function(
            "holo_reco_sfcw_time_domain_custom_positions_noaggregate")

        block_size = (32, 8, 1)
        grid_size = (1,
                     int(math.ceil(reco_positions_size / block_size[1])), 1)

        func(
            tx_positions_gpu,
            self.num_tx_antennas,
            rx_positions_gpu,
            self.num_rx_antennas,
            reco_positions_gpu,
            reco_positions_size,
            reco_image_gpu,
            time_signal_gpu,
            time_signal_length,
            block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(reco_image, reco_image_gpu)

        reco_image = reco_image.reshape(
            reco_positions_size, self.rx_positions.shape[0] // 3, self.tx_positions.shape[0] // 3)

        tx_positions_gpu.free()
        rx_positions_gpu.free()
        reco_positions_gpu.free()
        reco_image_gpu.free()
        time_signal_gpu.free()
        frequencies_gpu.free()
        # reco_normals_gpu.free()

        return reco_image

    def reconstruct_custom_positions_hypotheses(self,  reco_positions, meas_data_cal,
                                                frequencies, antenna_weights=None, reco_normals=None,
                                                use_normal_weights=False):
        """
            In contrast to the regular SFCW reconstruction, this method does not use 
            a voxel grid but instead takes 'reco_positions' (array of size Nx3) directly as
            input for generating hypotheses
        """
        assert self.reconstruction_initialized, "You need to call initialize_reconstruction() before this method"

        # print("Reconstruction for points: {} ({})".format(
        #     reco_positions, reco_positions.shape))

        mod = load_kernel_from_cu(os.path.join(
            os.path.dirname(__file__), "cuda", "holo_reco_sfcw_optimized.cu"))

        # time_signal = np.complex64(meas_data_cal.transpose(2, 0, 1))
        time_signal = np.complex64(meas_data_cal)
        time_signal_length = np.int32(len(frequencies))
        frequencies = np.float32(frequencies)
        reco_image = np.zeros(
            (reco_positions.shape[0], self.rx_positions.shape[0] // 3, self.tx_positions.shape[0] // 3), dtype=np.complex64).reshape(-1)

        reco_positions_size = np.int32((reco_positions.shape[0]))
        if reco_normals is not None:
            reco_positions = np.concatenate(
                [reco_positions, reco_normals], axis=0).reshape(-1).astype(np.float32)
        else:
            reco_positions = reco_positions.reshape(-1).astype(np.float32)

        c = 299792458.0
        # reco_normals = reco_normals.reshape(-1).astype(np.float32)
        # reco_normals_gpu = cuda.mem_alloc(reco_normals.nbytes)
        # cuda.memcpy_htod(reco_normals_gpu, reco_normals)

        # copy stuff to gpu
        reco_positions_gpu = cuda.mem_alloc(reco_positions.nbytes)
        frequencies_gpu = cuda.mem_alloc(frequencies.nbytes)
        tx_positions_gpu = cuda.mem_alloc(self.tx_positions.nbytes)
        rx_positions_gpu = cuda.mem_alloc(self.rx_positions.nbytes)
        reco_image_gpu = cuda.mem_alloc(reco_image.nbytes)

        time_signal = np.ravel(time_signal)
        if antenna_weights is not None:
            time_signal = np.complex64(
                time_signal*np.ravel(np.repeat(antenna_weights, len(frequencies))).transpose(2, 0, 1))
            # time_signal = time_signal

        time_signal_gpu = cuda.mem_alloc(time_signal.nbytes)

        cuda.memcpy_htod(tx_positions_gpu, self.tx_positions / c)
        cuda.memcpy_htod(rx_positions_gpu, self.rx_positions / c)
        cuda.memcpy_htod(reco_positions_gpu, reco_positions / c)
        cuda.memcpy_htod(time_signal_gpu, time_signal)
        addr = mod.get_global("frequencies")
        # move frequencies to __const__ memory
        cuda.memcpy_htod(addr[0], frequencies * 2 * math.pi)

        func = mod.get_function(
            "holo_reco_sfcw_time_domain_custom_positions_hypotheses")

        block_size = (32, 8, 1)
        grid_size = (1,
                     int(math.ceil(reco_positions_size / block_size[1])), 1)

        func(
            tx_positions_gpu,
            self.num_tx_antennas,
            rx_positions_gpu,
            self.num_rx_antennas,
            reco_positions_gpu,
            reco_positions_size,
            reco_image_gpu,
            time_signal_gpu,
            time_signal_length,
            block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(reco_image, reco_image_gpu)

        reco_image = reco_image.reshape(
            reco_positions_size, self.rx_positions.shape[0] // 3, self.tx_positions.shape[0] // 3)

        tx_positions_gpu.free()
        rx_positions_gpu.free()
        reco_positions_gpu.free()
        reco_image_gpu.free()
        time_signal_gpu.free()
        frequencies_gpu.free()
        # reco_normals_gpu.free()

        return reco_image

    def reconstruct_custom_positions_signal(self,  reco_positions, meas_data_cal,
                                            frequencies, antenna_weights=None, reco_normals=None,
                                            use_normal_weights=False):
        """
            In contrast to the regular SFCW reconstruction, this method does not use 
            a voxel grid but instead takes 'reco_positions' (array of size Nx3) directly as
            input for generating hypotheses
        """
        assert self.reconstruction_initialized, "You need to call initialize_reconstruction() before this method"

        # print("Reconstruction for points: {} ({})".format(
        #     reco_positions, reco_positions.shape))

        mod = load_kernel_from_cu(os.path.join(
            os.path.dirname(__file__), "cuda", "holo_reco_sfcw_optimized.cu"))

        # time_signal = np.complex64(meas_data_cal.transpose(2, 0, 1))
        time_signal = np.complex64(meas_data_cal)
        time_signal_length = np.int32(len(frequencies))
        frequencies = np.float32(frequencies)
        reco_image = np.zeros(
            (reco_positions.shape[0], self.rx_positions.shape[0] // 3, self.tx_positions.shape[0] // 3), dtype=np.complex64).reshape(-1)

        reco_positions_size = np.int32((reco_positions.shape[0]))
        if reco_normals is not None:
            reco_positions = np.concatenate(
                [reco_positions, reco_normals], axis=0).reshape(-1).astype(np.float32)
        else:
            reco_positions = reco_positions.reshape(-1).astype(np.float32)

        c = 299792458.0
        # reco_normals = reco_normals.reshape(-1).astype(np.float32)
        # reco_normals_gpu = cuda.mem_alloc(reco_normals.nbytes)
        # cuda.memcpy_htod(reco_normals_gpu, reco_normals)

        # copy stuff to gpu
        reco_positions_gpu = cuda.mem_alloc(reco_positions.nbytes)
        frequencies_gpu = cuda.mem_alloc(frequencies.nbytes)
        tx_positions_gpu = cuda.mem_alloc(self.tx_positions.nbytes)
        rx_positions_gpu = cuda.mem_alloc(self.rx_positions.nbytes)
        reco_image_gpu = cuda.mem_alloc(reco_image.nbytes)

        time_signal = np.ravel(time_signal)
        if antenna_weights is not None:
            time_signal = np.complex64(
                time_signal*np.ravel(np.repeat(antenna_weights, len(frequencies))).transpose(2, 0, 1))
            # time_signal = time_signal

        time_signal_gpu = cuda.mem_alloc(time_signal.nbytes)

        cuda.memcpy_htod(tx_positions_gpu, self.tx_positions / c)
        cuda.memcpy_htod(rx_positions_gpu, self.rx_positions / c)
        cuda.memcpy_htod(reco_positions_gpu, reco_positions / c)
        cuda.memcpy_htod(time_signal_gpu, time_signal)
        addr = mod.get_global("frequencies")
        # move frequencies to __const__ memory
        cuda.memcpy_htod(addr[0], frequencies * 2 * math.pi)

        func = mod.get_function(
            "holo_reco_sfcw_time_domain_custom_positions_signal")

        block_size = (32, 8, 1)
        grid_size = (1,
                     int(math.ceil(reco_positions_size / block_size[1])), 1)

        func(
            tx_positions_gpu,
            self.num_tx_antennas,
            rx_positions_gpu,
            self.num_rx_antennas,
            reco_positions_gpu,
            reco_positions_size,
            reco_image_gpu,
            time_signal_gpu,
            time_signal_length,
            block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(reco_image, reco_image_gpu)

        reco_image = reco_image.reshape(
            reco_positions_size, self.rx_positions.shape[0] // 3, self.tx_positions.shape[0] // 3)

        tx_positions_gpu.free()
        rx_positions_gpu.free()
        reco_positions_gpu.free()
        reco_image_gpu.free()
        time_signal_gpu.free()
        frequencies_gpu.free()
        # reco_normals_gpu.free()

        return reco_image

    # overwrites method of 'RadarControllerBase'

    def extract_depth(self, reco_image, frequencies=None, amplitude_filter_threshold_dB=15,
                      depth_filter_kernel_size=15, visualize=False):
        return self.extract_depth_parabola_fit(reco_image,
                                               amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
                                               depth_filter_kernel_size=depth_filter_kernel_size, visualize=visualize)

    def extract_depth_maximum_projection(self, reco_image, amplitude_filter_threshold_dB=15,
                                         depth_filter_kernel_size=15, visualize=False):
        """
            Extracts a depth map from the reconstructed volume
            To estimate the depth, for each xy-coordinate 
            the maximum along the z-axis (z-coordinate with the highest amplitude) is determined.
        """

        z_coordinate = np.empty(
            (len(self.grid_y_range), len(self.grid_x_range)))
        rec_data_max_idx_list = np.argmax(np.abs(reco_image), axis=2)

        z_coordinate[:, :] = self.grid_z_range[rec_data_max_idx_list]

        rec_data_max, intensity = self.maximum_projection(
            reco_image, amplitude_filter_threshold_dB=amplitude_filter_threshold_dB, visualize=visualize)

        # Modification of Vanessa
        # set it to NaN first for filtering, later to 0
        nan_indices = np.argwhere(np.isnan(rec_data_max))
        nan_indices_row = nan_indices[:, 0]
        nan_indices_col = nan_indices[:, 1]
        z_coordinate[nan_indices_row, nan_indices_col] = np.nan

        # Averaging Filter
        box_2D_kernel = convolution.Box2DKernel(depth_filter_kernel_size)
        z_coordinate = convolution.convolve(
            z_coordinate, box_2D_kernel, boundary='extend',  preserve_nan=True)

        if visualize:
            assert self.visualization_initialized, "You need to call initialize_visualization() before this method"

            fig = plt.figure("Show estimated z coordinates")
            ax = plt.gca()
            colormap = plt.cm.get_cmap('jet_r')
            # im = plt.imshow(z_coordinate,vmin = self.z_lim_min, \
            # vmax=self.z_lim_max, extent=[self.grid_x_range.min(), self.grid_x_range.max(), self.grid_y_range.max(), self.grid_y_range.min()])
            im = plt.imshow(z_coordinate, vmin=np.nanmin(z_coordinate), vmax=np.nanmax(z_coordinate), extent=[self.grid_x_range.min(
            ), self.grid_x_range.max(), self.grid_y_range.min(), self.grid_y_range.max()],
                cmap=colormap)
            plt.xlabel('$x$ (m)')
            plt.ylabel('$y$ (m)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(im, cax=cax, label="$z$ (m)", cmap=colormap)
            # fig.set_size_inches(self.width_inch, self.height_inch, forward=True)
            # ax.set_xlim([self.x_lim_min,self.x_lim_max])
            # ax.set_ylim([self.y_lim_min,self.y_lim_max])
            plt.draw()

        # set to 0 (as usual in depth cameras if value is invalid)
        z_coordinate[nan_indices_row, nan_indices_col] = 0

        return z_coordinate

    def extract_depth_parabola_fit(self, reco_image, amplitude_filter_threshold_dB=15, depth_filter_kernel_size=15, visualize=False, parabola_neighbors=1):
        """
            Extracts a depth map from the reconstructed volume
            To estimate the depth, for each xy-coordinate 
            the maximum along the z-axis (z-coordinate with the highest amplitude) is first determined
            After that, each maximum coordinate is further refined using a parabola fit together with the direct neighbors of
            the maximum coordinate.
            Compared to basic maximum projection (no refinement), this depth map does not contain strong
            discretization artifacts along the z-axis which would be caused by the voxel grid
        """

        abs_reco = np.abs(reco_image)  # complex -> float

        abs_reco /= np.max(abs_reco)
        abs_reco = 20 * np.log10(abs_reco)

        rec_data_max_z_indices = np.argmax(
            abs_reco, axis=2)

        pixel_x, pixel_y = np.meshgrid(np.arange(
            0, self.grid_x_range.shape[0]), np.arange(0, self.grid_y_range.shape[0]))

        max_val = np.max(abs_reco)

        rec_data_max = abs_reco[pixel_y, pixel_x,
                                rec_data_max_z_indices]

        abs_z = self.grid_z_range[rec_data_max_z_indices.reshape(-1)]
        abs_y = self.grid_y_range[pixel_y.reshape(-1)]
        abs_x = self.grid_x_range[pixel_x.reshape(-1)]
        abs_xyz = np.stack([abs_x, abs_y, abs_z], axis=-1)

        data_triple_y = [rec_data_max]
        for i in range(1, parabola_neighbors+1):
            rec_data_max_z_indices_prev = np.clip(
                rec_data_max_z_indices - i, a_min=0, a_max=self.grid_z_range.shape[0]-1)
            rec_data_max_z_indices_next = np.clip(
                rec_data_max_z_indices + i, a_min=0, a_max=self.grid_z_range.shape[0]-1)
            rec_data_max_prev = abs_reco[pixel_y, pixel_x,
                                         rec_data_max_z_indices_prev]
            rec_data_max_next = abs_reco[pixel_y, pixel_x,
                                         rec_data_max_z_indices_next]
            data_triple_y.insert(0, rec_data_max_prev)
            data_triple_y.append(rec_data_max_next)

        data_triple_y = np.stack(
            data_triple_y, axis=-1).reshape(-1, parabola_neighbors*2+1)

        data_triple_x = np.stack(
            [np.linspace(-parabola_neighbors, parabola_neighbors, parabola_neighbors*2+1)] * data_triple_y.shape[0], axis=0) * (self.grid_z_range[1] - self.grid_z_range[0])

        data_triple_z = self.grid_z_range.reshape(
            -1)[rec_data_max_z_indices.reshape(-1)]

        depth = np.zeros((abs_reco.shape[0], abs_reco.shape[1]))
        for x, y, z_, y_, x_ in zip(data_triple_x, data_triple_y, data_triple_z, pixel_y.reshape(-1), pixel_x.reshape(-1)):
            z = np.polyfit(x, y, 2)
            a = z[0]
            b = z[1]
            c = z[2]
            max_z = (-b / (2 * a)) + z_
            depth[y_, x_] = max_z

        if amplitude_filter_threshold_dB > 0:
            rec_data_max[rec_data_max < -
                         amplitude_filter_threshold_dB] = np.nan

        # Modification of Vanessa
        # set it to NaN first for filtering, later to 0
        nan_indices = np.argwhere(np.isnan(rec_data_max))
        nan_indices_row = nan_indices[:, 0]
        nan_indices_col = nan_indices[:, 1]
        depth[nan_indices_row, nan_indices_col] = np.nan

        # Averaging Filter
        box_2D_kernel = convolution.Box2DKernel(depth_filter_kernel_size)
        depth = convolution.convolve(
            depth, box_2D_kernel, boundary='extend',  preserve_nan=True)

        if visualize:
            assert self.visualization_initialized, "You need to call initialize_visualization() before this method"

            fig = plt.figure("Show image after maximum projection (depth)")
            ax = plt.gca()
            im = plt.imshow(rec_data_max, extent=[self.grid_x_range.min(),
                                                  self.grid_x_range.max(), self.grid_y_range.min(), self.grid_y_range.max()])
            plt.xlabel('$x$ (m)')
            plt.ylabel('$y$ (m)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label="Norm. Magnitude (dB)")
            # fig.set_size_inches(
            #    self.width_inch, self.height_inch, forward=True)
            plt.draw()

            fig = plt.figure("Show estimated z coordinates")
            ax = plt.gca()
            colormap = plt.cm.get_cmap('jet_r')
            # im = plt.imshow(z_coordinate,vmin = self.z_lim_min, \
            # vmax=self.z_lim_max, extent=[self.grid_x_range.min(), self.grid_x_range.max(), self.grid_y_range.max(), self.grid_y_range.min()])
            im = plt.imshow(depth, vmin=np.nanmin(depth), vmax=np.nanmax(depth), extent=[self.grid_x_range.min(
            ), self.grid_x_range.max(), self.grid_y_range.min(), self.grid_y_range.max()],
                cmap=colormap)
            plt.xlabel('$x$ (m)')
            plt.ylabel('$y$ (m)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(im, cax=cax, label="$z$ (m)", cmap=colormap)
            # fig.set_size_inches(self.width_inch, self.height_inch, forward=True)
            # ax.set_xlim([self.x_lim_min,self.x_lim_max])
            # ax.set_ylim([self.y_lim_min,self.y_lim_max])
            plt.draw()

        depth[nan_indices_row, nan_indices_col] = 0

        return depth

    def extract_pointcloud_parabola_fit(self, reco_image, amplitude_filter_threshold_dB=15, depth_filter_kernel_size=15, visualize=False, parabola_neighbors=1):
        """
            Extracts a pointcloud from the reconstructed volume
            In comparison to 'extract_depth_parabola_fit' 
            t his method does not only extract the maximum depth but all signals that
            are greater than amplitude threshold
        """
        abs_reco = np.abs(reco_image)  # complex -> float

        amplitude = abs_reco.copy()

        abs_reco /= np.max(abs_reco)
        abs_reco = 20 * np.log10(abs_reco)  # Y, X then Z

        coordinates_x, coordinates_y, coordinates_z = np.meshgrid(
            self.grid_x_range, self.grid_y_range[::-1], self.grid_z_range)
        coordinates = np.stack(
            [coordinates_x, coordinates_y, coordinates_z], axis=-1).reshape(len(self.grid_y_range),
                                                                            len(self.grid_x_range),
                                                                            len(self.grid_z_range), 3)

        # rec_data_max_z_indices = np.argmax(
        #     abs_reco, axis=2)

        rec_data_z_indices = abs_reco >= -amplitude_filter_threshold_dB
        indices = np.where(rec_data_z_indices)
        indices = np.stack([indices[0], indices[1], indices[2]], axis=-1)
        coordinates = coordinates[indices[:, 0], indices[:, 1], indices[:, 2]]
        amplitude = amplitude[indices[:, 0], indices[:, 1], indices[:, 2]]

        data_triple_y = [abs_reco[rec_data_z_indices]]
        for i in range(1, parabola_neighbors+1):
            rec_data_max_z_indices_prev = np.clip(
                indices[:, 2] - i, a_min=0, a_max=self.grid_z_range.shape[0]-1)
            rec_data_max_z_indices_next = np.clip(
                indices[:, 2] + i, a_min=0, a_max=self.grid_z_range.shape[0]-1)
            rec_data_max_prev = abs_reco[indices[:, 0], indices[:, 1],
                                         rec_data_max_z_indices_prev]
            rec_data_max_next = abs_reco[indices[:, 0], indices[:, 1],
                                         rec_data_max_z_indices_next]
            data_triple_y.insert(0, rec_data_max_prev)
            data_triple_y.append(rec_data_max_next)

        data_triple_y = np.stack(
            data_triple_y, axis=-1).reshape(-1, parabola_neighbors*2+1)

        data_triple_x = np.stack(
            [np.linspace(-parabola_neighbors, parabola_neighbors, parabola_neighbors*2+1)] * data_triple_y.shape[0], axis=0) * (self.grid_z_range[1] - self.grid_z_range[0])

        data_triple_z = self.grid_z_range.reshape(
            -1)[indices[:, 2].reshape(-1)]

        pc = np.zeros((indices.shape[0], 3))
        counter = 0
        valid_indices = []
        for ind, (x, y, z_, y_, x_) in enumerate(zip(data_triple_x, data_triple_y, data_triple_z, coordinates[:, 1].reshape(-1), coordinates[:, 0].reshape(-1))):
            z = np.polyfit(x, y, 2)
            a = z[0]
            b = z[1]
            c = z[2]
            max_z = (-b / (2 * a)) + z_
            if max_z > self.grid_z_range[-1] or max_z < self.grid_z_range[0]:
                continue

            valid_indices.append(ind)
            pc[counter, 2] = max_z
            pc[counter, 0] = x_
            pc[counter, 1] = y_
            counter = counter + 1
        valid_indices = np.array(valid_indices)
        pc = pc[:counter]
        amplitude_valid = amplitude[valid_indices]

        if visualize:
            assert self.visualization_initialized, "You need to call initialize_visualization() before this method"

            pc_vis_noparabola = o3d.geometry.PointCloud()
            pc_vis_noparabola.points = o3d.utility.Vector3dVector(
                coordinates.reshape(-1, 3) * np.array([1, 1, -1]).reshape(1, 3))
            pc_vis_noparabola.paint_uniform_color([1, 0, 0])

            pc_vis = o3d.geometry.PointCloud()
            pc_vis.points = o3d.utility.Vector3dVector(
                pc.reshape(-1, 3) * np.array([1, 1, -1]).reshape(1, 3))
            col = ((amplitude_valid / amplitude_valid.max())).reshape(-1)
            pc_vis.colors = o3d.utility.Vector3dVector(
                np.stack([np.zeros_like(col), col, np.zeros_like(col)], axis=-1))

            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.add_geometry(pc_vis)
            visualizer.add_geometry(pc_vis_noparabola)
            opt = visualizer.get_render_option()
            opt.show_coordinate_frame = True
            visualizer.run()
            visualizer.destroy_window()

        return pc, amplitude_valid
