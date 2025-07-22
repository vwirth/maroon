

import maroon
import maroon.utils.visualization_utils as vu
import maroon.utils.data_utils as du

import open3d as o3d
import os
import numpy as np
import json
from maroon.utils.visualization_types import *

import pycuda.driver as cuda
cuda.init()
pycuda_ctx = cuda.Device(0).retain_primary_context()


# maximum error that is displayed in color map
max_error = 0.05  # 0.01
# which error type to use, either 'projective' or 'chamfer' error
error_type = ERROR_TYPES["projective"]
# whether to apply segmentation mask to the source point cloud during error calculation
use_masks = True
# whether to apply segmentation mask to the destination point cloud during error calculation
use_dest_mask = False
# depth threshold for image-based triangulation
triangulation_threshold = 0.01
# how many frames to average for calculating a depth map
averaging_factor = 1
# which frame index to reconstruct, for all sensors
frame_index = -1
# used to construct source mask, with this setting right now it is practically ignored
dot_thresh = -10.0
# which reconstruction method to use:
# - the standard backprojection algorithm is 'fscw'
# - for additional experiments in the application section we use the 'fsk' method
radar_reconstruction_method = "fsk"
# used in ablations to selectively choose frequencies in raw radar data
# must be a list of indices, e.g. [0, 1, 2] between 0 and 127
frequency_indices = [120, 127]
# used in ablations to selectively choose antennas in raw radar data
# must be a list of indices, e.g. [0, 1, 2] between 0 and 93
tx_antenna_indices = None
rx_antenna_indices = None
# (MM-)2FSK specific option: whether to choose a depth prior or not
use_prior = False
# MM-2FSK specific option: which optical sensor to use as a depth prior
# choose from "kinect", "realsense", "zed" or "photogrammetry"
prior_depth_sensor = "photogrammetry"


def visualize_alignment(config,
                        error_type=ERROR_TYPES["projective"],
                        max_error=0.01,
                        offline=False):

    use_relative = config["use_relative_paths"]

    object_name = config["reconstruction_path"]
    if use_relative:
        base_path = config["base_path"]
    else:
        base_path = ""

    metadata_file = os.path.join(
        base_path, config["reconstruction_path"], "metadata.json")
    calib_path = ""

    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    if os.path.exists(os.path.join(
            base_path, config["reconstruction_path"], config["calibration_filename"])):
        calib_path = (os.path.join(
            base_path, config["reconstruction_path"]))
    else:
        assert "calibration" in metadata and len(metadata["calibration"]) > 0
        assert use_relative
        calib_path_tmp = os.path.join(
            base_path, "00_registration", metadata["calibration"])
        if os.path.exists(calib_path_tmp):
            calib_path = calib_path_tmp

    print("using calib path: ", calib_path)

    calib = {}
    with open(os.path.join(calib_path, config["calibration_filename"]), "r") as f:
        calib = json.load(f)

    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    o3d.visualization.gui.Application.instance.initialize()

    vis = vu.SensorVisualizer(2000, 2000)
    create_isometric = True
    kinect_space = "depth"

    if "mask_erosion" in metadata and config["mask_erosion"] > 0:
        erosion = metadata["mask_erosion"]
    else:
        erosion = config["mask_erosion"]

    if "distance_meters" in metadata and metadata["distance_meters"] == 0.3 and (not "mask_bb" in metadata or not "zmin" in metadata["mask_bb"]):
        if not "mask_bb" in metadata:
            metadata["mask_bb"] = {"zmin": 0.26}
        else:
            metadata["mask_bb"]["zmin"] = 0.26

        if os.path.exists(metadata_file):
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)
    if (not "mask_bb" in metadata or not "zmax" in metadata["mask_bb"]):
        if not "mask_bb" in metadata:
            metadata["mask_bb"] = {"zmax": 0.7}
        else:
            metadata["mask_bb"]["zmax"] = 0.7

        if os.path.exists(metadata_file):
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

    mask_bbmin = [None, None, None]
    mask_bbmax = [None, None, None]
    if "mask_3d_threshold" in metadata:
        mask_bbmax[2] = metadata["mask_3d_threshold"]
        print("loading mask threshold from metadata; ", mask_bbmax[2])

        del metadata["mask_3d_threshold"]
        if not "mask_bb" in metadata:
            metadata["mask_bb"] = {"zmax": mask_bbmax[2]}
        else:
            metadata["mask_bb"]["zmax"] = mask_bbmax[2]

        if os.path.exists(metadata_file):
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)
    if "mask_bb" in metadata:
        for idx, key in enumerate(["x", "y", "z"]):
            if key+"min" in metadata["mask_bb"]:
                mask_bbmin[idx] = metadata["mask_bb"][key+"min"]
            if key+"max" in metadata["mask_bb"]:
                mask_bbmax[idx] = metadata["mask_bb"][key+"max"]

    radar_erosion = int(round(erosion / 4, 0))
    realsense_erosion = int(round(erosion // 0.7))
    zed_erosion = int(round(erosion // 0.58))
    use_gt_mask = False

    sensor_data = {
        "radar": {"index": frame_index,
                  "use_mask": use_masks,
                  "isometric": create_isometric,
                  "amplitude_filter_threshold_dB": config["radar"]["amplitude_filter_threshold_dB"],
                  "mask_erosion": radar_erosion,
                  "use_intrinsic_parameters": config["radar"]["use_intrinsic_parameters"],
                  "use_gt_mask": use_gt_mask,
                  "reconstruction_method": radar_reconstruction_method,
                  "frequency_indices": frequency_indices,
                  "tx_antenna_indices": tx_antenna_indices,
                  "rx_antenna_indices": rx_antenna_indices,
                  "use_prior": use_prior,
                  "prior_depth_sensor": prior_depth_sensor},
        "photogrammetry": {"index": frame_index, "use_mask": True, "isometric": create_isometric, "mask_erosion": 0},
        "kinect": {"index": frame_index, "use_mask": use_masks, "isometric": create_isometric, "data_space": kinect_space, "mask_erosion": erosion,
                   "use_gt_mask": use_gt_mask},
        "realsense": {"index": frame_index, "use_mask": use_masks, "isometric": create_isometric, "mask_erosion": realsense_erosion,
                      "use_gt_mask": use_gt_mask},
        "zed": {"index": frame_index, "use_mask": use_masks, "isometric": create_isometric, "mask_erosion": zed_erosion,
                "use_gt_mask": use_gt_mask},
    }

    vis.initialize_sensors(sensor_data, config["sensors_in_use"], config, calib, base_path=os.path.join(
        base_path, config["reconstruction_path"]),  averaging=averaging_factor,
        triangulation_threshold=triangulation_threshold, error_type=error_type, use_dest_mask=use_dest_mask,
        dot_thresh=dot_thresh, max_error=max_error, mask_bbmin=mask_bbmin, mask_bbmax=mask_bbmax)

    # Run the event loop. This will not return until the last window is closed.
    vis._on_show_sensor("radar", True)
    vis._on_show_sensor("kinect", True)
    vis._on_show_sensor("realsense", True)
    vis._on_show_sensor("zed", True)
    vis._on_show_sensor("photogrammetry", True)
    o3d.visualization.gui.Application.instance.run()


def interactive(config, max_error, error_type):

    visualize_alignment(config,
                        error_type=error_type,
                        max_error=max_error)


def main():

    filename = "script_config"

    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config_path = os.path.join(project_path, "configs", filename+".json")
    assert os.path.exists(
        config_path), "Config path must be set correctly to the json file in the configs directory"
    with open(config_path) as f:
        config = json.load(f)

    interactive(config,
                max_error, error_type)


if __name__ == '__main__':
    pycuda_ctx.push()
    main()
    pycuda_ctx.pop()
