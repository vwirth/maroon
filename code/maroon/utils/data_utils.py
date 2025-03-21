
import maroon.utils.pointcloud_utils as pu

import matplotlib
import plotly.express as px
import plotly.graph_objects as go
import open3d as o3d
import os
import json
import cv2
import numpy as np


def get_data(sensor_type, data_path, frame_index, config, use_mask=False,
             averaging_factor=1,
             triangulation_threshold=pow(2, 16),
             amplitude_filter_threshold_dB=-1, kinect_space="color",
             use_intrinsic_parameters=False,
             manual_mask=None):
    points = None
    depth = None
    rgb = None
    normal = None
    mask = None
    proj = np.eye(4)
    aux = []

    if sensor_type == "radar":
        from maroon.sensors.radar_data_loader import RadarDataLoader

        use_empty_space_measurements = config["use_empty_space_measurements"]
        radar_loader = RadarDataLoader(data_path, **config["radar"]["reconstruction_capture_params"],
                                       **config["radar"]["reconstruction_reco_params"],
                                       use_empty_space_measurements=use_empty_space_measurements,
                                       averaging=averaging_factor)
        radar_loader.read_radar_frames()

        radar_points, radar_depth, radar_intensity = radar_loader.get_frame(frame_index, force_redo=config["radar"]["force_redo"],
                                                                            amplitude_filter_threshold_dB=amplitude_filter_threshold_dB,
                                                                            depth_filter_kernel_size=1,
                                                                            use_intrinsic_parameters=use_intrinsic_parameters,
                                                                            save_volume=True,
                                                                            save_pc=True,
                                                                            averaging_factor=1)

        radar_normal = radar_loader.compute_normal_from_pc(
            radar_points, radar_depth)

        if use_mask and manual_mask is not None:
            radar_points = radar_points * \
                manual_mask.astype(np.float32)[:, :, None]
            radar_depth = radar_depth * \
                manual_mask.astype(np.float32)
            radar_intensity = radar_intensity * \
                manual_mask.astype(np.float32)
            radar_normal = radar_normal * \
                manual_mask.astype(np.float32)[:, :, None]

        mesh = pu.triangulate_image_pointcloud(
            radar_points, threshold=triangulation_threshold)

        radar_intensity = radar_intensity.reshape(-1)
        radar_intensity = radar_intensity / radar_intensity.max()
        radar_intensity = np.stack([np.zeros_like(
            radar_intensity), radar_intensity, np.zeros_like(radar_intensity)], axis=-1)

        points = radar_points.reshape(-1, 3)
        rgb = radar_intensity
        proj = radar_loader.get_intrinsics(frame_index)
        depth = radar_depth
        normal = radar_normal.reshape(-1, 3)
        if use_mask:
            mask = (radar_depth != 0).astype(np.uint8)
            if manual_mask is not None:
                mask = manual_mask

    elif sensor_type == "kinect":
        from maroon.sensors.kinect_data_loader import KinectDataLoader

        kinect_loader = KinectDataLoader(data_path, space=kinect_space)

        kinect_rgb, kinect_depth, kinect_depth_t, kinect_points, kinect_normals = kinect_loader.get_frame(
            frame_index, use_mask=use_mask,  manual_mask=manual_mask, mask_idx=-1,  undistort=True, averaging_factor=averaging_factor)

        kinect_ir = kinect_loader.get_ir_image(
            frame_index=frame_index, use_mask=use_mask, manual_mask=manual_mask, undistort=True)
        if kinect_ir is None:
            kinect_ir = np.zeros_like(kinect_rgb[:, :, 0])

        mesh = pu.triangulate_image_pointcloud(
            kinect_points, threshold=triangulation_threshold*1000.0)

        kinect_points = kinect_points.reshape(-1, 3)
        kinect_rgb = (kinect_rgb[:, :, ::-1].reshape(-1, 3) / 255.0) * 0.75

        points = kinect_points
        rgb = kinect_rgb

        proj = kinect_loader.get_intrinsics()
        proj[3, 3] = 0
        proj[3, 2] = 1
        depth = kinect_depth_t
        normal = kinect_normals.reshape(-1, 3)
        zeros = np.zeros_like(kinect_ir.reshape(-1))
        aux.append(
            np.stack([kinect_ir.reshape(-1) / kinect_ir.max(), kinect_ir.reshape(-1) / kinect_ir.max(), kinect_ir.reshape(-1) / kinect_ir.max()], axis=-1))

        if use_mask:
            mask = kinect_loader.get_mask(-1)
            if manual_mask is not None:
                mask = manual_mask

    elif sensor_type == "photogrammetry":
        from maroon.sensors.camera_data_loader import CameraDataLoader

        photo_loader = CameraDataLoader(data_path)
        mesh = photo_loader.get_mesh(
            do_smoothing=True, use_mask=use_mask)
        photo_loader.load_images()
        photo_loader.load_calibration()
        _, depth, _ = photo_loader.get_image_data_from_idx(0)

        points = np.array(mesh.vertices)
        rgb = np.array(mesh.vertex_colors)
        normal = np.array(mesh.vertex_normals) * \
            np.array([1, 1, 1]).reshape(-1, 3)
        mesh = (np.array(mesh.vertices), np.array(mesh.triangles))

        if use_mask:
            mask = np.ones_like(depth).astype(np.uint8)

    elif sensor_type == "realsense":
        from maroon.sensors.realsense_data_loader import RealsenseDataLoader

        rs_frame_index = frame_index

        realsense_loader = RealsenseDataLoader(data_path)
        realsense_rgb, realsense_depth, realsense_points, realsense_normal = realsense_loader.get_frame(
            rs_frame_index, use_mask=use_mask, mask_idx=-1, averaging_factor=averaging_factor, manual_mask=manual_mask)
        mesh = pu.triangulate_image_pointcloud(
            realsense_points, threshold=triangulation_threshold*1000.0)

        realsense_points = realsense_points.reshape(-1, 3)
        realsense_rgb = realsense_rgb[:, :, ::-1].reshape(-1, 3) / 255.0

        points = realsense_points
        rgb = realsense_rgb
        proj = realsense_loader.get_intrinsics()
        proj[3, 3] = 0
        proj[3, 2] = 1
        depth = realsense_depth
        normal = realsense_normal.reshape(-1, 3)

        if use_mask:
            mask = realsense_loader.get_mask(-1)
            if manual_mask is not None:
                mask = manual_mask

    elif sensor_type == "zed":
        from maroon.sensors.zed_data_loader import ZEDDataLoader
        zed_frame_index = frame_index

        zed_loader = ZEDDataLoader(data_path)
        (zed_rgb, zed_depth, zed_points, zed_normal), _ = zed_loader.get_frame(
            zed_frame_index, use_mask=use_mask, mask_idx=-1, averaging_factor=averaging_factor, manual_mask=manual_mask)
        mesh = pu.triangulate_image_pointcloud(
            zed_points, threshold=triangulation_threshold*1000.0)

        zed_points = zed_points.reshape(-1, 3)
        zed_rgb = zed_rgb[:, :, ::-1].reshape(-1, 3) / 255.0

        points = zed_points
        rgb = zed_rgb
        proj = zed_loader.get_intrinsics()
        proj[3, 3] = 0
        proj[3, 2] = 1
        depth = zed_depth
        normal = zed_normal.reshape(-1, 3)

        if use_mask:
            mask = zed_loader.get_mask(-1)
            if manual_mask is not None:
                mask = manual_mask

    return points, rgb, depth, mask, normal, proj, mesh, aux


def transform_sensor_to_world_coordsys(calib, sensor_type, points, normals=None):
    T = np.array(calib["{}2world".format(sensor_type)]).reshape(4, 4)

    ones = np.ones((points.shape[0], 1))
    transformed_points = np.matmul(np.concatenate(
        [points, ones], axis=1), T.transpose())[:, :3]

    N = np.transpose(np.linalg.inv(T))
    ones = np.ones((normals.shape[0], 1))
    transformed_normals = np.matmul(np.concatenate(
        [normals, ones], axis=1), N.transpose())[:, :3]
    length = np.linalg.norm(transformed_normals, axis=-1)
    transformed_normals[length >
                        0] = transformed_normals[length > 0] / length[length > 0].reshape(-1, 1)

    return transformed_points, transformed_normals


def print_coordinates(str_tag, pc):
    assert len(pc.shape) == 1 or pc.shape[1] == 3
    if pc.shape[0] == 0:
        return

    if len(pc.shape) == 1:
        nonzero = pc != 0
        if pc[nonzero].shape[0] > 0:
            print(f"{str_tag} Val: {pc[nonzero].min()} - {pc[nonzero].max()}")
        else:
            print(f"{str_tag} Val: {pc.min()} - {pc.max()}")
    else:
        nonzero = pc[:, 2] != 0
        print(f"{str_tag} X: {pc[:,0].min()} - {pc[:,0].max()}")
        print(f"{str_tag} Y: {pc[:,1].min()} - {pc[:,1].max()}")
        if pc[nonzero].shape[0] > 0:
            print(f"{str_tag} Z: {pc[nonzero][:,2].min()} - {pc[:,2].max()}")
        else:
            print(f"{str_tag} Z: {pc[:,2].min()} - {pc[:,2].max()}")

    print("---------------------------")


def clip_bbs(calib, points, normals, common_space, aux=[], indices=None,
             xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=0, zmax=5):

    points_open3d, normals_open3d = transform_sensor_to_world_coordsys(calib,
                                                                       common_space, points, normals)

    mask_spatial_z = np.logical_and(
        points_open3d[:, 2] > zmin, points_open3d[:, 2] < zmax)
    mask_spatial_x = np.logical_and(
        points_open3d[:, 0] > xmin, points_open3d[:, 0] < xmax)
    mask_spatial_y = np.logical_and(
        points_open3d[:, 1] > ymin, points_open3d[:, 1] < ymax)
    mask = np.logical_and(mask_spatial_x, np.logical_and(
        mask_spatial_y, mask_spatial_z))

    # mask = np.ones((points_open3d.shape[0],)) > 0
    # mask[0:10] = True
    nonzero_indices = np.nonzero(mask)[0]
    if (indices is not None):
        index_mask = np.all(
            np.in1d(indices, nonzero_indices).reshape(indices.shape), axis=-1)
        indices_open3d = indices[index_mask]

        points_ = points.copy()
        normals_ = normals.copy()
    else:
        indices_open3d = None

        points_open3d = points_open3d[mask]
        normals_open3d = normals_open3d[mask]

        points_ = points[mask]
        normals_ = normals[mask]

    aux_ = []
    for a in aux:
        if a is not None:
            if indices is None:
                aux_.append(a[mask])
            else:
                aux_.append(a)
        else:
            aux_.append(None)

    assert points_open3d.shape[0] > 0
    assert normals_open3d.shape[0] == points_open3d.shape[0]

    return points_, normals_, points_open3d, normals_open3d, aux_, indices_open3d


def create_pixel_aligned_mesh(points, indices,
                              proj, height, width,
                              attributes=[], triangulation_threshold=pow(2, 16)):

    attributes_out = []
    points_out = None

    attr_index = 0
    for attr in attributes:
        if attr is None:
            attributes_out.append(None)
            continue

        if attr_index == 0:
            _, _, points_out, attribute_out = pu.project_interpolate(
                points, indices, proj, height, width, src_color_attribute=attr)
        else:
            _, _, _, attribute_out = pu.project_interpolate(
                points, indices, proj, height, width, src_color_attribute=attr)
        attributes_out.append(attribute_out)
        attr_index = attr_index + 1

    _, indices_out = pu.triangulate_image_pointcloud(
        points_out, threshold=triangulation_threshold)

    return points_out, indices_out, attributes_out
