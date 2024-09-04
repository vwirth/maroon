

import maroon.utils.spatial_alignment as salign
from maroon.sensors.kinect_data_loader import KinectDataLoader
from maroon.utils.visualization_types import *
import maroon.utils.metrics as metrics
import maroon.utils.data_utils as du
import maroon.utils.pointcloud_utils as pu

from functools import partial
from open3d_app import *
import matplotlib
import plotly.express as px
import open3d as o3d
import os
import json
import cv2
import numpy as np


class SensorVisualizerSettings:
    DEFAULT_SENSOR = "Radar"
    SENSORS = [
        "radar",
        "photogrammetry",
        "kinect",
        "realsense",
        "zed"
    ]
    DEFAULT_VIS_MODE = "Color"
    VIS_MODES = [
        "Color",
        "Normal",
        "Error",
        "Dependence",
        "Infrared"
    ]


class SensorVisualizer(AppWindow):
    def __init__(self, width, height):
        super(SensorVisualizer, self).__init__(width, height)

        self._sensors = gui.CollapsableVert("Sensors", 0,
                                            gui.Margins(self.em, 0, 0, 0))

        self._sensors_enabled = gui.Vert(0, gui.Margins(self.em, 0, 0, 0))
        self.sensors_enabled = {}
        self.sensor_masks = {}
        self.sensor_isometric = {}

        for name in (SensorVisualizerSettings.SENSORS):
            checks = o3d.visualization.gui.Checkbox(name)
            checks.set_on_checked(
                functools.partial(self._on_show_sensor, name))
            checks2 = o3d.visualization.gui.Checkbox("Mask")
            checks2.set_on_checked(
                functools.partial(self._on_mask_change, name))
            checks3 = o3d.visualization.gui.Checkbox("Iso")
            checks3.set_on_checked(
                functools.partial(self._on_isometric_change, name))

            horiz = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
            horiz.add_child(checks)

            # if name != "radar":
            horiz.add_child(checks2)
            horiz.add_child(checks3)

            self._sensors_enabled.add_child(horiz)
            # self._sensors_enabled.add_child(checks)
            # self._sensors_enabled.add_child(checks2)

            # self._sensors_enabled.add_fixed(self.separation_height)
            self.sensors_enabled[name] = checks
            self.sensor_masks[name] = checks2
            self.sensor_isometric[name] = checks3

        self._sensors.add_child(self._sensors_enabled)

        self._sensor_space = gui.Combobox()
        for name in (SensorVisualizerSettings.SENSORS):
            self._sensor_space.add_item(name)
        self._sensor_space.set_on_selection_changed(
            self._on_sensor_space_change)

        self._sensors.add_fixed(self.separation_height)
        self._sensors.add_child(gui.Label("Sensor Space"))
        self._sensors.add_child(self._sensor_space)

        # self._sensors.add_fixed(self.separation_height)
        # checks = o3d.visualization.gui.Checkbox("Show Mask")
        # checks.set_on_checked(self._on_mask_change)
        # self._sensors.add_child(checks)

        self._vis_type = gui.Combobox()
        for name in (SensorVisualizerSettings.VIS_MODES):
            self._vis_type.add_item(name)
        self._vis_type.set_on_selection_changed(self._on_vis_type_change)

        self._sensors.add_fixed(self.separation_height)
        self._sensors.add_child(gui.Label("Visualization Type"))
        self._sensors.add_child(self._vis_type)

        self._error_destination = gui.Combobox()
        for name in (SensorVisualizerSettings.SENSORS):
            self._error_destination.add_item(name)
        self._error_destination.set_on_selection_changed(
            self._on_error_destination_changed)

        self._sensors.add_fixed(self.separation_height)
        self._sensors.add_child(gui.Label("Error Destination"))
        self._sensors.add_child(self._error_destination)

        self.error_space_is_sensor = False
        checks = o3d.visualization.gui.Checkbox("Error Space = Sensor")
        checks.set_on_checked(
            functools.partial(self._on_error_space_is_sensor, name))
        self._sensors.add_fixed(self.separation_height)
        self._sensors.add_child(checks)

        self.bb_step_size = 0.1

        def create_bb_button(name, func, val):
            horiz = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
            label = gui.Label(name)
            button_neg = gui.Button("-")
            button_neg.set_on_clicked(
                functools.partial(func, -1 * self.bb_step_size))
            button_pos = gui.Button("+")
            button_pos.set_on_clicked(
                functools.partial(func, 1 * self.bb_step_size))
            text = gui.TextEdit()
            text.text_value = str(val)
            text.set_on_value_changed(functools.partial(func, 0))

            horiz.add_child(label)
            horiz.add_child(text)
            horiz.add_child(button_pos)
            horiz.add_child(button_neg)

            return horiz, text

        self.bbx = [-0.4, 0.4]
        self.bbx_text = [None, None]
        self.bby = [-0.4, 0.4]
        self.bby_text = [None, None]
        self.bbz = [0.2, 0.9]
        self.bbz_text = [None, None]

        self._sensors.add_fixed(self.separation_height)

        horiz, text = create_bb_button(
            "X (min)", self._on_minbbx_change, self.bbx[0])
        self._sensors.add_child(horiz)
        self.bbx_text[0] = text
        horiz, text = create_bb_button(
            "X (max)", self._on_maxbbx_change, self.bbx[1])
        self._sensors.add_child(horiz)
        self.bbx_text[1] = text

        horiz, text = create_bb_button(
            "Y (min)", self._on_minbby_change, self.bby[0])
        self._sensors.add_child(horiz)
        self.bby_text[0] = text
        horiz, text = create_bb_button(
            "Y (max)", self._on_maxbby_change, self.bby[1])
        self._sensors.add_child(horiz)
        self.bby_text[1] = text

        horiz, text = create_bb_button(
            "Z (min)", self._on_minbbz_change, self.bbz[0])
        self._sensors.add_child(horiz)
        self.bbz_text[0] = text
        horiz, text = create_bb_button(
            "Z (max)", self._on_maxbbz_change, self.bbz[1])
        self._sensors.add_child(horiz)
        self.bbz_text[1] = text

        self._settings_panel.add_fixed(self.separation_height)
        self._settings_panel.add_child(self._sensors)

        self.current_space = "radar"
        self.error_destination = "radar"
        self.vis_type = "Color"

    def initialize_sensors(self, sensor_data, config, calib, base_path="",
                           radar_reconstruction_method="fscw", averaging=1,
                           triangulation_threshold=0.01,
                           error_type=ERROR_TYPES["chamfer"], use_dest_mask=False,
                           dot_thresh=0.8, max_error=0.01, mask_bbmin=[None, None, None], mask_bbmax=[None, None, None]):
        self.sensor_data = sensor_data
        self.config = config
        self.calib = calib
        self.radar_reconstruction_method = radar_reconstruction_method
        self.averaging = averaging
        self.triangulation_threshold = triangulation_threshold
        self.error_type = error_type
        self.use_dest_mask = use_dest_mask
        self.dot_thresh = dot_thresh
        self.max_error = max_error
        self.mask_bbmin = mask_bbmin
        self.mask_bbmax = mask_bbmax
        self.has_3d_threshold = (self.mask_bbmin[0] is not None
                                 or self.mask_bbmin[1] is not None
                                 or self.mask_bbmin[2] is not None
                                 or self.mask_bbmax[0] is not None
                                 or self.mask_bbmax[1] is not None
                                 or self.mask_bbmax[2] is not None)
        self.sensors = {}
        self._triangles.checked = False
        # self._scene.scene.set_background([0, 0, 0, 1])
        self.settings.bg_color = gui.Color(0, 0, 0)
        self._apply_settings()

        if base_path is not None and len(base_path) > 0:
            for k in sensor_data.keys():
                if k == "radar":
                    detailed_filename = "radar_{}_{}_{}".format(float(config["radar"]["reconstruction_capture_params"]["frequency_low"]),
                                                                float(
                                                                    config["radar"]["reconstruction_capture_params"]["frequency_high"]),
                                                                int(config["radar"]["reconstruction_capture_params"]["frequency_points"]))

                    sensor_data[k]["path"] = os.path.join(
                        base_path, detailed_filename)
                else:
                    sensor_data[k]["path"] = os.path.join(base_path, k)

        if self.sensor_data["kinect"]["data_space"] == "depth" and (not "space" in self.calib["kinect_calib"] or self.calib["kinect_calib"]["space"] == "color"):
            kinect_loader = KinectDataLoader(
                self.sensor_data["kinect"]["path"], space=self.sensor_data["kinect"]["data_space"])
            K = np.array(kinect_loader.get_extrinsics("color", "depth"))
            K_inv = np.array(kinect_loader.get_extrinsics("depth", "color"))
            for key in self.calib.keys():
                if "2kinect" in key:
                    M = np.array(self.calib[key]).reshape(4, 4)
                    M = np.matmul(K, M)
                    self.calib[key] = M.tolist()
                elif "kinect2" in key:
                    M = np.array(self.calib[key]).reshape(4, 4)
                    M = np.matmul(M, K_inv)
                    self.calib[key] = M.tolist()
            print("******** Applying calibration correction: color -> depth")

        if self.sensor_data["kinect"]["data_space"] == "color" and ("space" in self.calib["kinect_calib"] and self.calib["kinect_calib"]["space"] == "depth"):
            kinect_loader = KinectDataLoader(
                self.sensor_data["kinect"]["path"], space=self.sensor_data["kinect"]["data_space"])
            K_inv = np.array(kinect_loader.get_extrinsics("color", "depth"))
            K = np.array(kinect_loader.get_extrinsics("depth", "color"))
            for key in self.calib.keys():
                if "2kinect" in key:
                    M = np.array(self.calib[key]).reshape(4, 4)
                    M = np.matmul(K, M)
                    self.calib[key] = M.tolist()
                elif "kinect2" in key:
                    M = np.array(self.calib[key]).reshape(4, 4)
                    M = np.matmul(M, K_inv)
                    self.calib[key] = M.tolist()
            print("******** Applying calibration correction: depth -> color")

    def execute_script(self, script, *args):
        script(self, *args)

    def take_screenshot(self, filename="screenshot.png"):
        if filename == "screenshot.png":
            counter = 0
            while os.path.exists(filename):
                counter = counter + 1
                tag = str(counter).zfill(3)
                filename = "screenshot_" + tag + ".png"
        print(f"Save Screenshot to: {filename}")
        if (os.path.exists(filename)):
            os.remove(filename)
        self.export_image(filename)
        while not os.path.exists(filename):
            self.window.post_redraw()
            o3d.visualization.gui.Application.instance.run_one_tick()

    def take_screenshot_object_side(self, filename="screenshot.png"):
        center = self.get_pointcloud_center()
        self._rotate_around_scene_xyz(
            [0, np.pi * 0.25, 0], center=center, update_camera=False)
        o3d.visualization.gui.Application.instance.run_one_tick()
        self.take_screenshot(filename)

        self._rotate_around_scene_xyz(
            [0, -np.pi * 0.25, 0], center=center, update_camera=False)
        o3d.visualization.gui.Application.instance.run_one_tick()

    def _on_show_sensor(self, sensor_type, visible, clip_min=None, clip_max=None):
        self.show_sensor(sensor_type, visible,
                         clip_min=clip_min, clip_max=clip_max)
        o3d.visualization.gui.Application.instance.run_one_tick()
        self.window.post_redraw()

    def change_mask(self, sensor_type, use_mask):
        if use_mask != self.sensor_data[sensor_type]["use_mask"] and sensor_type in self.sensors[sensor_type]:
            self.sensor_data[sensor_type]["use_mask"] = use_mask
            del self.sensors[sensor_type]
            self.show_sensor(
                sensor_type, self.sensor_data[sensor_type]["visible"])

    def change_isometric(self, sensor_type, isometric):
        if isometric != self.sensor_data[sensor_type]["isometric"] and sensor_type in self.sensors[sensor_type]:
            self.sensor_data[sensor_type]["isometric"] = isometric
            del self.sensors[sensor_type]
            self.show_sensor(
                sensor_type, self.sensor_data[sensor_type]["visible"])

    def get_pointcloud_aabb_metric(self, sensor_type, current_space):
        src_indices = self.sensors[sensor_type][current_space]["indices"]
        src_points = self.sensors[sensor_type][current_space]["points"][np.unique(
            src_indices)]

        bb_min_x = src_points[:, 0].min()
        bb_max_x = src_points[:, 0].max()
        bb_min_y = src_points[:, 1].min()
        bb_max_y = src_points[:, 1].max()
        bb_min_z = src_points[:, 2].min()
        bb_max_z = src_points[:, 2].max()

        # clamp to antenna geometry
        bb_min_x = max(bb_min_x, -0.07)
        bb_max_x = min(bb_max_x, 0.07)
        bb_min_y = max(bb_min_y, -0.07)
        bb_max_y = min(bb_max_y, 0.07)

        surface_area = (bb_max_x - bb_min_x) * (bb_max_y - bb_min_y)

        return surface_area / (0.14*0.14)

    def get_pointcloud_surface_metric(self, sensor_type, current_space, threshold):
        src_indices = self.sensors[sensor_type][current_space]["indices"]
        src_normals = self.sensors[sensor_type][current_space]["normals"][np.unique(
            src_indices)]

        dot_prod = -src_normals[:, 2]
        angle_radians = np.arccos(
            dot_prod, np.linalg.norm(src_normals, axis=-1))
        angle_degree = np.degrees(angle_radians)

        # np.sum(angle_degree > threshold) / angle_degree.shape[0]
        return np.median(angle_degree)

    def export(self, sensor_type, path):
        self.export_geometry(sensor_type, path)

    def show_sensor(self, sensor_type, visible, changed_clipping=False, clip_min=None, clip_max=None):
        self.sensors_enabled[sensor_type].checked = visible
        self.sensor_masks[sensor_type].checked = self.sensor_data[sensor_type]["use_mask"]
        self.sensor_isometric[sensor_type].checked = self.sensor_data[sensor_type]["isometric"]

        if visible:
            current_space = self.current_space
            vis_type = self.vis_type
            error_destination = self.error_destination
            error_space = current_space
            if self.error_type == ERROR_TYPES["projective"]:
                error_space = error_destination
            if self.error_space_is_sensor:
                error_space = sensor_type

            if current_space == "photogrammetry" or (error_space == "photogrammetry" and not self.error_space_is_sensor):
                return

            self.add_sensor(sensor_type)
            self.transform_sensor(
                sensor_type, current_space, isometric=self.sensor_data[sensor_type]["isometric"])

            src_points = self.sensors[sensor_type][current_space]["points"]
            src_rgb = self.sensors[sensor_type][current_space]["rgb"]
            src_depth = self.sensors[sensor_type][current_space]["depth"]
            src_normals = self.sensors[sensor_type][current_space]["normals"]
            src_infrared = self.sensors[sensor_type][current_space]["aux"]
            src_indices = self.sensors[sensor_type][current_space]["indices"]
            src_dependence = self.sensors[sensor_type][current_space]["dependence"]

            if (not "space" in self.sensor_data[sensor_type] or
                not "vis" in self.sensor_data[sensor_type] or
                not "visible" in self.sensor_data[sensor_type] or
                self.sensor_data[sensor_type]["space"] != current_space or
                self.sensor_data[sensor_type]["vis"] != vis_type or
                ("error_destination" in self.sensor_data[sensor_type]
                 and self.sensor_data[sensor_type]["error_destination"] != error_destination) or
                ("error_space" in self.sensor_data[sensor_type]
                 and self.sensor_data[sensor_type]["error_space"] != error_space) or
                    changed_clipping or not self.has_geometry(sensor_type)):

                if not self.sensor_data[sensor_type]["use_mask"]:

                    src_points, src_normals, src_points_open3d, src_normals_open3d, src_aux, src_indices = du.clip_bbs(self.calib,
                                                                                                                       src_points, src_normals, current_space,
                                                                                                                       aux=[
                                                                                                                           src_rgb, src_infrared, src_dependence],
                                                                                                                       indices=src_indices,
                                                                                                                       xmin=self.bbx[
                                                                                                                           0], xmax=self.bbx[1],
                                                                                                                       ymin=self.bby[
                                                                                                                           0], ymax=self.bby[1],
                                                                                                                       zmin=self.bbz[0], zmax=self.bbz[1])

                    src_rgb = src_aux[0]
                    src_infrared = src_aux[1]
                    src_dependence = src_aux[2]

                if clip_min is not None and clip_max is not None:
                    src_points, src_normals, src_points_open3d, src_normals_open3d, src_aux, src_indices = du.clip_bbs(self.calib,
                                                                                                                       src_points, src_normals, current_space,
                                                                                                                       aux=[
                                                                                                                           src_rgb, src_infrared, src_dependence],
                                                                                                                       indices=src_indices,
                                                                                                                       xmin=clip_min[
                                                                                                                           0], xmax=clip_max[0],
                                                                                                                       ymin=clip_min[
                                                                                                                           1], ymax=clip_max[1],
                                                                                                                       zmin=clip_min[2], zmax=clip_max[2])

                    src_rgb = src_aux[0]
                    src_infrared = src_aux[1]
                    src_dependence = src_aux[2]

                src_points, src_normals = salign.to_world(
                    src_points, current_space, self.calib, normals=src_normals)

                src_points_open3d, src_normals_open3d = transform_world_to_open3d_coordsys(
                    current_space, src_points.copy(), src_normals.copy())

                if sensor_type == "radar":
                    normalize = matplotlib.cm.colors.Normalize(
                        vmin=src_rgb.min(), vmax=src_rgb.max())
                    # http://www.kennethmoreland.com/color-advice/
                    # 'inferno', 'plasma', 'coolwarm'
                    s_map = matplotlib.cm.ScalarMappable(
                        cmap=matplotlib.colormaps.get_cmap("viridis"), norm=normalize)
                    src_rgb = s_map.to_rgba(src_rgb[:, 1])[:, :3]

                if vis_type == "Color":
                    self.add_geometry(sensor_type, create_open3d_pointcloud(
                        src_points_open3d, src_rgb, indices=src_indices),  update_camera=False)
                elif vis_type == "Normal":
                    self.add_geometry(sensor_type, create_open3d_pointcloud(
                        src_points_open3d, src_normals_open3d, indices=src_indices),  update_camera=False)
                elif vis_type == "Error":

                    if sensor_type == error_destination:
                        error = src_rgb
                    else:
                        self.compute_error(
                            sensor_type, error_destination, error_space)

                        error = self.sensors[sensor_type][error_space]["error"][error_destination]["colors"]

                        # projective error only works if sensor_type is transformed into the same space
                        # as the error_destination
                        # in case the sensor space we wish to visualize (current_space)
                        # deviates from this error_destination
                        # we have to transform the points first
                        if error_space != current_space or error.shape != src_points.shape:
                            src_points = self.sensors[sensor_type][error_space]["points"]
                            src_normals = self.sensors[sensor_type][error_space]["normals"]
                            src_indices = self.sensors[sensor_type][error_space]["indices"]

                            src_points, src_normals = salign.transform(
                                src_points, error_space, current_space, self.calib, normals=src_normals)

                            src_points, src_normals = salign.to_world(
                                src_points, current_space, self.calib, normals=src_normals)

                            src_points_open3d, src_normals_open3d = transform_world_to_open3d_coordsys(
                                current_space, src_points.copy(), src_normals.copy())

                        # elif error.shape != src_points.shape:
                        #     src_points = self.sensors[sensor_type][error_space]["points"]
                        #     src_normals = self.sensors[sensor_type][error_space]["normals"]
                        #     src_indices = self.sensors[sensor_type][error_space]["indices"]

                        #     src_points, src_normals, src_points_open3d, src_normals_open3d, src_aux, src_indices = du.clip_bbs(self.calib,
                        #                                                                                                         src_points, src_normals, current_space,
                        #                                                                                                         aux=[
                        #                                                                                                             src_rgb, src_infrared, src_dependence],
                        #                                                                                                         indices=src_indices,
                        #                                                                                                         xmin=self.bbx[
                        #                                                                                                             0], xmax=self.bbx[1],
                        #                                                                                                         ymin=self.bby[
                        #                                                                                                             0], ymax=self.bby[1],
                        #                                                                                                         zmin=self.bbz[0], zmax=self.bbz[1])

                        #     src_points, src_normals = salign.to_world(
                        #         src_points, current_space, self.calib, normals=src_normals)

                        #     src_points_open3d, src_normals_open3d = transform_world_to_open3d_coordsys(
                        #         current_space, src_points.copy(), src_normals.copy())

                    self.sensor_data[sensor_type]["error_destination"] = error_destination
                    self.sensor_data[sensor_type]["error_space"] = error_space
                    if error.shape[0] > src_points_open3d.shape[0]:
                        error = error[np.unique(src_indices)]

                    self.add_geometry(sensor_type, create_open3d_pointcloud(
                        src_points_open3d, error, indices=src_indices),  update_camera=False)
                elif vis_type == "Dependence":
                    self.add_geometry(sensor_type, create_open3d_pointcloud(
                        src_points_open3d, src_dependence, indices=src_indices),  update_camera=False)
                elif vis_type == "Infrared":
                    if src_infrared is None:
                        src_infrared = src_rgb
                    else:

                        h = self.sensors[sensor_type][sensor_type]["height"]
                        w = self.sensors[sensor_type][sensor_type]["width"]

                        raw = self.sensors[sensor_type][sensor_type]["aux"].reshape(
                            h, w, 3)
                        cv2.imshow("Infrared", raw /
                                   raw.max())
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    self.add_geometry(sensor_type, create_open3d_pointcloud(
                        src_points_open3d, src_infrared, indices=src_indices), update_camera=False)

                self.sensor_data[sensor_type]["space"] = current_space
                self.sensor_data[sensor_type]["vis"] = vis_type
                self.sensor_data[sensor_type]["visible"] = True

            elif not self.sensor_data[sensor_type]["visible"]:
                self.change_geometry_visibility(sensor_type, visible)
                self.sensor_data[sensor_type]["visible"] = True
        else:
            self.change_geometry_visibility(sensor_type, visible)
            if "visible" in self.sensor_data[sensor_type]:
                self.sensor_data[sensor_type]["visible"] = False

    def compute_error(self, sensor_type, dest_space, common_space=None):

        if common_space == None:
            common_space = dest_space

        # projective error only works correctly for isometric meshes
        # since otherwise many points might map to the same projective pixel
        if self.error_type == ERROR_TYPES["projective"]:
            self.add_sensor(dest_space)

            self.transform_sensor(
                sensor_type, common_space, isometric=True)
            self.sensor_isometric[sensor_type].checked = True

            self.transform_sensor(
                dest_space, common_space, isometric=True)
            self.sensor_isometric[dest_space].checked = True

        if not common_space in self.sensors[sensor_type]:
            self.transform_sensor(
                sensor_type, common_space, isometric=self.sensor_data[sensor_type]["isometric"])
        if not common_space in self.sensors[dest_space]:
            self.transform_sensor(
                dest_space, common_space, isometric=self.sensor_data[dest_space]["isometric"])

        print("Compute Error: {} -> {} in {}".format(sensor_type,
              dest_space, common_space))

        if not "error" in self.sensors[sensor_type][common_space] or not dest_space in self.sensors[sensor_type][common_space]["error"]:

            src_points = self.sensors[sensor_type][common_space]["points"]
            src_rgb = self.sensors[sensor_type][common_space]["rgb"]
            src_depth = self.sensors[sensor_type][common_space]["depth"]
            src_normals = self.sensors[sensor_type][common_space]["normals"]
            src_infrared = self.sensors[sensor_type][common_space]["aux"]
            src_indices = self.sensors[sensor_type][common_space]["indices"]
            src_dependence_raw = self.sensors[sensor_type][common_space]["dependence_raw"]
            if sensor_type == common_space:
                src_mask = self.sensors[sensor_type][common_space]["mask"]

            dest_points = self.sensors[dest_space][common_space]["points"]
            dest_depth = self.sensors[dest_space][common_space]["depth"]
            dest_indices = self.sensors[dest_space][common_space]["indices"]

            common_proj = self.sensors[common_space][common_space]["proj"]
            common_depth = self.sensors[common_space][common_space]["depth"]
            dest_mask = self.sensors[common_space][common_space]["mask"]

            height, width = common_depth.shape

            src_mask = np.logical_and(
                src_points[:, 2] != 0, src_dependence_raw[:, 0] > self.dot_thresh)

            if common_space != "photogrammetry" and self.has_3d_threshold is not None:
                src_points_radar = salign.transform(
                    src_points, common_space, "radar", self.calib)

                for idx, thresh in enumerate(self.mask_bbmin):
                    if thresh is None:
                        continue
                    src_mask = np.logical_and(
                        src_mask, src_points_radar[:, idx] >= thresh)
                for idx, thresh in enumerate(self.mask_bbmax):
                    if thresh is None:
                        continue
                    src_mask = np.logical_and(
                        src_mask, src_points_radar[:, idx] <= thresh)

            # if sensor_type == "kinect" and common_space != "photogrammetry":
            #     src_points_radar = salign.transform(
            #         src_points, common_space, "radar", self.calib)

            #     src_mask = np.logical_and(
            #         src_mask, src_points_radar[:, 2] <= 0.7)

            src_mask = src_mask.astype(np.uint8)

            # src_mask = np.ones(src_points.shape[0])
            if self.error_type == ERROR_TYPES["chamfer"]:
                dest_mask = np.ones(dest_points.shape[0])
                if self.use_dest_mask:
                    dest_mask = dest_points[:, 2] != 0

                if common_space != "photogrammetry" and self.has_3d_threshold is not None:
                    dest_points_radar = salign.transform(
                        dest_points, common_space, "radar", self.calib)

                    for idx, thresh in enumerate(self.mask_bbmin):
                        if thresh is None:
                            continue
                        dest_mask = np.logical_and(
                            dest_mask, dest_points_radar[:, idx] >= thresh)
                    for idx, thresh in enumerate(self.mask_bbmax):
                        if thresh is None:
                            continue
                        dest_mask = np.logical_and(
                            dest_mask, dest_points_radar[:, idx] <= thresh)

                # if dest_space == "kinect" and common_space != "photogrammetry":
                #     dest_points_radar = salign.transform(
                #         dest_points, common_space, "radar", self.calib)

                #     dest_mask = np.logical_and(
                #         dest_mask, dest_points_radar[:, 2] <= 0.7)

            else:
                # projective error types
                dest_mask = np.ones((dest_depth.shape[0], dest_depth.shape[1]))
                if self.use_dest_mask:
                    dest_mask = (dest_depth != 0).astype(np.uint8)

                if common_space != dest_space and sensor_type == common_space:
                    # pc has to be projected into common space first, so 2D mask is not valid here
                    # let's just use the src mask
                    dest_mask = src_mask

            cmap_error = "coolwarm"
            error_metrics, src_error_colors, src_error = compute_visualization_metrics(sensor_type, dest_space, common_space,
                                                                                       src_points, src_normals, src_rgb, src_mask,
                                                                                       dest_points, dest_depth, dest_mask,
                                                                                       common_proj, height, width,
                                                                                       max_error=self.max_error, error_type=self.error_type,
                                                                                       cmap_error=cmap_error,
                                                                                       src_indices=src_indices,
                                                                                       dest_indices=dest_indices)
            if True:
                show_legend(src_error, max_error=self.max_error,
                            error_type=error_type_to_str(self.error_type))

            if not "error" in self.sensors[sensor_type][common_space]:
                self.sensors[sensor_type][common_space]["error"] = {}
            self.sensors[sensor_type][common_space]["error"][dest_space] = {
                "metrics": error_metrics,
                "colors": src_error_colors,
                "error": src_error
            }

    def export_error_metrics(self, sensor_type, metadata_filename):

        metrics_data = {}
        if os.path.exists(metadata_filename):
            with open(metadata_filename, "r") as f:
                metrics_data = json.load(f)

        def get_npy_filename():
            counter = 0
            npy_filename = os.path.join(os.path.dirname(
                metadata_filename), str(counter).zfill(6) + ".npy")
            while os.path.exists(npy_filename):
                counter = counter + 1
                npy_filename = os.path.join(os.path.dirname(
                    metadata_filename), str(counter).zfill(6) + ".npy")

            return npy_filename

        def dict_update_or_set(d, keys, values):
            if not keys[0] in d:
                d[keys[0]] = {}

            if len(keys) > 1:
                d[keys[0]] = dict_update_or_set(d[keys[0]], keys[1:], values)
            else:
                d[keys[0]] = values

            return d

        for common_space, values in self.sensors[sensor_type].items():
            if not "error" in values:
                continue
            src = sensor_type

            for dest, val in values["error"].items():
                error_file = get_npy_filename()

                values = {
                    "mean": val["metrics"][0].item(),
                    "stddev": val["metrics"][1].item(),
                    "max": val["metrics"][2].item(),
                    "chamfer": val["metrics"][3],
                    "file": os.path.basename(error_file),

                }
                if "metadata" in self.sensor_data:
                    values["metadata"] = self.sensor_data["metadata"]

                metrics_data = dict_update_or_set(
                    metrics_data, [src, dest, common_space, error_type_to_str(self.error_type)], values)

                np.save(error_file, val["error"])

        with open(metadata_filename, "w") as f:
            json.dump(metrics_data, f)

    def transform_sensor(self, sensor_type, common_space, isometric=False):
        if sensor_type == common_space:
            return

        if self.sensor_data[sensor_type]["space"] != common_space or (common_space in self.sensors[sensor_type] and self.sensors[sensor_type][common_space]["isometric"] != isometric):
            src_points = self.sensors[sensor_type][sensor_type]["points"]
            src_rgb = self.sensors[sensor_type][sensor_type]["rgb"]
            src_depth = self.sensors[sensor_type][sensor_type]["depth"]
            src_normals = self.sensors[sensor_type][sensor_type]["normals"]
            src_aux = self.sensors[sensor_type][sensor_type]["aux"]
            src_indices = self.sensors[sensor_type][sensor_type]["indices"]

            if not common_space in self.sensors:
                self.add_sensor(common_space)

            common_depth = self.sensors[common_space][common_space]["depth"]
            common_proj = self.sensors[common_space][common_space]["proj"]

            height, width = common_depth.shape

            src_points, src_normals = salign.transform(
                src_points, sensor_type, common_space, self.calib, normals=src_normals)

            scale = 1000.0
            if common_space == "radar":
                scale = 1.0

            src_infrared = src_aux

            if isometric:
                if common_space == "radar":
                    src_points[:, 1] *= -1
                    common_proj[:, 1] *= -1

                src_points, src_indices, [src_rgb, src_normals, src_infrared] = du.create_pixel_aligned_mesh(src_points, src_indices, common_proj, height, width,
                                                                                                             attributes=[
                                                                                                                 src_rgb, src_normals, src_aux],
                                                                                                             triangulation_threshold=self.triangulation_threshold*scale)

                src_points = src_points.reshape(-1, 3)
                if common_space == "radar":
                    src_points[:, 1] *= -1
                    common_proj[:, 1] *= -1

                src_rgb = src_rgb.reshape(-1, 3)
                src_normals = src_normals.reshape(-1, 3)
                if src_infrared is not None:
                    src_infrared = src_infrared.reshape(-1, 3)

                isometric_mesh_path = os.path.join(
                    self.sensor_data[sensor_type]["path"], "mesh_radar_isometric.obj")
                if self.sensor_data[sensor_type]["use_mask"] and os.path.exists(os.path.join(self.sensor_data[sensor_type]["path"], "mesh_masked.obj")):
                    isometric_mesh_path = os.path.join(
                        self.sensor_data[sensor_type]["path"], "mesh_radar_isometric_masked.obj")

                if sensor_type == "photogrammetry" and common_space == "radar" and not os.path.exists(isometric_mesh_path):
                    pc = o3d.geometry.TriangleMesh()
                    pc.vertices = o3d.utility.Vector3dVector(
                        src_points)
                    pc.vertex_colors = o3d.utility.Vector3dVector(
                        src_rgb)
                    pc.triangles = o3d.utility.Vector3iVector(
                        src_indices)
                    o3d.io.write_triangle_mesh(isometric_mesh_path, pc)

            cmap_view = "inferno"
            src_dependence_raw, src_dependence = compute_view_dependence(
                src_points, src_normals, common_space, common_proj, common_depth.shape[1], common_depth.shape[1], cmap_view)

            self.sensors[sensor_type][common_space] = {
                "points": src_points,
                "rgb": src_rgb,
                "depth": src_depth,
                "normals": src_normals,
                "indices": src_indices,
                "aux": src_infrared,
                "dependence_raw": src_dependence_raw,
                "dependence": src_dependence,
                "isometric": isometric
            }

    def compute_gt_mask(self, common_space):
        sensor_type = "photogrammetry"
        if not sensor_type in self.sensors:
            self.add_sensor("photogrammetry")

        src_points = self.sensors[sensor_type][sensor_type]["points"]
        src_rgb = self.sensors[sensor_type][sensor_type]["rgb"]
        src_depth = self.sensors[sensor_type][sensor_type]["depth"]
        src_normals = self.sensors[sensor_type][sensor_type]["normals"]
        src_aux = self.sensors[sensor_type][sensor_type]["aux"]
        src_indices = self.sensors[sensor_type][sensor_type]["indices"]

        if not common_space in self.sensors:
            _, _, dest_depth, _, _, dest_proj, _, _ = du.get_data(
                common_space, self.sensor_data[common_space]["path"], self.sensor_data[common_space]["index"], self.config,
                use_mask=False,
                radar_reconstruction_method=self.radar_reconstruction_method, averaging_factor=self.averaging,
                triangulation_threshold=self.triangulation_threshold,
                amplitude_filter_threshold_dB=self.sensor_data["radar"]["amplitude_filter_threshold_dB"],
                kinect_space=self.sensor_data["kinect"]["data_space"],
                use_intrinsic_parameters=self.sensor_data["radar"]["use_intrinsic_parameters"])
            common_depth = dest_depth
            common_proj = dest_proj
        else:
            common_depth = self.sensors[common_space][common_space]["depth"]
            common_proj = self.sensors[common_space][common_space]["proj"]

        height, width = common_depth.shape

        src_points, src_normals = salign.transform(
            src_points, sensor_type, common_space, self.calib, normals=src_normals)

        scale = 1000.0
        if common_space == "radar":
            scale = 1.0

        src_infrared = src_aux

        if True:
            if common_space == "radar":
                src_points[:, 1] *= -1
                common_proj[:, 1] *= -1

            src_points, src_indices, [src_rgb, src_normals, src_infrared] = du.create_pixel_aligned_mesh(src_points, src_indices, common_proj, height, width,
                                                                                                         attributes=[
                                                                                                             src_rgb, src_normals, src_aux],
                                                                                                         triangulation_threshold=self.triangulation_threshold*scale)

            if common_space == "radar":
                src_points[:, :, 1] *= -1
                common_proj[:, 1] *= -1

        return (src_points[:, :, 2] != 0).astype(np.uint8)

    def add_sensor(self, sensor_type):

        if not sensor_type in self.sensors:

            gt_mask = None
            if sensor_type != "photogrammetry" and ("use_gt_mask" in self.sensor_data[sensor_type] and self.sensor_data[sensor_type]["use_gt_mask"]):
                gt_mask = self.compute_gt_mask(sensor_type)

            src_points, src_rgb, src_depth, src_mask, src_normals, src_proj, src_mesh, src_aux = du.get_data(
                sensor_type, self.sensor_data[sensor_type]["path"], self.sensor_data[sensor_type]["index"], self.config,
                use_mask=self.sensor_data[sensor_type]["use_mask"],
                manual_mask=gt_mask,
                radar_reconstruction_method=self.radar_reconstruction_method, averaging_factor=self.averaging,
                triangulation_threshold=self.triangulation_threshold,
                amplitude_filter_threshold_dB=self.sensor_data["radar"]["amplitude_filter_threshold_dB"],
                kinect_space=self.sensor_data["kinect"]["data_space"],
                use_intrinsic_parameters=self.sensor_data["radar"]["use_intrinsic_parameters"])

            if self.sensor_data[sensor_type]["mask_erosion"] > 0 and src_mask is not None:

                erosion = self.sensor_data[sensor_type]["mask_erosion"]
                src_mask_orig = src_mask.copy()
                while True:
                    src_mask = cv2.erode(src_mask_orig.astype(np.uint8),
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (erosion, erosion)))
                    elem, counts = np.unique(src_mask, return_counts=True)
                    erosion = max(0, erosion - 1)
                    if elem.shape[0] > 1 and counts[elem == 1] > 10:
                        break

                src_points = src_points.reshape(
                    src_depth.shape[0], src_depth.shape[1], 3) * src_mask.astype(np.float32)[:, :, None]
                src_points = src_points.reshape(-1, 3)
                src_depth = src_depth * src_mask.astype(np.float32)

            src_dependence_raw, src_dependence = compute_view_dependence(
                src_points, src_normals, sensor_type, src_proj, src_depth.shape[1], src_depth.shape[0], "inferno")

            self.sensors[sensor_type] = {
                sensor_type: {
                    "points": src_points,
                    "rgb": src_rgb,
                    "depth": src_depth,
                    "mask": src_mask,
                    "normals": src_normals,
                    "width": src_depth.shape[1],
                    "height": src_depth.shape[0],
                    "proj": src_proj,
                    "indices": src_mesh[1],
                    "mesh_points": src_mesh[0],
                    "aux": src_aux[0] if len(src_aux) > 0 else None,
                    "dependence_raw": src_dependence_raw,
                    "dependence": src_dependence
                }
            }

            self.sensor_data[sensor_type]["space"] = sensor_type
            if not "visible" in self.sensor_data[sensor_type]:
                self.sensor_data[sensor_type]["visible"] = True

    def _update_data(self, changed_clipping=False):
        for sensor_type in self.sensors.keys():
            if "visible" in self.sensor_data[sensor_type]:
                self.show_sensor(
                    sensor_type, self.sensor_data[sensor_type]["visible"], changed_clipping=changed_clipping)

    def _on_mask_change(self, name, show):
        self.change_mask(name, show)

    def _on_isometric_change(self, name, show):
        self.change_isometric(name, show)

    def _on_sensor_space_change(self, name, index):
        self.current_space = SensorVisualizerSettings.SENSORS[self._sensor_space.selected_index]
        self._update_data()

    def _on_vis_type_change(self, name, index):
        # self.vis_type = SensorVisualizerSettings.VIS_MODES[self._vis_type.selected_index]
        self.vis_type = SensorVisualizerSettings.VIS_MODES[index]
        self._update_data()

    def _on_error_destination_changed(self, name, index):
        # self.error_destination = SensorVisualizerSettings.SENSORS[
        #     self._error_destination.selected_index]
        self.error_destination = SensorVisualizerSettings.SENSORS[index]
        self._update_data()

    def _on_error_space_is_sensor(self, name, val):
        self.error_space_is_sensor = val
        self._update_data()

    def _on_minbbx_change(self, offset, event=None):
        old_val = self.bbx[0]
        if offset == 0:
            size = float(event)
        else:
            size = old_val + offset

        self.bbx[0] = min(size, self.bbx[1])
        self.bbx_text[0].text_value = str(self.bbx[0])
        if self.bbx[0] != old_val:
            self._update_data(changed_clipping=True)

    def _on_maxbbx_change(self, offset, event=None):
        old_val = self.bbx[1]
        if offset == 0:
            size = float(event)
        else:
            size = old_val + offset

        self.bbx[1] = max(size, self.bbx[0])
        self.bbx_text[1].text_value = str(self.bbx[1])

        if self.bbx[1] != old_val:
            self._update_data(changed_clipping=True)

    def _on_minbby_change(self, offset, event=None):
        old_val = self.bby[0]
        if offset == 0:
            size = float(event)
        else:
            size = old_val + offset

        self.bby[0] = min(size, self.bby[1])
        self.bby_text[0].text_value = str(self.bby[0])

        if self.bby[0] != old_val:
            self._update_data(changed_clipping=True)

    def _on_maxbby_change(self, offset, event=None):
        old_val = self.bby[1]
        if offset == 0:
            size = float(event)
        else:
            size = old_val + offset

        self.bby[1] = max(size, self.bby[0])
        self.bby_text[1].text_value = str(self.bby[1])

        if self.bby[1] != old_val:
            self._update_data(changed_clipping=True)

    def _on_minbbz_change(self, offset, event=None):
        old_val = self.bbz[0]
        if offset == 0:
            size = float(event)
        else:
            size = old_val + offset

        self.bbz[0] = min(size, self.bbz[1])
        self.bbz_text[0].text_value = str(self.bbz[0])

        if self.bbz[0] != old_val:
            self._update_data(changed_clipping=True)

    def _on_maxbbz_change(self, offset, event=None):
        old_val = self.bbz[1]
        if offset == 0:
            size = float(event)
        else:
            size = old_val + offset

        self.bbz[1] = max(size, self.bbz[0])
        self.bbz_text[1].text_value = str(self.bbz[1])

        if self.bbz[1] != old_val:
            self._update_data(changed_clipping=True)


def error_type_to_str(error_type):
    for k, v in ERROR_TYPES.items():
        if error_type == v:
            return k


def transform_world_to_open3d_coordsys(sensor_type, points, normals=None):
    points[:, 2] *= -1
    if normals is not None:
        normals[:, 2] *= -1

    # if sensor_type != "photogrammetry":
    #     points[:, 2] *= -1
    #     if normals is not None:
    #         normals[:, 2] *= -1

    # if sensor_type != "radar":
    #     points[:, 1] *= -1
    #     normals[:, 1] *= -1

    return points, normals


def transform_error_to_meters(sensor_type, error, error_type):
    if error_type == ERROR_TYPES["curvature"]:
        return error

    if sensor_type == "radar":
        if error_type == ERROR_TYPES["projective"]:
            error = error / 1000.0
    if sensor_type == "kinect":
        error = error / 1000.0
    if sensor_type == "zed":
        error = error / 1000.0
    if sensor_type == "realsense":
        error = error / 1000.0
    if sensor_type == "photogrammetry":
        pass

    return error


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


def compute_view_dependence(points, normals, common_space, common_proj, width, height, cmap_view):
    points_common_projective = points.copy()

    if (common_space == "radar"):
        points_common_projective[:, 1] *= -1

    dependence_raw = metrics.compute_view_dependence(points_common_projective,
                                                     normals,
                                                     common_proj,
                                                     width, height)

    normalize = matplotlib.cm.colors.Normalize(
        vmin=0, vmax=1)
    # http://www.kennethmoreland.com/color-advice/
    # 'inferno', 'plasma', 'coolwarm'
    s_map = matplotlib.cm.ScalarMappable(
        cmap=matplotlib.colormaps.get_cmap(cmap_view), norm=normalize)
    dependence = s_map.to_rgba(dependence_raw[:, 0])[:, :3]

    return dependence_raw, dependence


def compute_visualization_metrics(src, dest, common_space, src_points_common_, src_normals, src_rgb, src_valid_mask,
                                  dest_points_common_, dest_depth, dest_valid_mask, common_proj,
                                  height, width,
                                  src_indices=None,
                                  dest_indices=None,
                                  max_error=0.01,
                                  error_type=ERROR_TYPES["projective"],
                                  cmap_error="coolwarm"):

    src_shape = src_points_common_.shape[0]
    dest_shape = dest_points_common_.shape[0]

    src_points_common = src_points_common_.copy()
    src_normals_common = src_normals.copy()
    dest_points_common = dest_points_common_.copy()
    src_valid_mask_common = src_valid_mask.copy()
    dest_valid_mask_common = dest_valid_mask.copy()

    if src_indices is not None:
        src_points_common = src_points_common[np.unique(
            src_indices)]
        src_normals_common = src_normals_common[np.unique(
            src_indices)]
        src_valid_mask_common = src_valid_mask[np.unique(src_indices)]
    if dest_indices is not None:
        dest_points_common = dest_points_common[np.unique(
            dest_indices)]
        if error_type == ERROR_TYPES["chamfer"]:
            dest_valid_mask_common = dest_valid_mask[np.unique(dest_indices)]

    src_points_common_projective = src_points_common.copy()
    dest_points_common_projective = dest_points_common.copy()

    valid = np.ones(src_shape) > 0

    chamfer = 0
    if error_type == ERROR_TYPES["chamfer"]:

        error, valid = metrics.nearest_neighbor_error(
            src_points_common, dest_points_common, mask_src=src_valid_mask_common,
            mask_dest=dest_valid_mask_common)
        # does not work because PCs do not have the same dimension
        error2, valid2 = metrics.nearest_neighbor_error(
            dest_points_common, src_points_common, mask_src=dest_valid_mask_common,
            mask_dest=src_valid_mask_common)
        chamfer = 0.5 * np.abs(error[valid]).mean() + \
            0.5 * np.abs(error2[valid2]).mean()

    elif error_type == ERROR_TYPES["projective"]:
        if common_space != dest:
            dest_depth = pu.project(
                dest_points_common_projective, common_proj, height, width)

        error, valid = metrics.projective_error_without_open3d(
            src_points_common_projective, dest_depth, common_proj, mask_src=src_valid_mask_common,
            mask_dest=dest_valid_mask_common)

    error = transform_error_to_meters(common_space, error, error_type)
    chamfer = transform_error_to_meters(common_space, chamfer, error_type)

    mean_error = np.abs(error[valid]).mean()
    stddev_error = np.abs(error[valid]).std()
    if (np.abs(error[valid]).max() > (error[valid]).max()):
        max_val_error = - np.abs(error[valid]).max()
    else:
        max_val_error = (error[valid]).max()

    # chamfer dist is the sum of error, hausdorff is the max
    print(
        f"{src}->{dest}: Error type {error_type_to_str(error_type)} mean: {mean_error} (+- {stddev_error}), max: {max_val_error}")

    if max_error is None:
        max_error = np.max(np.abs(error[valid]))
    min_error = -max_error
    # if (np.min(error[valid]) > 0):
    #     min_error = 0

    print("[ERROR] min: {}, max: {} -- visualized min: {}, max: {}".format(
        np.min(error[valid]), np.max(error[valid]), min_error, max_error))

    # normalize = matplotlib.cm.colors.Normalize(
    #     vmin=min_error, vmax=max_error)
    normalize = matplotlib.colors.SymLogNorm(
        vmin=min_error, vmax=max_error, linthresh=0.005)
    # http://www.kennethmoreland.com/color-advice/
    # 'inferno', 'plasma', 'coolwarm'
    s_map = matplotlib.cm.ScalarMappable(
        cmap=matplotlib.colormaps.get_cmap(cmap_error), norm=normalize)
    colors = s_map.to_rgba(error)[:, :3]
    colors[np.logical_not(valid)] = np.array([0.0, 0.0, 0.0]).reshape(-1, 3)

    if src_indices is not None:
        colors_ = np.ones((src_shape, 3))
        if error.shape[0] < colors_.shape[0]:
            colors_[np.unique(src_indices)] = colors
        else:

            colors_ = colors
        colors = colors_.copy()

    return (mean_error, stddev_error, max_val_error, chamfer), colors, error[valid]


def show_legend(error, max_error=0.01, cmap="coolwarm", error_type="chamfer"):
    if max_error is None:
        max_error = np.max(np.abs(error))
    # if error.min() < 0:
    min_error = -max_error
    if error.min() >= 0:
        min_error = 0

    # normalize = matplotlib.cm.colors.Normalize(
    #     vmin=min_error, vmax=max_error)
    linthresh = 0.005
    normalize = matplotlib.colors.SymLogNorm(
        vmin=min_error, vmax=max_error, linthresh=linthresh)

    normalize_zero_one = matplotlib.colors.Normalize(
        vmin=0, vmax=1)
    # http://www.kennethmoreland.com/color-advice/
    # 'inferno', 'plasma', 'coolwarm'
    s_map = matplotlib.cm.ScalarMappable(
        cmap=matplotlib.colormaps.get_cmap(cmap), norm=normalize_zero_one)

    length = 100
    linspace_length = 11
    if error.min() >= 0:
        min_error = 0
        length = length // 2
        linspace_length = linspace_length // 2

    linspace = np.linspace(0, 1, length+1)
    value_error = normalize.inverse(linspace)

    color_legend = s_map.to_rgba(linspace)[:, :3]
    color_legend = np.stack([color_legend] * 5, axis=0)
    ticks = value_error
    indices = np.arange(0, 101, 5)

    text_val_scale = 100
    if error_type == ERROR_TYPES["curvature"]:
        text_val_scale = 1

    # space_to_ticks = ((ticks - min_error) /
    #                   (max_error - min_error)) * length
    space_to_ticks = linspace * length

    ticks = ticks[indices]
    space_to_ticks = space_to_ticks[indices]

    font_family = "Times New Roman"
    title_font_family = "Times New Roman"

    scale_fig = px.imshow(color_legend)
    scale_fig.update_xaxes(
        tickprefix="cm", nticks=linspace_length, tickvals=space_to_ticks.tolist(), ticktext=(np.round(ticks * text_val_scale, 1)).tolist())
    scale_fig.update_layout(
        font_family=font_family,
        title_font_family=title_font_family

    )
    for val in space_to_ticks.tolist():
        scale_fig.add_vline(x=val, line_width=1,
                            line_dash="dot", line_color="black")
    scale_fig.show()

    # scale_fig.write_image("scale.svg")


def reset_visualizer_options():
    global SRC_MODE
    global RENDER_SRC
    global RENDER_DEST

    RENDER_SRC = [True]
    RENDER_DEST = [True]
    SRC_MODE = ["color"]


def increase_point_size(vis):
    opt = vis.get_render_option()
    opt.point_size = opt.point_size + 1


def toggle_background_color(vis):
    opt = vis.get_render_option()
    if opt.background_color[0] == 0:
        opt.background_color = (np.ones(3))
    else:
        opt.background_color = (np.zeros(3))


def update(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, vis):
    vis.clear_geometries()
    if render_src[0]:
        if src_mode[0] == "color":
            vis.add_geometry(pcs_src[0], reset_bounding_box=False)
        elif src_mode[0] == "error":
            vis.add_geometry(pcs_src[1], reset_bounding_box=False)
        elif src_mode[0] == "dependence":
            vis.add_geometry(pcs_src[2], reset_bounding_box=False)
        elif src_mode[0] == "normal":
            vis.add_geometry(pcs_src[3], reset_bounding_box=False)
        elif src_mode[0] == "infrared":
            vis.add_geometry(pcs_src[4], reset_bounding_box=False)
        elif src_mode[0] == "curvature":
            vis.add_geometry(pcs_src[5], reset_bounding_box=False)

    if render_dest[0]:
        if dest_mode[0] == "color":
            vis.add_geometry(pcs_dest[0], reset_bounding_box=False)
        elif dest_mode[0] == "normal":
            vis.add_geometry(pcs_dest[1], reset_bounding_box=False)
        elif dest_mode[0] == "dependence":
            vis.add_geometry(pcs_dest[2], reset_bounding_box=False)
        elif dest_mode[0] == "curvature":
            vis.add_geometry(pcs_dest[3], reset_bounding_box=False)
        elif dest_mode[0] == "kinect":
            vis.add_geometry(pcs_dest[-1], reset_bounding_box=False)

    return True


def toggle_src(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, update_func, vis):
    render_src[0] = not render_src[0]
    return update_func(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, vis)


def render_color(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, update_func, vis):
    src_mode[0] = "color"
    dest_mode[0] = "color"
    return update_func(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, vis)


def render_normals(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, update_func, vis):
    src_mode[0] = "normal"
    dest_mode[0] = "normal"
    return update_func(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, vis)


def render_dependence(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, update_func, vis):
    src_mode[0] = "dependence"
    dest_mode[0] = "dependence"
    return update_func(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, vis)


def render_error(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, update_func, vis):
    src_mode[0] = "error"
    return update_func(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, vis)


def render_infrared(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, update_func, vis):
    src_mode[0] = "infrared"
    return update_func(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, vis)


def render_curvature(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, update_func, vis):
    src_mode[0] = "curvature"
    dest_mode[0] = "curvature"
    return update_func(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, vis)


def toggle_dest(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, update_func, vis):
    render_dest[0] = not render_dest[0]
    return update_func(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, vis)


def toggle_sensor(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, update_func, vis):
    if dest_mode[0] != "kinect":
        dest_mode[0] = "kinect"
    else:
        dest_mode[0] = "color"
    return update_func(render_src, render_dest, src_mode, dest_mode, pcs_src, pcs_dest, vis)


def open_legend(error, max_error, cmap, error_type, vis):
    show_legend(error, max_error, cmap, error_type)


def take_screenshot(vis, filename="screenshot.png"):
    if filename == "screenshot.png":
        counter = 0
        while os.path.exists(filename):
            counter = counter + 1
            tag = str(counter).zfill(3)
            filename = "screenshot_" + tag + ".png"
    print(f"Save Screenshot to: {filename}")
    if (os.path.exists(filename)):
        os.remove(filename)
    vis.capture_screen_image(filename, do_render=True)


def render_to_image(rec_config, pc, rgb, w, h):

    renderer_pc = o3d.visualization.rendering.OffscreenRenderer(w, h)
    renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))

    if "intrinsics" in rec_config:
        renderer_pc.scene.camera.set_projection(
            rec_config["intrinsics"], rec_config["near"], rec_config["far"], float(w), float(h))

    else:
        renderer_pc.scene.camera.set_projection(
            o3d.visualization.rendering.Camera.Projection.Ortho, rec_config["xmin"], rec_config["xmax"], rec_config["ymin"], rec_config["ymax"], rec_config["zmin"], rec_config["zmax"])

    center = [0, 0, -1]  # look_at target
    eye = [0, 0, 0]  # camera position
    up = [0, 1, 0]  # camera orientation

    renderer_pc.scene.camera.look_at(center, eye, up)

    view = renderer_pc.scene.camera.get_view_matrix()
    proj = renderer_pc.scene.camera.get_projection_matrix()

    inv_proj = np.linalg.inv(proj)
    matrix = np.matmul(np.linalg.inv(renderer_pc.scene.camera.get_view_matrix(
    )), inv_proj)

    col_scale = 1.0
    if rgb[pc[:, 2] != 0].max() > 1.0:
        col_scale = 255.0

    pc_src = o3d.geometry.PointCloud()
    pc_src.points = o3d.utility.Vector3dVector(
        pc[pc[:, 2] != 0].reshape(-1, 3))
    pc_src.colors = o3d.utility.Vector3dVector(
        rgb[pc[:, 2] != 0].reshape(-1, 3) / col_scale)
    # pc_src.paint_uniform_color([0, 0.7, 0.7])
    renderer_pc.scene.add_geometry(
        "pcd", pc_src, o3d.visualization.rendering.MaterialRecord())
    rgb_image = np.asarray(renderer_pc.render_to_image())

    if "intrinsics" in rec_config:
        linear_depth_image = np.asarray(
            renderer_pc.render_to_depth_image(z_in_view_space=True)).astype(np.float32)
        linear_depth_image[np.isinf(linear_depth_image)] = 0

        depth_image = (proj[2][2] * -linear_depth_image + proj[2][3])
        depth_image[linear_depth_image == 0] = 0
        depth_image[linear_depth_image > 0] = depth_image[linear_depth_image >
                                                          0] / linear_depth_image[linear_depth_image > 0]

    else:
        depth_image = np.asarray(
            renderer_pc.render_to_depth_image(z_in_view_space=False)).astype(np.float32)
        linear_depth_image = depth_image * \
            (rec_config["zmax"] - rec_config["zmin"]) + rec_config["zmin"]
        depth_image = depth_image * 2.0 - 1.0

    return linear_depth_image, depth_image, rgb_image, matrix


def create_open3d_pointcloud(points, colors, indices=None):

    if indices is None:
        assert points.shape[0] == colors.shape[0], "Points: {} colors: {}".format(
            points.shape, colors.shape)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(
            points.reshape(-1, 3))
        pc.colors = o3d.utility.Vector3dVector(
            colors.reshape(-1, 3))
    else:
        assert points[np.unique(
            indices)].shape[0] == colors[np.unique(indices)].shape[0], "Points: {} colors: {}".format(points[np.unique(
                indices)].shape, colors[np.unique(indices)].shape)
        pc = o3d.geometry.TriangleMesh()
        pc.vertices = o3d.utility.Vector3dVector(
            points.reshape(-1, 3))
        pc.triangles = o3d.utility.Vector3iVector(
            indices.reshape(-1, 3))
        pc.vertex_colors = o3d.utility.Vector3dVector(
            colors.reshape(-1, 3))
    return pc
