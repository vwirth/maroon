# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
import math
import functools

isMacOS = (platform.system() == "Darwin")


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


class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.UNLIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.UNLIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    def __init__(self, width, height):
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height, 0, 0)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        self.separation_height = separation_height
        self.em = em

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button("Fly")
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Environment")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)

        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label("Lighting profiles"))
        view_ctrls.add_child(self._profiles)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)

        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path +
                             "/*_ibl.ktx"):

            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(advanced)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))

        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(
            self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        self._wireframe = gui.Checkbox("Wireframe")
        self._wireframe.set_on_checked(self._on_show_wireframe)

        self._triangles = gui.Checkbox("Triangles")
        self._triangles.set_on_checked(self._on_show_triangles)
        self._triangles.checked = True

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        grid.add_child(self._wireframe)
        grid.add_child(self._triangles)

        material_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)

        self._geometry = gui.CollapsableVert("Geometry", 0,
                                             gui.Margins(em, 0, 0, 0))

        self._geometry.set_is_open(False)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self._geometry)

        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...",
                               AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, False)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        self._apply_settings()

        self._geometries = {}
        self._wireframes = {}
        self._pointclouds = {}

    def change_geometry_visibility(self, name, do_render):
        if not name in self._geometries:
            return
        if not self.has_geometry(name):
            return
        self._scene.scene.show_geometry(name, do_render)

        if do_render:
            if not self._triangles.checked:
                self._show_triangle(name, self._triangles.checked)
            if self._wireframe.checked:
                self._show_wireframe(name, self._wireframe.checked)
        # if not do_render:
        #     self._scene.scene.remove_geometry(name)
        # else:
        #     self._add_geometry(
        #         name, self._geometries[name]["geo"], is_model=self._geometries[name]["is_model"], update_camera=False)

    def update_geometry(self, name, update_camera=False):
        geo = self.get_geometry(name)
        self.add_geometry(
            name, geo["geo"], update_camera=update_camera)

    def get_pointcloud_center(self):
        center = np.array([0, 0, 0])
        cnt = 0
        for name, pc in self._pointclouds.items():
            if not self._scene.scene.geometry_is_visible(name):
                continue
            center = center + pc.get_axis_aligned_bounding_box(
            ).get_center()
            cnt = cnt + 1
        if cnt > 0:
            center = center / cnt

        return center

    def get_pointcloud_aabb(self):
        bbmax = np.array([-math.inf, -math.inf, -math.inf])
        bbmin = np.array([math.inf, math.inf, math.inf])
        for name, pc in self._pointclouds.items():
            if not self._scene.scene.geometry_is_visible(name):
                continue

            bbmax = np.maximum(
                bbmax, pc.get_axis_aligned_bounding_box().get_max_bound())
            bbmin = np.minimum(
                bbmin, pc.get_axis_aligned_bounding_box().get_min_bound())

        return bbmin, bbmax

    def delete_geometries(self):
        keys = list(self._geometries.keys())
        for name in keys:
            self.remove_geometry(name)

        self._geometries = {}
        self._pointclouds = {}
        self._wireframes = {}

    def rotate_pc(self, pc, R, center):
        verts = np.array(pc.vertices)
        verts_rot = np.matmul(
            verts - center[None, :], R.transpose()) + center[None, :]
        pc.vertices = o3d.utility.Vector3dVector(verts_rot)
        return pc

    def translate_pc(self, pc, t):
        verts = np.array(pc.vertices) + t[None, :]
        pc.vertices = o3d.utility.Vector3dVector(verts)
        return pc

    def _center_pointclouds_around_camera(self):
        pc_center = self.get_pointcloud_center()
        bb_min, bb_max = self.get_pointcloud_aabb()
        fovy = self._scene.scene.camera.get_field_of_view()
        # fovy = 50.0
        area = (bb_max - bb_min) * 0.5
        d = area[1] / np.tan(np.radians(fovy * 0.5))
        offset = -(pc_center[2] + d)

        camera_pos = pc_center
        camera_pos[2] -= d
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            bb_min - np.array([0, 0, offset]), bb_max - np.array([0, 0, offset]))
        pc_center = pc_center - np.array([0, 0, offset])
        # self._scene.setup_camera(
        #     fovy, aabb, pc_center)
        self._scene.setup_camera(
            25, o3d.geometry.AxisAlignedBoundingBox(bb_min, bb_max), self.get_pointcloud_center())
        self.window.post_redraw()

    def _rotate_around_scene_xyz(self, xyz, center=None, update_camera=True):

        bounds = self._scene.scene.bounding_box
        R = o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz(
            np.array(xyz))

        if center is None:
            center = bounds.get_center()
        for geo_name, geo in self._geometries.items():
            # self._geometries[geo_name]["geo"].rotate(R=R, center=center)
            if not self._scene.scene.geometry_is_visible(geo_name):
                continue
            self._geometries[geo_name]["geo"] = self.rotate_pc(
                self._geometries[geo_name]["geo"], R, center)
            self.update_geometry(geo_name, update_camera=update_camera)

    def _translate_scene_xyz(self, xyz, center=None, update_camera=True):
        for geo_name, geo in self._geometries.items():
            # self._geometries[geo_name]["geo"].rotate(R=R, center=center)
            if not self._scene.scene.geometry_is_visible(geo_name):
                continue
            self._geometries[geo_name]["geo"] = self.translate_pc(
                self._geometries[geo_name]["geo"], np.array(xyz).reshape(-1))
            self.update_geometry(geo_name, update_camera=update_camera)

    def _center_camera(self):
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(50, bounds, bounds.get_center())
        self.window.post_redraw()

    def _center_camera_around_pc(self):
        bb_min, bb_max = self.get_pointcloud_aabb()
        bb_min = bb_min - 0.05
        bb_max = bb_max + 0.05

        area = bb_max - bb_min

        max_dim = 1.25 * np.max(area)
        # this is how setup_camera with arcball setup calculates the cam center
        cam_center = self.get_pointcloud_center() + np.array([0, 0, max_dim])
        fovy = np.degrees(np.arctan((area[1] * 0.5) / (np.max(area) * 0.5)))

        # print("fovy: ", fovy)
        fovy = 50

        self._scene.setup_camera(
            fovy, o3d.geometry.AxisAlignedBoundingBox(bb_min, bb_max), self.get_pointcloud_center())
        self.window.post_redraw()

    def _add_geometry(self, name, geo, is_model=False, update_camera=True):
        try:
            if self.has_geometry(name):
                return False
            if is_model:
                # TriangleMeshModel model
                self._scene.scene.add_model(name, geo)
            else:
                # Point cloud
                self._scene.scene.add_geometry(name, geo,
                                               self.settings.material)

            if update_camera:
                bounds = self._scene.scene.bounding_box
                self._scene.setup_camera(50, bounds, bounds.get_center())

            return True
        except Exception as e:
            print(e)
        return False

    def add_geometry(self, name, geo, update_camera=True):
        em = self.window.theme.font_size
        separation_height = int(round(0.1 * em))
        if type(geo) == o3d.visualization.rendering.TriangleMeshModel:
            is_model = True
        else:
            is_model = False
            # geometry_type = geo.get_geometry_type()
            # is_mesh = geometry_type & o3d.io.CONTAINS_TRIANGLES

        if self._add_geometry(name, geo, is_model=is_model, update_camera=update_camera):
            if not (name in self._geometries.keys()):
                horiz = o3d.visualization.gui.Horiz()
                checks = o3d.visualization.gui.CheckableTextTreeCell(
                    name, True, functools.partial(self.change_geometry_visibility, name))

                def testfunc():
                    self.remove_geometry(name)
                remove_btn = gui.Button("Del")
                remove_btn.set_on_clicked(testfunc)
                remove_btn.vertical_padding_em = 0
                horiz.add_child(checks)
                horiz.add_child(remove_btn)

                self._geometry.add_fixed(separation_height)
                self._geometry.add_child(horiz)
                self.window.set_needs_layout()
                # self._geometry.add_child(remove_btn)

                self._geometries[name] = {
                    "geo": geo,
                    "is_model": is_model,
                    "gui": horiz,
                }
        else:
            self._remove_geometry(name)
            self._add_geometry(name, geo, is_model=is_model,
                               update_camera=update_camera)
            self._geometries[name]["geo"] = geo
            self._geometries[name]["is_model"] = is_model
            if name in self._wireframes:
                del self._wireframes[name]
            if name in self._pointclouds:
                del self._pointclouds[name]

        if not self._triangles.checked:
            self._show_triangle(name, self._triangles.checked)
        if self._wireframe.checked:
            self._show_wireframe(name, self._wireframe.checked)

    def _remove_geometry(self, name):

        if self._scene.scene.has_geometry(name):
            self._scene.scene.remove_geometry(name)

    def export_geometry(self, name, path):
        if self.has_geometry(name):
            geo = self._pointclouds[name]
            o3d.io.write_point_cloud(path, geo)

    def remove_geometry(self, name):

        self._remove_geometry(name)
        if name in self._geometries:
            print("Removing {}".format(name))
            self._geometries[name]["gui"].visible = False
            self._geometries[name]["gui"].enabled = False
            del self._geometries[name]["gui"]
            del self._geometries[name]

        if name in self._wireframes:
            del self._wireframes[name]

        if name in self._pointclouds:
            del self._pointclouds[name]

        self.window.set_needs_layout()

    def has_geometry(self, name):
        return self._scene.scene.has_geometry(name)

    def get_geometry(self, name):
        if not (name in self._geometries.keys()):
            return None
        return self._geometries[name]

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
            self.settings.material.shader == Settings.UNLIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _show_wireframe(self, name, show):
        if not name in self._geometries:
            return
        if not self._scene.scene.geometry_is_visible(name):
            return

        geo_name = name
        geo = self._geometries[name]

        if show:
            self._triangles.checked = True
            if geo_name in self._wireframes:
                visible = self._scene.scene.geometry_is_visible(geo_name)
                self._remove_geometry(geo_name)
                self._add_geometry(
                    geo_name, self._wireframes[geo_name], update_camera=False)
                self._scene.scene.show_geometry(name, visible)
            else:
                if geo["is_model"]:
                    new_mod = o3d.TriangleMeshModel()
                    meshes = []
                    for m_info in geo["geo"].meshes:
                        geometry_type = m_info.mesh.get_geometry_type()
                        is_mesh = geometry_type & o3d.io.CONTAINS_TRIANGLES
                        if not is_mesh:
                            continue
                        mean_color = np.array(
                            m_info.mesh.vertex_colors).transpose().mean(-1)
                        line = o3d.geometry.LineSet.create_from_triangle_mesh(
                            m_info.mesh)
                        line.paint_uniform_color(mean_color)
                        meshes.append(o3d.visualization.rendering.TriangleMeshModel.MeshInfo(
                            line, m_info.mesh_name, m_info.material_idx))
                    new_mod.meshes = meshes
                    new_mod.materials = geo["geo"].materials
                    self._wireframes[geo_name] = new_mod
                else:
                    geometry_type = geo["geo"].get_geometry_type()
                    is_mesh = geometry_type & o3d.io.CONTAINS_TRIANGLES
                    if is_mesh:
                        mean_color = np.array(
                            geo["geo"].vertex_colors).transpose().mean(-1)
                        line = o3d.geometry.LineSet.create_from_triangle_mesh(
                            geo["geo"])
                        line.paint_uniform_color(mean_color)
                        self._wireframes[geo_name] = line

                if geo_name in self._wireframes:
                    visible = self._scene.scene.geometry_is_visible(geo_name)
                    self._remove_geometry(geo_name)
                    self._add_geometry(
                        geo_name, self._wireframes[geo_name], is_model=geo["is_model"], update_camera=False)
                    self._scene.scene.show_geometry(name, visible)
                    # self.add_geometry(geo_name, self._wireframes[geo_name], update_camera=False)
        else:
            visible = self._scene.scene.geometry_is_visible(geo_name)
            self._remove_geometry(geo_name)
            self._add_geometry(geo_name, geo["geo"], update_camera=False)
            self._scene.scene.show_geometry(name, visible)

    def _on_show_wireframe(self, show):
        if show:
            self._triangles.checked = True
        for geo_name, geo in self._geometries.items():
            self._show_wireframe(geo_name, show)

    def clip_bb(self, name, bb_min, bb_max):
        if not name in self._geometries:
            return

        geo_name = name
        geo = self._geometries[name]["geo"]
        vertices = np.array(geo.vertices)
        indices = np.array(geo.triangles)

        for idx, (mi, ma) in enumerate(zip(bb_min, bb_max)):
            mask_0 = np.logical_and(
                vertices[indices[:, 0], idx] >= mi, vertices[indices[:, 0], idx] <= ma)
            mask_1 = np.logical_and(
                vertices[indices[:, 1], idx] >= mi, vertices[indices[:, 1], idx] <= ma)
            mask_2 = np.logical_and(
                vertices[indices[:, 2], idx] >= mi, vertices[indices[:, 2], idx] <= ma)
            mask_all = np.logical_and(mask_0, np.logical_and(mask_1, mask_2))

            indices = indices[mask_all]

        geo.vertices = o3d.utility.Vector3dVector(vertices)
        geo.triangles = o3d.utility.Vector3iVector(indices)
        self._geometries[name]["geo"] = geo
        self.update_geometry(name)

    def _show_triangle(self, name, show):
        if not name in self._geometries:
            return

        geo_name = name
        geo = self._geometries[name]
        if not show:
            if geo_name in self._pointclouds:
                visible = self._scene.scene.geometry_is_visible(geo_name)
                self._remove_geometry(geo_name)
                self._add_geometry(
                    geo_name, self._pointclouds[geo_name], update_camera=False)
                self._scene.scene.show_geometry(name, visible)
            else:
                if geo["is_model"]:
                    new_mod = o3d.TriangleMeshModel()
                    meshes = []
                    for m_info in geo["geo"].meshes:
                        geometry_type = m_info.mesh.get_geometry_type()
                        is_mesh = geometry_type & o3d.io.CONTAINS_TRIANGLES
                        if not is_mesh:
                            continue
                        pc = o3d.geometry.PointCloud()
                        indices = np.unique(m_info.mesh.triangles)
                        pc.points = o3d.utility.Vector3dVector(
                            np.array(m_info.mesh.vertices)[indices])
                        pc.colors = o3d.utility.Vector3dVector(
                            np.array(m_info.mesh.vertex_colors)[indices])

                        meshes.append(o3d.visualization.rendering.TriangleMeshModel.MeshInfo(
                            pc, m_info.mesh_name, m_info.material_idx))
                    new_mod.meshes = meshes
                    new_mod.materials = geo["geo"].materials
                    self._pointclouds[geo_name] = new_mod
                else:
                    geometry_type = geo["geo"].get_geometry_type()
                    is_mesh = geometry_type & o3d.io.CONTAINS_TRIANGLES
                    if is_mesh:
                        pc = o3d.geometry.PointCloud()
                        indices = np.unique(geo["geo"].triangles)
                        pc.points = o3d.utility.Vector3dVector(
                            np.array(geo["geo"].vertices)[indices])
                        pc.colors = o3d.utility.Vector3dVector(
                            np.array(geo["geo"].vertex_colors)[indices])

                        # print_coordinates(
                        #     "VERTICES: ", np.array(geo["geo"].vertices))
                        # print_coordinates("POINTS: ", np.array(
                        #     geo["geo"].vertices)[indices])

                        self._pointclouds[geo_name] = pc
                if geo_name in self._pointclouds:
                    visible = self._scene.scene.geometry_is_visible(geo_name)
                    self._remove_geometry(geo_name)
                    self._add_geometry(
                        geo_name, self._pointclouds[geo_name], is_model=geo["is_model"], update_camera=False)
                    self._scene.scene.show_geometry(name, visible)
                    # self.add_geometry(geo_name, self._wireframes[geo_name], update_camera=False)
        else:
            visible = self._scene.scene.geometry_is_visible(geo_name)
            self._remove_geometry(geo_name)
            self._add_geometry(geo_name, geo["geo"], update_camera=False)
            self._scene.scene.show_geometry(name, visible)

    def _on_show_triangles(self, show):
        if not show:
            self._wireframe.checked = False
        for geo_name, geo in self._geometries.items():
            self._show_triangle(geo_name, show)

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def close(self):
        self.window.close()
        o3d.visualization.gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.window.quit())
        gui.Application.instance.quit()
        o3d.visualization.gui.Application.instance.run_one_tick()

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def load(self, path):
        # self._scene.scene.clear_geometry()

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_model(path)
        if mesh is None:
            print("[Info]", path, "appears to be a point cloud")
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        # if geometry is not None or mesh is not None:
        #     try:
        #         if mesh is not None:
        #             # Triangle model
        #             self._scene.scene.add_model("__model__", mesh)
        #         else:
        #             # Point cloud
        #             self._scene.scene.add_geometry("__model__", geometry,
        #                                            self.settings.material)
        #         bounds = self._scene.scene.bounding_box
        #         self._scene.setup_camera(60, bounds, bounds.get_center())
        #     except Exception as e:
        #         print(e)
        name = "__model__"
        counter = 1
        while name in self._geometries:
            name = "__model" + str(counter) + "__"

        if mesh is not None:

            self.add_geometry(name, mesh)
        elif geometry is not None:
            self.add_geometry(name, geometry)

    def export_image(self, path, width=0, height=0):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)
        print("Rendering to: ", path)


def main():
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1024, 768)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
