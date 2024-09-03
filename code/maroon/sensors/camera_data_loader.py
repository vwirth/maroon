import os
import maroon.utils.file_ops as file_ops
import maroon.utils.pointcloud_utils as pu
import numpy as np
import open3d as o3d
import json
import cv2
import pymeshlab
import matplotlib
import math

import pycuda.driver as cuda
from maroon.cuda.cuda import load_kernel_from_cu


def project_depth_to_3d_camera(depth_image, intrinsics, extrinsics, distortion, sphere_mask=None, is_blender=False):
    I = np.eye(4)
    I[:3, :3] = np.linalg.inv(intrinsics)
    K = extrinsics

    wpos_list = []
    ids = []

    linearized_depth = (depth_image * (sphere_mask[:, :] > 0)).reshape(-1)

    u = np.linspace(0, depth_image.shape[1]-1,
                    depth_image.shape[1]).astype(np.int32)
    v = np.linspace(0, depth_image.shape[0]-1,
                    depth_image.shape[0]).astype(np.int32)
    px, py = np.meshgrid(u, v)
    ones = np.ones(px.shape)
    px_coords = np.stack([px, py, ones], axis=-1).reshape(-1, 3)
    px_coords = px_coords * linearized_depth[:, None]
    px_coords = np.concatenate([px_coords, ones.reshape(-1)[:, None]], axis=-1)

    if sphere_mask is not None:
        mask_ids, count = np.unique(sphere_mask[:, :], return_counts=True)
        sorted_idx = np.argsort(count)
        sphere_ids = mask_ids[sorted_idx][mask_ids[sorted_idx] > 0]

        for sid in sphere_ids:
            # select linearized coordinates that belong to sid
            valid_coords_id = np.array(np.nonzero(
                (depth_image * (sphere_mask[:, :] == sid)).reshape(-1)))[0]

            wpos = np.matmul(K, np.matmul(I, px_coords[valid_coords_id].transpose()))[
                :3].transpose()

            if is_blender:
                # transform into blender space
                tmp = wpos[:, 2].copy()
                wpos[:, 2] = wpos[:, 1]
                wpos[:, 1] = -tmp

            wpos_list.append(wpos)
            ids.append(sid)

    else:
        valid_coords = np.array(np.nonzero(linearized_depth))[0]
        wpos = (np.matmul(K, np.matmul(I, px_coords[valid_coords].transpose()))[
            :3].transpose())

        if is_blender:
            # transform into blender space
            tmp = wpos[:, 2].copy()
            wpos[:, 2] = wpos[:, 1]
            wpos[:, 1] = -tmp

        wpos_list.append(wpos)
        ids.append(-1)

    return wpos_list, ids


class CameraDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        assert os.path.exists(self.data_path)

        self.image_name_list = []
        for file in (os.listdir(os.path.join(data_path, "depth"))):
            if os.path.isfile(os.path.join(data_path, "depth", file)):
                file, file_ext = os.path.splitext(file)
                if file_ext == ".tif":
                    self.image_name_list.append(file)
        # print("image_name_list: ", self.image_name_list)

    def load_images(self):
        image_info = []
        rgbs, _ = file_ops.load_rgb(os.path.join(
            self.data_path, "rgb"), self.image_name_list)
        depths = file_ops.load_depth(os.path.join(
            self.data_path, "depth"), self.image_name_list, extension=".tif")

        beauty_shot = False
        if beauty_shot:
            index = 0
            depth_tmp = depths[index].copy()
            depth_tmp[depth_tmp > 500] = 500
            depth_tmp[depth_tmp == 0] = 500
            normalize = matplotlib.cm.colors.Normalize(
                vmin=depth_tmp.min(), vmax=depth_tmp.max())
            # http://www.kennethmoreland.com/color-advice/
            # 'inferno', 'plasma', 'coolwarm'
            s_map = matplotlib.cm.ScalarMappable(
                cmap=matplotlib.colormaps.get_cmap('Greys_r'), norm=normalize)
            colors = s_map.to_rgba(depth_tmp)[:, :, :3]
            print("colors: ", colors.shape)

            cv2.imshow("depth", colors)
            cv2.imshow("rgb", rgbs[index])
            cv2.waitKey(0)

            cv2.imwrite("beautiful_photo_depth.png",
                        (colors * 255).astype(np.uint8))
            cv2.imwrite("beautiful_photo_rgb.png", rgbs[index])
            exit(0)

        self.rgbs = rgbs
        self.depths = depths

    def get_image_data_from_filename(self, file_name):

        for idx, fn in enumerate(self.image_name_list):
            if file_name in fn:
                return self.get_image_data_from_idx(idx)

    def get_image_data_from_idx(self, idx):

        depth_image = self.depths[idx]
        I = np.eye(4)
        I[:3, :3] = np.linalg.inv(self.intrinsics[idx])
        # K = self.extrinsics[idx]
        K = np.eye(4)

        u = np.linspace(0, depth_image.shape[1]-1,
                        depth_image.shape[1]).astype(np.int32)
        v = np.linspace(0, depth_image.shape[0]-1,
                        depth_image.shape[0]).astype(np.int32)
        px, py = np.meshgrid(u, v)
        ones = np.ones(px.shape)
        px_coords = np.stack([px, py, ones], axis=-1)

        px_coords = px_coords * depth_image[:, :, None]
        px_coords = np.concatenate(
            [px_coords, ones[:, :, None]], axis=-1).reshape(-1, 4)

        wpos = (np.matmul(K, np.matmul(I, px_coords.transpose()))[
            :3].transpose())
        wpos = wpos.reshape(depth_image.shape[0], depth_image.shape[1], 3)

        return self.rgbs[idx], self.depths[idx], wpos

    def create_pixel_aligned_mesh(self, points, indices,
                                  proj, height, width,
                                  attributes=[], triangulation_threshold=pow(2, 16)):

        attributes_out = []
        points_out = None

        attr_index = 0
        for attr in attributes:

            if attr_index == 0:
                points_out, attribute_out = pu.project_interpolate(
                    points, indices, proj, height, width, src_color_attribute=attr)
            else:
                _, attribute_out = pu.project_interpolate(
                    points, indices, proj, height, width, src_color_attribute=attr)

            if attr is None:
                attributes_out.append(None)
            else:
                attributes_out.append(attribute_out)
            attr_index = attr_index + 1

        _, indices_out = pu.triangulate_image_pointcloud(
            points_out, threshold=triangulation_threshold)

        return points_out, indices_out, attributes_out

    def get_merged_image_data_from_filename(self, file_name):
        for idx, fn in enumerate(self.image_name_list):
            if file_name in fn:
                return self.get_merged_image_data_from_idx(idx)

    def get_extrinsics_from_filename(self, file_name):
        for idx, fn in enumerate(self.image_name_list):
            if file_name in fn:
                return self.extrinsics[idx]

    def get_merged_image_data_from_idx(self, idx):
        rgb, depth, wpos = self.get_image_data_from_idx(idx)
        mesh = self.get_mesh()
        points = np.array(mesh.vertices).reshape(-1, 3)
        triangles = np.array(mesh.triangles).reshape(-1)

        I = np.eye(4)
        I[:3, :3] = self.intrinsics[idx]
        I[3, 3] = 0
        I[3, 2] = 1
        K = self.extrinsics[idx]

        points_hom = np.concatenate(
            [points, np.ones((points.shape[0], 1))], axis=-1)
        points_proj = np.matmul(np.linalg.inv(K), points_hom.transpose())[
            :3].transpose().reshape(-1, 3).astype(np.float32)

        interpolated_points, interpolated_pixels, dest_depth_interpolated, _ = pu.project_interpolate(
            points_proj, triangles, I, self.depths[idx].shape[0], self.depths[idx].shape[1])
        dest_depth_interpolated = dest_depth_interpolated[:, :, 2]

        I_inv = np.eye(4)
        I_inv[:3, :3] = np.linalg.inv(self.intrinsics[idx])

        u = np.linspace(0, dest_depth_interpolated.shape[1]-1,
                        dest_depth_interpolated.shape[1]).astype(np.int32)
        v = np.linspace(0, dest_depth_interpolated.shape[0]-1,
                        dest_depth_interpolated.shape[0]).astype(np.int32)
        px, py = np.meshgrid(u, v)
        ones = np.ones(px.shape)
        px_coords = np.stack([px, py, ones], axis=-1)

        px_coords = px_coords * dest_depth_interpolated[:, :, None]
        px_coords = np.concatenate(
            [px_coords, ones[:, :, None]], axis=-1).reshape(-1, 4)

        wpos = (np.matmul(K, np.matmul(I_inv, px_coords.transpose()))[
            :3].transpose())
        wpos = wpos.reshape(
            dest_depth_interpolated.shape[0], dest_depth_interpolated.shape[1], 3)

        points_hom = np.concatenate(
            [interpolated_points, np.ones((interpolated_points.shape[0], 1))], axis=-1)
        interpolated_points = np.matmul(K, points_hom.transpose())[
            :3].transpose().reshape(-1, 3).astype(np.float32)

        return rgb, dest_depth_interpolated, wpos, interpolated_points, interpolated_pixels

    def load_calibration(self):
        if hasattr(self, "intrinsics") and len(self.intrinsics) > 0:
            return

        xml_path = os.path.join(self.data_path, "cams.xml")
        assert os.path.exists(xml_path)

        intrinsics = []
        extrinsics = []
        distortion = []
        for i, image_name in enumerate(self.image_name_list):
            I, E, D = file_ops.load_calibration(
                os.path.join(self.data_path, xml_path), image_name)
            intrinsics.append(I)
            extrinsics.append(E)
            distortion.append(D)

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.distortion = distortion

    def undistort(self, index):
        assert hasattr(self, "depths")
        assert hasattr(self, "intrinsics")
        assert len(self.depths) > 0
        assert len(self.depths) == len(self.intrinsics)
        assert index < len(self.depths)

        depth = self.depths[index]
        intrinsics = self.intrinsics[index]
        distortion = self.distortion[index]

        height = depth.shape[0]
        width = depth.shape[1]
        newcameramatrix, _ = cv2.getOptimalNewCameraMatrix(
            intrinsics, distortion, (width, height), 1, (width, height))
        undistorted_image = cv2.undistort(
            depth, intrinsics, distortion, None, newcameramatrix)
        cv2.imshow("undistorted", cv2.resize(
            undistorted_image, (800, 600)))
        cv2.waitKey(0)
        return undistorted_image

    def get_vertices_from_depth(self, use_masks=False, padding_x=30, padding_y=30, plane_id=1,
                                relative_blue_val=0.5, relative_red_val=0.5):
        assert hasattr(self, "depths")
        assert hasattr(self, "intrinsics")
        assert len(self.depths) > 0
        assert len(self.depths) == len(self.intrinsics)

        if use_masks:
            assert hasattr(self, "masks")
            assert len(self.depths) == len(self.masks)

            print("Unprojecting depth images")
            vertices_per_id = {}
            plane_vertices = []
            for (rgb_image, depth_image, sphere_mask, intrinsics, extrinsics, distortion) in zip(self.rgbs, self.depths, self.masks, self.intrinsics, self.extrinsics, self.distortion):
                sphere_mask_plane = mask_plane(
                    rgb_image, sphere_mask, roi_id=plane_id,
                    padding_x=padding_x, padding_y=padding_y,
                    relative_blue_val=relative_blue_val,
                    relative_red_val=relative_red_val)
                wpos, ids = project_depth_to_3d_camera(
                    depth_image, intrinsics, extrinsics, distortion,
                    sphere_mask=sphere_mask_plane)
                # cv2.imshow("depth: ", depth_image / depth_image.max())
                # cv2.imshow("mask: ", sphere_mask / sphere_mask.max())
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                for i, sid in enumerate(ids):
                    if sid == plane_id:
                        plane_vertices.append(wpos[i])
                        continue

                    # if sid % self.circle_id_multiple > 0:
                    #     continue
                    if sid not in vertices_per_id:
                        vertices_per_id[sid] = [wpos[i]]
                    else:
                        vertices_per_id[sid].append(wpos[i])

                        #  = np.concatenate(
                        #     [vertices_per_id[sid], wpos[i]], axis=0)

            updated_verts = {}
            for id, verts in vertices_per_id.items():
                if len(verts) < self.num_spheres:
                    print("Found invalid vertices with id: ", id)
                else:
                    updated_verts[id] = np.concatenate(verts, axis=0)

            return updated_verts, np.concatenate(plane_vertices)
        else:
            print("Unprojecting depth images")
            vertices = None
            for (depth_image, intrinsics, extrinsics, distortion) in zip(self.depths, self.intrinsics, self.extrinsics, self.distortion):
                wpos, ids = project_depth_to_3d_camera(
                    depth_image, intrinsics, extrinsics, distortion,
                    sphere_mask=None)

                if vertices is None:
                    vertices = wpos[i]
                else:
                    vertices = np.concatenate(
                        [vertices, wpos[0]], axis=0)
            return vertices

    def get_mesh(self, do_smoothing=False, use_mask=False, mask_idx=-1, mask_dilation=0):
        mesh_name = "mesh.obj"
        if use_mask:
            mesh_name = "mesh_masked.obj"
            if not os.path.exists(os.path.join(self.data_path, mesh_name)):
                mesh_name = "mesh.obj"

        if os.path.exists(os.path.join(self.data_path, "mesh_smoothed_cleaned.ply")):
            mesh_name = "mesh_smoothed_cleaned.ply"

        assert os.path.exists(os.path.join(self.data_path, mesh_name))

        if not do_smoothing:
            # mesh = o3d.io.read_triangle_mesh(
            #     os.path.join(self.data_path, "mesh.obj"))

            # for some reason this is necessary to use meshlab
            # operations later on as 'o3d.io.read_triangle_mesh'
            # yields different indices
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(os.path.join(self.data_path, mesh_name))
            mesh = ms.current_mesh()

            pc = o3d.geometry.TriangleMesh()
            pc.vertices = o3d.utility.Vector3dVector(
                mesh.vertex_matrix())
            pc.vertex_colors = o3d.utility.Vector3dVector(
                mesh.vertex_color_matrix()[:, :3])
            pc.vertex_normals = o3d.utility.Vector3dVector(
                mesh.vertex_normal_matrix()[:, :3])
            pc.triangles = o3d.utility.Vector3iVector(
                mesh.face_matrix())
            mesh = pc
        else:
            if not os.path.exists(os.path.join(self.data_path, mesh_name.split(".")[0]+"_smoothed.obj")):
                print("Apply mesh smoothing...")
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(os.path.join(self.data_path, mesh_name))
                ms.apply_coord_laplacian_smoothing(stepsmoothnum=30)
                ms.save_current_mesh(os.path.join(
                    self.data_path, mesh_name.split(".")[0]+"_smoothed.obj"))

            # mesh = o3d.io.read_triangle_mesh(
            #     os.path.join(self.data_path, "mesh_smoothed.obj"))

            # for some reason this is necessary to use meshlab
            # operations later on as 'o3d.io.read_triangle_mesh'
            # yields different indices
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(os.path.join(
                self.data_path, mesh_name.split(".")[0]+"_smoothed.obj"))
            mesh = ms.current_mesh()

            pc = o3d.geometry.TriangleMesh()
            pc.vertices = o3d.utility.Vector3dVector(
                mesh.vertex_matrix())
            pc.vertex_colors = o3d.utility.Vector3dVector(
                mesh.vertex_color_matrix()[:, :3])
            pc.vertex_normals = o3d.utility.Vector3dVector(
                mesh.vertex_normal_matrix()[:, :3])
            pc.triangles = o3d.utility.Vector3iVector(
                mesh.face_matrix())
            mesh = pc

        if mask_idx >= 0:
            mask_path = os.path.join(
                self.data_path, "mask", self.image_name_list[mask_idx]+".png")
            print("mask_path: ", mask_path)
            mask = cv2.imread(mask_path)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            if mask_dilation > 0:
                mask = cv2.erode(mask.astype(np.uint8),
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (mask_dilation, mask_dilation)))

            self.load_calibration()

            points = np.array(mesh.vertices).reshape(-1, 3)
            indices = np.array(mesh.triangles).reshape(-1, 3)
            I = self.intrinsics[mask_idx]
            K = self.extrinsics[mask_idx]

            points_hom = np.concatenate(
                [points, np.ones((points.shape[0], 1))], axis=-1)
            points_proj = np.matmul(np.linalg.inv(K), points_hom.transpose())[
                :3].transpose().reshape(-1, 3).astype(np.float32)

            verts_cam = np.matmul(points_proj, I.transpose())
            verts_cam = (verts_cam[:, :2] / verts_cam[:, 2]
                         [:, None]).astype(np.int32)

            valid_x = np.logical_and(
                verts_cam[:, 0] > 0, verts_cam[:, 0] < mask.shape[1])
            valid_y = np.logical_and(
                verts_cam[:, 1] > 0, verts_cam[:, 1] < mask.shape[0])
            valid = np.logical_and(valid_x, valid_y)
            verts_cam = verts_cam[valid]

            valid_indices = np.array(np.nonzero(valid)).squeeze()

            valid_mask = mask[verts_cam[:, 1], verts_cam[:, 0]] > 0
            verts_cam = verts_cam[valid_mask]
            points_valid = points[valid][valid_mask]

            valid_indices = valid_indices[valid_mask]

            first_valid = np.in1d(indices[:, 0], valid_indices)
            second_valid = np.in1d(indices[:, 1], valid_indices)
            third_valid = np.in1d(indices[:, 2], valid_indices)
            indices_valid = np.logical_and(
                first_valid, np.logical_and(second_valid, third_valid))

            mesh.triangles = o3d.utility.Vector3iVector(
                indices[indices_valid])

        return mesh
