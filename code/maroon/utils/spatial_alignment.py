
import numpy as np


def transform_world(points, src, dest, extrinsics: dict, normals=None, use_world_transform=True):
    if src == dest:
        if normals is None:
            return points
        else:
            return points, normals

    T = np.eye(4)

    if "{}2{}".format(src, dest) in extrinsics:
        print("found: ", "{}2{}".format(src, dest))
        T = np.array(extrinsics["{}2{}".format(src, dest)]).reshape(4, 4)
    else:
        if src != "radar":
            T = np.array(extrinsics["{}2radar".format(src)]).reshape(4, 4)

        if src != "radar" and dest != "radar":
            dest_to_radar = np.array(
                extrinsics["{}2radar".format(dest)]).reshape(4, 4)
            T = np.matmul(np.linalg.inv(dest_to_radar), T)

        if src == "radar" and dest != "radar":
            dest_to_radar = np.array(
                extrinsics["{}2radar".format(dest)]).reshape(4, 4)
            T = np.linalg.inv(dest_to_radar)

    if not use_world_transform:
        src_to_world = np.array(
            extrinsics["{}2world".format(src)]).reshape(4, 4)
        T = np.matmul(T, np.linalg.inv(src_to_world))

    ones = np.ones((points.shape[0], 1))
    transformed_points = np.matmul(np.concatenate(
        [points, ones], axis=1), T.transpose())[:, :3]

    if normals is None:
        return transformed_points
    else:
        N = np.transpose(np.linalg.inv(T))
        ones = np.ones((normals.shape[0], 1))
        transformed_normals = np.matmul(np.concatenate(
            [normals, ones], axis=1), N.transpose())[:, :3]
        length = np.linalg.norm(transformed_normals, axis=-1)
        transformed_normals[length >
                            0] = transformed_normals[length > 0] / length[length > 0].reshape(-1, 1)

        return transformed_points, transformed_normals


def to_world(points, src, extrinsics: dict, normals=None):
    T = np.eye(4)

    src_to_world = np.array(
        extrinsics["{}2world".format(src)]).reshape(4, 4)
    T = np.matmul(T, (src_to_world))

    ones = np.ones((points.shape[0], 1))
    transformed_points = np.matmul(np.concatenate(
        [points, ones], axis=1), T.transpose())[:, :3]

    if normals is None:
        return transformed_points
    else:
        N = np.transpose(np.linalg.inv(T))
        ones = np.ones((normals.shape[0], 1))
        transformed_normals = np.matmul(np.concatenate(
            [normals, ones], axis=1), N.transpose())[:, :3]
        length = np.linalg.norm(transformed_normals, axis=-1)
        transformed_normals[length >
                            0] = transformed_normals[length > 0] / length[length > 0].reshape(-1, 1)

        return transformed_points, transformed_normals


def transform(points, src, dest, extrinsics: dict, normals=None, use_world_transform=True):
    if src == dest:
        if normals is None:
            return points
        else:
            return points, normals

    T = np.eye(4)

    if "{}2{}".format(src, dest) in extrinsics:
        print("found: ", "{}2{}".format(src, dest))
        T = np.array(extrinsics["{}2{}".format(src, dest)]).reshape(4, 4)
    else:
        if src != "radar":
            T = np.array(extrinsics["{}2radar".format(src)]).reshape(4, 4)

        if src != "radar" and dest != "radar":
            dest_to_radar = np.array(
                extrinsics["{}2radar".format(dest)]).reshape(4, 4)
            T = np.matmul(np.linalg.inv(dest_to_radar), T)

        if src == "radar" and dest != "radar":
            dest_to_radar = np.array(
                extrinsics["{}2radar".format(dest)]).reshape(4, 4)
            T = np.linalg.inv(dest_to_radar)

    if not use_world_transform:
        src_to_world = np.array(
            extrinsics["{}2world".format(src)]).reshape(4, 4)
        T = np.matmul(T, np.linalg.inv(src_to_world))

    ones = np.ones((points.shape[0], 1))
    transformed_points = np.matmul(np.concatenate(
        [points, ones], axis=1), T.transpose())[:, :3]

    if normals is None:
        return transformed_points
    else:
        N = np.transpose(np.linalg.inv(T))
        ones = np.ones((normals.shape[0], 1))
        transformed_normals = np.matmul(np.concatenate(
            [normals, ones], axis=1), N.transpose())[:, :3]
        length = np.linalg.norm(transformed_normals, axis=-1)
        transformed_normals[length >
                            0] = transformed_normals[length > 0] / length[length > 0].reshape(-1, 1)

        return transformed_points, transformed_normals
