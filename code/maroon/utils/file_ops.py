
import xml.etree.ElementTree as ET
import math
import numpy as np
import cv2
import os
import open3d as o3d


def load_rgb(directory, image_name_list=[]):
    imgs = []
    paths = []
    if (len(image_name_list) > 0):
        for image_name in image_name_list:
            rgb_path = os.path.join(directory, image_name+".jpg")
            rgb_path2 = os.path.join(directory, image_name+".JPG")
            assert os.path.exists(
                rgb_path) or os.path.exists(
                rgb_path2), "Could not find depth image at: {}".format(rgb_path)
            if os.path.exists(rgb_path2):
                rgb_path = rgb_path2
            rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
            paths.append(rgb_path)
            imgs.append(rgb_image)
    else:
        for file in sorted(os.listdir(directory)):
            rgb_path = os.path.join(directory, file)
            assert os.path.exists(
                rgb_path), "Could not find depth image at: {}".format(rgb_path)
            rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
            paths.append(rgb_path)
            imgs.append(rgb_image)
    return imgs, paths


def load_depth(directory, image_name_list=[], extension=".png"):
    imgs = []
    if (len(image_name_list) > 0):
        for image_name in image_name_list:
            depth_path = os.path.join(directory, image_name+extension)
            assert os.path.exists(
                depth_path), "Could not find depth image at: {}".format(depth_path)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if (len(depth_image.shape) == 3):
                depth_image = depth_image[:, :, 0]
            imgs.append(depth_image)
    else:
        for file in sorted(os.listdir(directory)):
            depth_path = os.path.join(directory, file)
            assert os.path.exists(
                depth_path), "Could not find depth image at: {}".format(depth_path)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if (len(depth_image.shape) == 3):
                depth_image = depth_image[:, :, 0]
            imgs.append(depth_image)
    return imgs


def load_masks(directory, image_name_list=[], extension=".png"):
    imgs = []
    successful_loaded = 0
    if (len(image_name_list) > 0):
        for image_name in image_name_list:
            mask_path = os.path.join(directory, image_name+extension)
            if (not os.path.exists(mask_path)):
                print("Path does not exist: ", mask_path)
                imgs.append(None)
                continue
            sphere_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if len(sphere_mask.shape) > 2:
                sphere_mask = sphere_mask[:, :, 0]
            imgs.append(sphere_mask)
            successful_loaded = successful_loaded + 1
    else:
        for file in sorted(os.listdir(directory)):
            mask_path = os.path.join(directory, file)
            if (not os.path.exists(mask_path)):
                print("Path does not exist: ", mask_path)
                imgs.append(None)
                continue
            sphere_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if len(sphere_mask.shape) > 2:
                sphere_mask = sphere_mask[:, :, 0]
            imgs.append(sphere_mask)
            successful_loaded = successful_loaded + 1
    return imgs, successful_loaded


def load_calibration(xml_path, image_name):
    # Camera Settings
    camera_root = ET.parse(xml_path).getroot()

    # Load Camera Calibration
    sensors = camera_root.findall("chunk")[0].findall("sensors")[
        0].findall("sensor")
    cameras = camera_root.findall("chunk")[0].findall("cameras")[
        0].findall("camera")

    if (len(sensors) < len(cameras)):
        print("Warning: Detected multiple cameras of the same sensor type (same model + camera intrinsics). This means those cameras can not be automatically distinguished from each other.")

    calibration = None
    camera = None
    sensor_id = None
    for i in cameras:
        if i.get("label") == str(image_name):
            sensor_id = i.get("sensor_id")
            camera = i
            break
    for i in sensors:
        if i.get("id") == str(sensor_id):
            calibration = i.find("calibration")
            break
    if calibration == None:
        raise Exception("No calibration found!" + str(image_name))

    # Intrinsics
    f = float(calibration.find("f").text)
    w = int(calibration.find("resolution").get("width"))
    h = int(calibration.find("resolution").get("height"))
    # cx and cy are only offsets for the picture center
    cx = w / 2.0 + float(calibration.find("cx").text)
    cy = h / 2.0 + float(calibration.find("cy").text)
    intrinsics = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])  # 3x3
    # intrinsics_inv = np.linalg.inv(intrinsics)

    # Distortion (not always saved into the cams.xml (only after cameras are optimized))
    # k1 = float(calibration.find("k1").text)
    # k2 = float(calibration.find("k2").text)
    # k3 = float(calibration.find("k3").text)
    # k4 = float(calibration.find("k4").text)
    # p1 = float(calibration.find("p1").text)
    # p2 = float(calibration.find("p2").text)

    # OpenCV requires k5 and k6
    # distortion = np.array([k1, k2, p1, p2, k3, k4, 0, 0])
    distortion = None

    # World Transform
    # cameras = camera_root.findall("chunk")[0].findall("cameras")[0].findall("camera")
    # camera = None
    # for i in cameras:
    #     if i.get("id") == str(depth_image_index) and i.get("label") == image_name:
    #         camera = i
    #         break
    if camera == None:
        raise Exception("No camera found!")

    transform = np.reshape(np.array(camera.find(
        "transform").text.split(' ')), (-1, 4)).astype(float)  # 4x4
    # To get world coordinate you need chunk rot | chunk trans
    # and then chunk.transform * cam.transform
    if not camera_root.findall("chunk")[0].find("transform") == None:
        chunk_rotation = np.reshape(np.array(camera_root.findall("chunk")[0].find(
            "transform").find("rotation").text.split(' ')), (-1, 3)).astype(float)  # 3x3
        chunk_translation = np.array(camera_root.findall("chunk")[0].find(
            "transform").find("translation").text.split(' ')).astype(float)  # 1x3
        chunk_transform = np.identity(4)
        chunk_transform[0:3, 0:3] = chunk_rotation
        chunk_transform[0:3, 3] = chunk_translation
        transform = np.matmul(chunk_transform, transform)

    # do not invert as it is already inverted
    # transform = np.linalg.inv(transform)
    R = transform[0:3, 0:3]  # Rotation    3x3
    T = transform[0:3, 3]   # Translation 1x3

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = T

    return intrinsics, extrinsic, distortion
