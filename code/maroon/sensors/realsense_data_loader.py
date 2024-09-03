

import os

import numpy as np
import cv2
import open3d as o3d
import json
import datetime
import pyrealsense2 as rs
import shutil


class RealsenseDataLoader():
    def __init__(self, realsense_data_path: str):
        self.data_path = realsense_data_path

#   def process_bag_tmp(self, bag_path: str):
#         assert os.path.exists(bag_path)
#         if not os.path.exists(self.data_path):
#             os.makedirs(self.data_path)

#         print("Loading from: ", bag_path)
#        # Create pipeline
#         pipeline = rs.pipeline()
#         # Create a config object
#         config = rs.config()
#         # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
#         rs.config.enable_device_from_file(
#             config, bag_path, repeat_playback=False)

#         if not os.path.exists(os.path.join(self.data_path, "rgb")):
#             os.makedirs(os.path.join(self.data_path, "rgb"))

#         if not os.path.exists(os.path.join(self.data_path, "depth")):
#             os.makedirs(os.path.join(self.data_path, "depth"))

#         ctx = rs.context()
#         playback = ctx.load_device(bag_path)
#         devices = ctx.query_devices()
#         print("devices: ", devices)
#         playback.set_real_time(True)
#         sensors = playback.query_sensors()
#         print("sensors: ", sensors)
#         depth_sensor = playback.first_depth_sensor()
#         depth_scale = depth_sensor.get_depth_scale()
#         depth_profile = depth_sensor.get_stream_profiles()


#         frame_queue = rs.frame_queue(capacity=10, keep_frames=True)
#         progress = 0
#         def callback_func(frame:rs.frame):
#             align = rs.align(rs.stream.color)
#             composite_frame = rs.composite_frame(frame)
#             print("composite_frame: ", composite_frame)
#             aligned_frames = align.process(composite_frame)
#             # Get depth frame
#             depth_frame = aligned_frames.get_depth_frame()
#             color_frame = aligned_frames.get_color_frame()
#             print("depth_frame: ", depth_frame)
#             print("color_frame: ", color_frame)
#             frame_number = frame.get_frame_number()

#             print("got_frame: !", frame_number)
#             # progress = progress + 1


#         for s in sensors:
#             print("stream_profiles: ", s.get_stream_profiles())
#             if len(s.get_stream_profiles()) == 0:
#                 continue
#             if s.is_depth_sensor():
#                 s.open(s.get_stream_profiles())
#                 print("start sensor")
#                 s.start(queue=frame_queue)
#                 #s.start(callback_func)

#         duration = playback.get_duration()
#         progress = 0
#         posCurr = playback.get_position()
#         while True:
#             posNext = playback.get_position()
#             progress_perc = (posNext / (duration.total_seconds() * 1e9)) * 100.0
#             # print("posNext: {} curr: {}, perc: {} duration: {}".format(posNext, posCurr, progress_perc, duration.total_seconds()))
#             if posNext < posCurr and posCurr > 0:
#                 break
#             posCurr = posNext

#         for s in sensors:
#             if len(s.get_stream_profiles()) == 0:
#                 continue
#             s.stop()
#             s.close()

#         print("left: ", frame_queue.capacity())
#         for i in range(0, frame_queue.capacity()):
#             frame = frame_queue.poll_for_frame()
#             progress = progress + 1
#             print("frame: ", frame)
#             callback_func(frame)
#             print("got_frame: ", progress)

#         print("starting...")
#         import time
#         time.sleep(10)
#         print("queue: ", len(frame_queue))
#         exit(0)

    def process_bag(self, bag_path: str):
        assert os.path.exists(bag_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        print("Loading from: ", bag_path)
       # Create pipeline
        pipeline = rs.pipeline()
        # Create a config object
        config = rs.config()
        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(
            config, bag_path, repeat_playback=True)
        # # Configure the pipeline to stream the depth stream
        # # Change this parameters according to the recorded bag file resolution
        # config.enable_stream(rs.stream.depth, rs.format.any, framerate=30)
        # config.enable_stream(rs.stream.color, rs.format.any, framerate=30)

        if not os.path.exists(os.path.join(self.data_path, "rgb")):
            os.makedirs(os.path.join(self.data_path, "rgb"))

        if not os.path.exists(os.path.join(self.data_path, "depth")):
            os.makedirs(os.path.join(self.data_path, "depth"))

        # align_to = rs.stream.color
        # align = rs.align(align_to)

        # def callback(frame):
        #     print("Got frame")
        #     nonlocal align
        #     # nonlocal counter, flags
        #     # if counter > number_of_images:
        #     #     return
        #     frames = frame.as_frameset()
        #     aligned_frames = align.process(frames)
        #     # Get depth frame
        #     depth_frame = aligned_frames.get_depth_frame()
        #     color_frame = aligned_frames.get_color_frame()
        #     timestamp = depth_frame.get_timestamp()
        #     print("Timestamp: {}".format(timestamp))

        #     del frame

        #     # for f in frame.as_frameset():
        #     #     p = f.get_profile()
        #     #     print("f: ", f)
        #     #     print("p: ",p )

        # Start streaming from file
        queue = rs.frame_queue(50, keep_frames=True)
        profile = pipeline.start(config)
        # Needed so frames don't get dropped during processing:
        playback = profile.get_device().as_playback()

        # playback.pause()

        depth_profile = rs.video_stream_profile(
            profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        depth_fps = depth_profile.fps()

        color_profile = rs.video_stream_profile(
            profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        color_fps = color_profile.fps()

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        align_to = rs.stream.color
        align = rs.align(align_to)

        pc = rs.pointcloud()

        json_data = {}
        if os.path.exists(os.path.join(self.data_path, "calibration.json")):
            with open(os.path.join(self.data_path, "calibration.json")) as f:
                json_data = json.load(f)
        json_data["capture"] = {
            "scale": depth_scale,
        }
        json_data["rgb"] = {
            "fx": color_intrinsics.fx,
            "fy": color_intrinsics.fy,
            "cx": color_intrinsics.ppx,
            "cy": color_intrinsics.ppy,
            "width": color_intrinsics.width,
            "height": color_intrinsics.height,
            "fps": color_fps,
        }
        json_data["depth"] = {
            "fx": depth_intrinsics.fx,
            "fy": depth_intrinsics.fy,
            "cx": depth_intrinsics.ppx,
            "cy": depth_intrinsics.ppy,
            "width": depth_intrinsics.width,
            "height": depth_intrinsics.height,
            "fps": depth_fps,
        }
        with open(os.path.join(self.data_path, "calibration.json"), "w") as f:
            json.dump(json_data, f)

        def get_timestamp(frame_no, frames, typestamp_type):
            filename = str(frame_no).zfill(6)
            timestamp = frames.get_timestamp()

            timestamp_ms = timestamp
            timestamp_ns = timestamp_ms * 1e6
            time_domain = frames.get_frame_timestamp_domain()
            if time_domain == rs.timestamp_domain.global_time:
                typestamp_type = "absolute_global[ns]"
            elif time_domain == rs.timestamp_domain.hardware_clock:
                typestamp_type = "relative_device[ns]"
            elif time_domain == rs.timestamp_domain.system_time:
                typestamp_type = "absolute_system[ns]"
            else:
                raise ValueError("Unknown time_domain")

            dt = datetime.datetime.fromtimestamp(timestamp_ns * 1e-9)
            # timestamps.append(int(timestamp_ns))
            return timestamp, timestamp_ns, typestamp_type

        def store_frame(frame_no, frames, align):
            filename = str(frame_no).zfill(6)

            if not os.path.exists(os.path.join(self.data_path, "rgb", filename+".jpg")):

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)
                # Get depth frame
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                points = pc.calculate(depth_frame)

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())[:, :, ::-1]

                cv2.imwrite(os.path.join(self.data_path, "rgb",
                            filename+".jpg"), color_image)
                cv2.imwrite(os.path.join(self.data_path, "depth",
                            filename+".png"), depth_image)
                return True
            return False

        timestamps = []
        updated = False
        typestamp_type = ""

        playback.set_real_time(True)
        playback.seek(datetime.timedelta(0))
        # Streaming loop
        duration = playback.get_duration()
        last_position = playback.get_position()

        saved_frames = []
        dropped_frames = []
        first_loop = True
        max_frame = 0
        min_frame = 1000

        max_iterations = 20
        iteration = 0
        stuck_tries = 0
        while ((len(dropped_frames) > 0) or first_loop or (len(saved_frames)+min_frame) < max_frame) and iteration < max_iterations:
            new_position = playback.get_position()
            progress_perc = (
                new_position / (duration.total_seconds() * 1e9)) * 100.0
            if not first_loop and new_position != last_position:
                print("New: {} last: {}, progress: {}".format(
                    new_position, last_position, progress_perc))
                stuck_tries = 0

            if new_position == last_position:
                stuck_tries = stuck_tries + 1
                # print("Stuck count: {}".format(stuck_tries))
            else:
                stuck_tries = 0

            if new_position < last_position or stuck_tries > 10000:
                if not new_position < last_position and stuck_tries > 10000:
                    # print("---------------- Reset to Zero")
                    playback.seek(datetime.timedelta(0))
                    # print("position: ", playback.get_position())
                else:
                    playback.set_real_time(False)
                stuck_tries = 0

                if first_loop:
                    for i in range(min_frame, max_frame+1):
                        dropped_frames.append(i)
                        timestamps.append(None)
                print("dropped frames: ", dropped_frames)
                print("Saved frames: ", len(saved_frames))
                print("Number frames: ", max_frame - min_frame)
                print("max_frame: ", max_frame)
                print("min_frame: ", min_frame)
                print("----------------------------------------------")
                first_loop = False
                iteration = iteration + 1
            last_position = new_position
            # pipeline.poll_for_frames()
            received, frames = pipeline.try_wait_for_frames(1)
            if frames is None or not received or not frames.is_frame():
                continue
            max_frame_old = max_frame
            max_frame = max(max_frame, frames.get_frame_number())
            min_frame = min(min_frame, frames.get_frame_number())
            if first_loop:
                continue
            else:
                if max_frame_old != max_frame:
                    print(
                        "UPDATED MAX NUMBER: {} -> {}".format(max_frame_old, max_frame))
                    for i in range(max_frame_old, max_frame):
                        dropped_frames.append(i)
                        timestamps.append(None)

            frame_no = frames.get_frame_number()
            if frame_no not in dropped_frames:
                continue
            if len(timestamps) > frame_no and timestamps[frame_no] != None:
                continue

            dropped_frames.remove(frame_no)
            updated_ = store_frame(frame_no, frames, align)
            if updated_:
                updated = updated_

            timestamp, timestamp_ns, typestamp_type = get_timestamp(
                frame_no, frames, typestamp_type)
            saved_frames.append(frame_no)
            timestamps[frame_no-min_frame] = timestamp_ns
            if updated_:
                processed_frames = sum(
                    [1 if t is not None else 0 for t in timestamps])
                if processed_frames % 10 == 0:
                    print("Processing ({}/?)".format(processed_frames))

        pipeline.stop()

        if updated:
            with open(os.path.join(self.data_path, "timestamps.txt"), "w") as f:
                f.write(f"filename {typestamp_type}\n")
                for i, t in enumerate(timestamps):
                    filename = str(i).zfill(6)
                    f.write("{} {}\n".format(filename, str(t)))

    def get_timestamps(self):
        timestamp_file = os.path.join(self.data_path, "timestamps.txt")
        if not os.path.exists(timestamp_file):
            for f in os.listdir(os.path.dirname(self.data_path)):
                if f.endswith(".bag"):
                    self.process_bag(os.path.join(
                        os.path.dirname(self.data_path), f))
        assert os.path.exists(timestamp_file)

        system_absolute_timestamps = []
        with open(timestamp_file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                tokens = line.split(" ")
                system_absolute_timestamps.append(np.uint64(tokens[1]) * 1e-9)

        return system_absolute_timestamps

    def get_synced_timestamps(self, calibration_path=""):
        timestamp_file = os.path.join(self.data_path, "timestamps_synced.txt")

        if len(calibration_path) > 0:
            calibration_file = os.path.join(
                calibration_path, "temporal_alignment.json")
        else:
            calibration_file = ""

        if not os.path.exists(timestamp_file) and not os.path.exists(calibration_file):
            return []

        if os.path.exists(timestamp_file):
            absolute_timestamps = []
            with open(timestamp_file, 'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    tokens = line.split(" ")
                    absolute_timestamps.append(np.uint64(tokens[1]) * 1e-9)
        else:
            with open(calibration_file) as f:
                calib_data = json.load(f)
            if not "zed2radar" in calib_data:
                return []
            unsynced_timestamps = self.get_timestamps()
            absolute_timestamps = []
            for t in unsynced_timestamps:
                absolute_timestamps.append(
                    t + calib_data["realsense2radar"]["shift_seconds"])
        return absolute_timestamps

    def load_calibration(self):
        assert os.path.exists(os.path.join(
            self.data_path, "calibration.json")), "Path does not exist: {}".format(os.path.join(
                self.data_path, "calibration.json"))

        calib = {}
        with open(os.path.join(self.data_path, "calibration.json")) as f:
            calib = json.load(f)

        return calib

    def num_frames(self):
        if (not os.path.exists(os.path.join(self.data_path, "rgb"))
                or not os.path.exists(os.path.join(self.data_path, "depth"))):
            for f in os.listdir(os.path.dirname(self.data_path)):
                if f.endswith(".bag"):
                    self.process_bag(os.path.join(
                        os.path.dirname(self.data_path), f))

        num_frames = 0
        for frame in os.listdir(os.path.join(self.data_path, "rgb")):
            if ".jpg" in frame:
                num_frames = num_frames + 1
        return num_frames

    def get_intrinsics(self):
        calib = self.load_calibration()

        K_l = np.eye(4)
        K_l[0, 0] = calib["rgb"]["fx"]
        K_l[1, 1] = calib["rgb"]["fy"]
        K_l[0, 2] = calib["rgb"]["cx"]
        K_l[1, 2] = calib["rgb"]["cy"]
        return K_l

    def get_mask(self, frame_index=0):
        if frame_index == -1:
            frame_index = self.num_frames() - 1

        mask_dir = os.path.join(self.data_path, "mask")
        filename = str(frame_index).zfill(6)
        mask_file = os.path.join(mask_dir, filename + ".png")
        mask = None

        if not os.path.exists(mask_file):
            print("Did not find mask: {}".format(mask_file))
            counter = frame_index + 1
            while counter >= 0:
                filename_ = str(counter).zfill(6)
                mask_file_ = os.path.join(mask_dir, filename_ + ".png")
                if os.path.exists(mask_file_):
                    shutil.copyfile(mask_file_, mask_file)
                    print("Copy mask: {}->{}".format(mask_file_, mask_file))
                    break
                counter = counter - 1

        if os.path.exists(mask_file):
            mask = (cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                    > 0).astype(np.uint8)

        return mask

    def get_frame(self, frame_index=0, use_mask=False, mask_idx=None, averaging_factor=1, manual_mask=None):
        if frame_index == -1:
            frame_index = self.num_frames() - 1
        average = averaging_factor > 1
        averaging_factor = min(averaging_factor, (self.num_frames()))

        if (not os.path.exists(os.path.join(self.data_path, "rgb"))
                or not os.path.exists(os.path.join(self.data_path, "depth"))):
            for f in os.listdir(os.path.dirname(self.data_path)):
                if f.endswith(".bag"):
                    self.process_bag(os.path.join(
                        os.path.dirname(self.data_path), f))

        rgb_dir = os.path.join(self.data_path, "rgb")
        depth_dir = os.path.join(self.data_path, "depth")
        mask_dir = os.path.join(self.data_path, "mask")
        filename = str(frame_index).zfill(6)

        assert os.path.exists(rgb_dir)
        assert os.path.exists(os.path.join(
            rgb_dir, filename + ".jpg")), "Path does not exist: {}".format(os.path.join(
                rgb_dir, filename + ".jpg"))

        rgb = cv2.imread(os.path.join(
            rgb_dir, filename + ".jpg"), cv2.IMREAD_UNCHANGED)
        if not average:
            depth = cv2.imread(os.path.join(
                depth_dir, filename + ".png"), cv2.IMREAD_UNCHANGED)
        else:
            num_frames = (self.num_frames())
            avg_depth = None
            num_avg = 0
            if frame_index == num_frames-1:
                r = np.arange(frame_index - averaging_factor +
                              2, frame_index+2, 1)
            else:
                r = np.arange(-averaging_factor//2, averaging_factor//2, 1)

            for i in r:
                index = frame_index + i
                if (index < 0):
                    index = num_frames + index
                if (index >= num_frames):
                    index = (index - num_frames)
                filename_avg = str(index).zfill(6)
                if not os.path.exists(os.path.join(
                    depth_dir, filename_avg + ".png")):
                    continue

                print("Averaging index: ", index)

                depth = cv2.imread(os.path.join(
                    depth_dir, filename_avg + ".png"), cv2.IMREAD_UNCHANGED)
                if depth is None:
                    continue
                if num_avg == 0:
                    pixel_cnts = np.zeros_like(depth)
                    avg_depth = depth.astype(np.float32)
                    pixel_cnts[avg_depth > 0] = 1
                else:
                    pixel_cnts[depth > 0] += 1                   
                    avg_depth[depth > 0] = depth.astype(np.float32)[depth > 0] + avg_depth[depth > 0]

                num_avg = num_avg + 1
            depth = np.zeros_like(avg_depth)
            depth[pixel_cnts > 0] = (avg_depth[pixel_cnts > 0] / pixel_cnts[pixel_cnts > 0]).astype(np.uint16)

        mask_filename = filename 
        if mask_idx is not None:
            if mask_idx == -1:
                mask_idx = self.num_frames() -1
            mask_filename = str(mask_idx).zfill(6)

        mask_file = os.path.join(mask_dir, mask_filename + ".png")
        mask = None
        if use_mask:
            if not os.path.exists(mask_file):
                print("Did not find mask: {}".format(mask_file))
                counter = frame_index + 1
                while counter >= 0:
                    filename_ = str(counter).zfill(6)
                    mask_file_ = os.path.join(mask_dir, filename_ + ".png")
                    if os.path.exists(mask_file_):
                        shutil.copyfile(mask_file_, mask_file)
                        print("Copy mask: {}->{}".format(mask_file_, mask_file))
                        break
                    counter = counter - 1

            if os.path.exists(mask_file):
                mask = (cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                        > 0).astype(np.uint8)

        if use_mask and manual_mask is not None:
            mask = manual_mask

        if mask is not None:
            depth = depth * mask

        calib = self.load_calibration()

        K_l = np.eye(3)
        K_l[0, 0] = calib["rgb"]["fx"]
        K_l[1, 1] = calib["rgb"]["fy"]
        K_l[0, 2] = calib["rgb"]["cx"]
        K_l[1, 2] = calib["rgb"]["cy"]

        pixel_x, pixel_y = np.meshgrid(np.linspace(
            0, depth.shape[1]-1, depth.shape[1]), np.linspace(0, depth.shape[0]-1, depth.shape[0]))
        pixels = np.stack([pixel_x * depth, pixel_y * depth, depth], axis=-1)
        xyz = np.matmul(pixels, np.linalg.inv(K_l).transpose())

        def compute_pixel_normals(xyz):
            wpos_lin_x = xyz.reshape(-1, 3)
            wpos_lin_y = xyz.transpose(1, 0, 2).reshape(-1, 3)
            normal_x = []
            for i in range(0, 3):
                normal_x.append(np.convolve(
                    wpos_lin_x[:, i], np.array([-1, 0, 1]), mode='same'))
            normal_x = np.stack(
                normal_x, axis=-1).reshape(xyz.shape[0], xyz.shape[1], 3)
            normal_y = []
            for i in range(0, 3):
                normal_y.append(np.convolve(
                    wpos_lin_y[:, i], np.array([1, 0, -1]), mode='same'))
            normal_y = np.stack(
                normal_y, axis=-1).reshape(xyz.shape[1], xyz.shape[0], 3).transpose(1, 0, 2)
            normal = np.cross(normal_x, normal_y)
            normal_length = np.linalg.norm(normal, axis=-1)
            normal[normal_length > 0] = normal[normal_length > 0] / \
                normal_length[normal_length > 0][:, None]
            normal = normal * np.array([1, 1, 1])[None, :]

            return normal

        normal = compute_pixel_normals(xyz)

        return rgb, depth, xyz, normal
