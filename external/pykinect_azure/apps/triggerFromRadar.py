import cv2
import argparse
import os
import queue
import threading
import sys
import functools
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime
import signal
import pykinect_azure as pykinect
from timerGui import *


from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer, QDateTime

# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# User configuration options
do_timestamp_test = True
show_img = False
write_img = True
show_timestamp_plot = False

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
# change to _MASTER for external sync
device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE

parser = argparse.ArgumentParser(
    prog='triggerFromRadar',
    description='Triggers the Azure Kinect from a Radar Signal')
parser.add_argument("-d", "--directory", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"),
                    help='Directory where the output video file is stored')
parser.add_argument("-f", "--file", type=str, default="output",
                    help='Name of the output. In the video mode a file with an .mkv ending is generated. In frame mode, a directory is generated.')
parser.add_argument("-m", "--mode", type=int, default=1,
                    help='Recording mode, 0 = video mode, 1 = image mode, 2 = single frame on repeat')
parser.add_argument("-c", "--count", type=int, default=-1,
                    help="How many frames should be recorded? Default=-1 (infinite)")
parser.add_argument("-b", "--body", type=bool,
                    default=False, help="Enable body tracking")

args = parser.parse_args()

frame_queue = queue.Queue()
device = None
loop = True
timestamps = []
system_timestamps = []
bodyTracker = None
thread_initialized = False
last_timestamp = time.time()


def sigint_handler(signal, frame):
    global loop
    print('Interrupted')

    if loop:
        loop = False
        device.stop_cameras()
        device.stop_imu()
    else:
        exit(0)


signal.signal(signal.SIGINT, sigint_handler)


# Write the images asynchronosuly because otherwise the maximum performance of the
# Kinect would be drastically lower because writing takes very long
def write_image_asynchronously(frame_queue: queue.Queue):
    global thread_initialized
    thread_initialized = True

    while True:
        # Get an element from the queue.

        rgb, depth, id, timestamp = frame_queue.get()
        # print("time for block: ", time.time()-blocked)

        # filename = str(id).zfill(6) + ".png"
        if do_timestamp_test:
            filename = timestamp.strftime("%H_%M_%S_%f") + ".png"
        else:
            filename = str(id).zfill(6) + ".png"
        # Plot the image

        if write_img:
            cv2.imwrite(os.path.join(args.directory,
                        args.file, "rgb", filename), rgb)
            cv2.imwrite(os.path.join(args.directory,
                        args.file, "depth", filename), depth)
        # print("time for write: ", time.time()-now)
        # let the queue know we are finished with this element so the main thread can figure out
        # when the queue is finished completely
        frame_queue.task_done()

        if not loop:
            print(
                f"Waiting for the file writing process to finish...  {frame_queue.qsize()} frames left")


def initialize_device():
    global device
    global bodyTracker

    # Start device
    if (args.mode == 0):
        video_filename = os.path.join(args.directory, args.file + ".mkv")
        device = pykinect.start_device(
            config=device_config, record=True, record_filepath=video_filename)
    else:
        device = pykinect.start_device(config=device_config)

    if args.body:
        # Start body tracker
        bodyTracker = pykinect.start_body_tracker()


def receive_and_process(frame_counter):
    global device
    global bodyTracker
    global last_timestamp

    # Get capture
    capture = device.update()

    # Get the color depth image from the capture
    ret_d, depth_image = capture.get_colored_depth_image()
    ret_c, color_image = capture.get_color_image()

    # if no content received, repeat the loop without
    # displaying or storing images
    if not ret_d or not ret_c:
        return False

    depth_object = capture.get_depth_image_object()
    color_object = capture.get_depth_image_object()

    # print("depth in bytes: ", sys.getsizeof(depth_image))

    if args.body:
        # Get body tracker frame
        body_frame = bodyTracker.update()

    # log depth timestamps

    system_time_datetime = get_system_time()
    print("[NTP] system time: {}".format(system_time_datetime))
    last_timestamp = system_time_datetime

    timestamps.append(depth_object.get_device_timestamp_usec())  # microsec
    # millisec -> microsec
    system_timestamps.append(system_time_datetime.timestamp() * 1e3)

    if (args.mode == 1 or args.mode == 2):
        frame_queue.put((color_image, depth_image,
                        frame_counter, system_time_datetime))

    if show_img:
        cv2.imshow('Depth Image', depth_image)
        cv2.imshow('Color Image', color_image)

    return True


def loop_func():
    global loop
    global device

    frame_counter = 0
    while loop:
        start = time.perf_counter()

        ret = receive_and_process(frame_counter)
        if not ret:
            continue

        frame_counter = frame_counter + 1

        # measure the recording time
        end = time.perf_counter()
        # print(f"{(1.0 / (end-start)):0.1f} FPS")

        # Press q key to stop
        if cv2.waitKey(1) == ord('q') or args.mode == 2 or frame_counter == args.count:
            loop = False
            device.stop_cameras()
            device.stop_imu()


if __name__ == "__main__":
    assert os.path.exists(args.directory)

    if show_img:
        cv2.namedWindow('Depth Image', cv2.WINDOW_AUTOSIZE |
                        cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('Color Image', cv2.WINDOW_AUTOSIZE |
                        cv2.WINDOW_KEEPRATIO)

    if (args.mode > 0):
        if (not os.path.exists(os.path.join(args.directory, args.file))):
            os.mkdir(os.path.join(args.directory, args.file))
            os.mkdir(os.path.join(args.directory, args.file, "rgb"))
            os.mkdir(os.path.join(args.directory, args.file, "depth"))

    if (args.mode == 1 or args.mode == 2):
        # Start a thread that runs write_image_asynchronously(frame_queue). Marking it as daemon allows the python
        # program to exit even though that thread is still running. The thread will then be stopped
        # when the program exits
        threading.Thread(target=functools.partial(
            write_image_asynchronously, frame_queue), daemon=True).start()
        # threading.Thread(target=functools.partial(start_timer_window), daemon=True).start()

    while (not thread_initialized):
        print("Waiting for Thread to be initialized...")
        time.sleep(1)

    initialize_device()
    while True:

        loop_func()

        if (args.mode == 1 or args.mode == 2 or not loop):
            # Wait for the frames to finish
            frame_queue.join()

        if (args.mode != 2):
            break
        # reinitialization does not work, maybe try:
        # https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1474
        device.start_cameras(device_config)

    cv2.destroyAllWindows()

    if show_timestamp_plot:
        # plot relative timestamps
        T = system_timestamps
        x = np.arange(0, len(T)-1)
        # print("timestamps: ", len(timestamps))
        y = []
        for i, k in enumerate(T[1:], 1):
            y.append((k - T[i-1]))
        # print("timestamps: ", timestamps)
        # print("y: ", y)

        plt.plot(x, y)
        plt.show()
