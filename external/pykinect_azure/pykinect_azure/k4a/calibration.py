import ctypes

from pykinect_azure.k4a import _k4a
import os
import codecs


class Calibration:

    def __init__(self, calibration_handle: _k4a.k4a_calibration_t):

        self._handle = calibration_handle
        self.color_calibration = self._handle.color_camera_calibration
        self.depth_calibration = self._handle.depth_camera_calibration
        self.color_params = self.color_calibration.intrinsics.parameters.param
        self.depth_params = self.depth_calibration.intrinsics.parameters.param
        # maps from color to depth
        self.extrinsic = self._handle.extrinsics[_k4a.K4A_CALIBRATION_TYPE_COLOR][_k4a.K4A_CALIBRATION_TYPE_DEPTH]
        self.extrinsic_depth2color = self._handle.extrinsics[_k4a.K4A_CALIBRATION_TYPE_DEPTH][_k4a.K4A_CALIBRATION_TYPE_COLOR]

    def __del__(self):

        self.reset()

    def __str__(self):

        message = (
            "Rgb Intrinsic parameters: \n"
            f"\tcx: {self.color_params.cx}\n"
            f"\tcy: {self.color_params.cy}\n"
            f"\tfx: {self.color_params.fx}\n"
            f"\tfy: {self.color_params.fy}\n"
            f"\tk1: {self.color_params.k1}\n"
            f"\tk2: {self.color_params.k2}\n"
            f"\tk3: {self.color_params.k3}\n"
            f"\tk4: {self.color_params.k4}\n"
            f"\tk5: {self.color_params.k5}\n"
            f"\tk6: {self.color_params.k6}\n"
            f"\tcodx: {self.color_params.codx}\n"
            f"\tcody: {self.color_params.cody}\n"
            f"\tp2: {self.color_params.p2}\n"
            f"\tp1: {self.color_params.p1}\n"
            f"\tmetric_radius: {self.color_params.metric_radius}\n"
            "Depth Intrinsic parameters: \n"
            f"\tcx: {self.depth_params.cx}\n"
            f"\tcy: {self.depth_params.cy}\n"
            f"\tfx: {self.depth_params.fx}\n"
            f"\tfy: {self.depth_params.fy}\n"
            f"\tk1: {self.depth_params.k1}\n"
            f"\tk2: {self.depth_params.k2}\n"
            f"\tk3: {self.depth_params.k3}\n"
            f"\tk4: {self.depth_params.k4}\n"
            f"\tk5: {self.depth_params.k5}\n"
            f"\tk6: {self.depth_params.k6}\n"
            f"\tcodx: {self.depth_params.codx}\n"
            f"\tcody: {self.depth_params.cody}\n"
            f"\tp2: {self.depth_params.p2}\n"
            f"\tp1: {self.depth_params.p1}\n"
            f"\tmetric_radius: {self.depth_params.metric_radius}\n"
        )
        return message

    def get_extrinsics(self, fr, to):
        
        if fr == "color" and to == "depth":
            K = self.extrinsic
        else:
            K = self.extrinsic_depth2color
        R = K.rotation
        t = K.translation
            
        return [[R[0], R[1], R[2], t[0]], 
                [R[3], R[4], R[5], t[1]],
                [R[6], R[7], R[8], t[2]],
                [0, 0, 0, 1]]
        
        
    def get_matrix(self, camera: _k4a.k4a_calibration_type_t):
        if camera == _k4a.K4A_CALIBRATION_TYPE_COLOR:
            return [[self.color_params.fx, 0, self.color_params.cx],
                    [0, self.color_params.fy, self.color_params.cy],
                    [0, 0, 1]]
        elif camera == _k4a.K4A_CALIBRATION_TYPE_DEPTH:
            return [[self.depth_params.fx, 0, self.depth_params.cx],
                    [0, self.depth_params.fy, self.depth_params.cy],
                    [0, 0, 1]]

    def is_valid(self):
        return self._handle

    def handle(self):
        return self._handle

    def reset(self):
        if self.is_valid():
            self._handle = None

    def from_file(file_path, depth_mode=_k4a.K4A_DEPTH_MODE_NFOV_UNBINNED, resolution=_k4a.K4A_COLOR_RESOLUTION_1080P):
        assert os.path.exists(file_path)

        with codecs.open(file_path, encoding='utf-8') as f:
            calib_data = f.read()
        calib_data = calib_data.encode('ascii')

        calib_handle = _k4a.k4a_calibration_t()

        _k4a.k4a_calibration_get_from_raw(
            calib_data, len(calib_data)+1, depth_mode, resolution, calib_handle)
        calib = Calibration(calib_handle)
        return calib

    def convert_3d_to_3d(self, source_point3d: _k4a.k4a_float3_t(),
                         source_camera: _k4a.k4a_calibration_type_t,
                         target_camera: _k4a.k4a_calibration_type_t) -> _k4a.k4a_float3_t():

        target_point3d = _k4a.k4a_float3_t()

        _k4a.VERIFY(
            _k4a.k4a_calibration_3d_to_3d(
                self._handle, source_point3d, source_camera, target_camera, target_point3d),
            "Failed to convert from 3D to 3D")

        return target_point3d

    def convert_2d_to_3d(self, source_point2d: _k4a.k4a_float2_t,
                         source_depth: float,
                         source_camera: _k4a.k4a_calibration_type_t,
                         target_camera: _k4a.k4a_calibration_type_t) -> _k4a.k4a_float3_t():

        target_point3d = _k4a.k4a_float3_t()
        valid = ctypes.c_int()

        _k4a.VERIFY(
            _k4a.k4a_calibration_2d_to_3d(self._handle, source_point2d, source_depth, source_camera, target_camera,
                                          target_point3d, valid), "Failed to convert from 2D to 3D")

        return target_point3d

    def convert_3d_to_2d(self, source_point3d: _k4a.k4a_float3_t,
                         source_camera: _k4a.k4a_calibration_type_t,
                         target_camera: _k4a.k4a_calibration_type_t) -> _k4a.k4a_float2_t():

        target_point2d = _k4a.k4a_float2_t()
        valid = ctypes.c_int()

        _k4a.VERIFY(
            _k4a.k4a_calibration_3d_to_2d(self._handle, source_point3d, source_camera, target_camera, target_point2d,
                                          valid), "Failed to convert from 3D to 2D")

        return target_point2d, valid

    def convert_2d_to_2d(self, source_point2d: _k4a.k4a_float2_t,
                         source_depth: float,
                         source_camera: _k4a.k4a_calibration_type_t,
                         target_camera: _k4a.k4a_calibration_type_t) -> _k4a.k4a_float2_t():

        target_point2d = _k4a.k4a_float2_t()
        valid = ctypes.c_int()

        _k4a.VERIFY(
            _k4a.k4a_calibration_2d_to_2d(self._handle, source_point2d, source_depth, source_camera, target_camera,
                                          target_point2d, valid), "Failed to convert from 2D to 2D")

        return target_point2d

    def convert_color_2d_to_depth_2d(self, source_point2d: _k4a.k4a_float2_t,
                                     depth_image: _k4a.k4a_image_t) -> _k4a.k4a_float2_t():

        target_point2d = _k4a.k4a_float2_t()
        valid = ctypes.c_int()

        _k4a.VERIFY(
            _k4a.k4a_calibration_color_2d_to_depth_2d(self._handle, source_point2d, depth_image, target_point2d,
                                                      valid), "Failed to convert from Color 2D to Depth 2D")

        return target_point2d
