

#include <cstdint>
#include <limits>
#define _USE_MATH_DEFINES
#include <cuComplex.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>

extern "C" {

const int LOCAL_WARP_SIZE = 32;
const float FLT_MAX = std::numeric_limits<int>::max();

__constant__ float project[16];
__constant__ float project_inv[16];

__device__ float4 project_point(float4 point) {
  float4 clip;
  clip.x = project[0] * point.x + project[1] * point.y + project[2] * point.z +
           project[3];
  clip.y = project[4] * point.x + project[5] * point.y + project[6] * point.z +
           project[7];
  clip.z = project[8] * point.x + project[9] * point.y + project[10] * point.z +
           project[11];
  clip.w = project[12] * point.x + project[13] * point.y +
           project[14] * point.z + project[15];
  return clip;
}

__device__ float4 backproject(float4 point) {
  float4 world;
  world.x = project_inv[0] * point.x + project_inv[1] * point.y +
            project_inv[2] * point.z + project_inv[3];
  world.y = project_inv[4] * point.x + project_inv[5] * point.y +
            project_inv[6] * point.z + project_inv[7];
  world.z = project_inv[8] * point.x + project_inv[9] * point.y +
            project_inv[10] * point.z + project_inv[11];
  world.w = project_inv[12] * point.x + project_inv[13] * point.y +
            project_inv[14] * point.z + project_inv[15];
  return world;
}

__global__ void project_depth(float4 *points, int num_points, int width,
                              int height, uint8_t *mask, float *out) {
  int point_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (point_idx >= num_points) return;

  float4 p_orig = points[point_idx];
  float4 p = project_point(p_orig);
  int2 px;

  px.x = floor((p.x / p.w));
  px.y = floor((p.y / p.w));

  if (px.x < width && px.y < height && px.x >= 0 && px.y >= 0 &&
      mask[px.y * width + px.x] > 0) {
    out[px.y * width + px.x] = p.z;
  } else if (mask[px.y * width + px.x] > 0) {
    printf("mask: %u\n", mask[px.y * width + px.x]);
  }
}

__global__ void project_depth_color(float4 *points, int num_points,
                                    float3 *color, int width, int height,
                                    uint8_t *mask, float *out_depth,
                                    float3 *out_color) {
  int point_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (point_idx >= num_points) return;

  float4 p_orig = points[point_idx];
  float4 p = project_point(p_orig);
  int2 px;

  px.x = floor((p.x / p.w));
  px.y = floor((p.y / p.w));

  if (px.x < width && px.y < height && px.x >= 0 && px.y >= 0 &&
      mask[px.y * width + px.x] > 0) {
    out_depth[px.y * width + px.x] = p.z;
    out_color[px.y * width + px.x] = color[point_idx];
  }
}

__global__ void view_dependence(float4 *points, int num_points, float3 *normal,
                                int width, int height, float3 *dependence) {
  int point_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (point_idx >= num_points) return;

  float4 p_orig = points[point_idx];
  float3 n = normal[point_idx];
  float4 p = project_point(p_orig);
  int2 px;

  px.x = floor((p.x / p.w));
  px.y = floor((p.y / p.w));

  if (px.x < width && px.y < height && px.x >= 0 && px.y >= 0) {
    float4 back = backproject(p);

    float4 ray = back;
    float len = sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
    ray.x = ray.x / len;
    ray.y = ray.y / len;
    ray.z = ray.z / len;

    float dot = abs(ray.x * n.x + ray.y * n.y + ray.z * n.z);

    dependence[point_idx].x = dot;
    dependence[point_idx].y = float(dot > 0.75);
    dependence[point_idx].z = float(dot > 0.55);
  }
}

__global__ void compute_pixels(float4 *points, int num_points, int width,
                               int height, float2 *pixels) {
  int point_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (point_idx >= num_points) return;

  float4 p_orig = points[point_idx];
  float4 p = project_point(p_orig);
  float2 px;
  px.x = ((p.x / p.w));
  px.y = ((p.y / p.w));

  pixels[point_idx].x = px.x;
  pixels[point_idx].y = px.y;
}

__device__ int valid_pixel(float2 px, int width, int height) {
  if (px.x < 0 || px.x >= width) return 0;
  if (px.y < 0 || px.y >= height) return 0;
  return 1;
}

__global__ void compute_fragments_per_triangle(float2 *pixels, int num_pixels,
                                               uint32_t *faces, int num_faces,
                                               int width, int height,
                                               uint32_t *fragments_per_face) {
  int face_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (face_idx >= num_faces) return;

  uint32_t i1 = faces[face_idx * 3 + 0];
  uint32_t i2 = faces[face_idx * 3 + 1];
  uint32_t i3 = faces[face_idx * 3 + 2];

  uint32_t min_x = width;
  uint32_t max_x = 0;
  uint32_t min_y = height;
  uint32_t max_y = 0;

  float2 v1 = pixels[i1];
  float2 v2 = pixels[i2];
  float2 v3 = pixels[i3];

  int valid_px = valid_pixel(v1, width, height) +
                 valid_pixel(v2, width, height) +
                 valid_pixel(v3, width, height);
  if (valid_px == 0) {
    fragments_per_face[face_idx] = 0;
    return;
  }

  min_x = min(min_x, int(v1.x));
  min_x = min(min_x, int(v2.x));
  min_x = min(min_x, int(v3.x));
  max_x = max(max_x, int(v1.x));
  max_x = max(max_x, int(v2.x));
  max_x = max(max_x, int(v3.x));
  min_y = min(min_y, int(v1.y));
  min_y = min(min_y, int(v2.y));
  min_y = min(min_y, int(v3.y));
  max_y = max(max_y, int(v1.y));
  max_y = max(max_y, int(v2.y));
  max_y = max(max_y, int(v3.y));

  min_x = max(min_x, 0);
  min_y = max(min_y, 0);
  max_x = min(max_x, width - 1);
  max_y = min(max_y, height - 1);

  fragments_per_face[face_idx] = (max_x - min_x + 1) * (max_y - min_y + 1);
}

__global__ void interpolate_points(float2 *pixels, float4 *points,
                                   uint32_t *faces, int num_faces, int width,
                                   int height, uint32_t *fragment_start_indices,
                                   float2 *interpolated_pixels,
                                   float3 *interpolated_points) {
  int face_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (face_idx >= num_faces) return;

  uint32_t i1 = faces[face_idx * 3 + 0];
  uint32_t i2 = faces[face_idx * 3 + 1];
  uint32_t i3 = faces[face_idx * 3 + 2];

  float2 v1 = pixels[i1];
  float2 v2 = pixels[i2];
  float2 v3 = pixels[i3];

  int fragment_start_index = fragment_start_indices[face_idx];

  int fragments = (int)(fragment_start_indices[face_idx + 1]) -
                  (int)(fragment_start_indices[face_idx]);
  if (fragments == 0) return;

  float4 p1 = points[i1];
  float4 p2 = points[i2];
  float4 p3 = points[i3];

  uint32_t min_x = width;
  uint32_t max_x = 0;
  uint32_t min_y = height;
  uint32_t max_y = 0;

  min_x = min(min_x, int(v1.x));
  min_x = min(min_x, int(v2.x));
  min_x = min(min_x, int(v3.x));
  max_x = max(max_x, int(v1.x));
  max_x = max(max_x, int(v2.x));
  max_x = max(max_x, int(v3.x));
  min_y = min(min_y, int(v1.y));
  min_y = min(min_y, int(v2.y));
  min_y = min(min_y, int(v3.y));
  max_y = max(max_y, int(v1.y));
  max_y = max(max_y, int(v2.y));
  max_y = max(max_y, int(v3.y));

  min_x = max(min_x, 0);
  min_y = max(min_y, 0);
  max_x = min(max_x, width - 1);
  max_y = min(max_y, height - 1);

  int bb_width = (max_x - min_x + 1);
  int bb_height = (max_y - min_y + 1);

  for (int32_t y_ind = 0; y_ind < bb_height; y_ind++) {
    for (int32_t x_ind = 0; x_ind < bb_width; x_ind++) {
      float2 p;
      p.x = min_x + x_ind + 0.5;
      p.y = min_y + y_ind + 0.5;

      float lambda_1 =
          ((v2.y - v3.y) * (p.x - v3.x) + (v3.x - v2.x) * (p.y - v3.y)) /
          ((v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y));
      float lambda_2 =
          ((v3.y - v1.y) * (p.x - v3.x) + (v1.x - v3.x) * (p.y - v3.y)) /
          ((v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y));
      float lambda_3 = 1 - lambda_2 - lambda_1;

      if (lambda_1 < 0 || lambda_2 < 0 || lambda_3 < 0 || isnan(lambda_1) ||
          isnan(lambda_2) || isnan(lambda_3)) {
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].x =
            -1;
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].y =
            -1;
      } else {
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].x =
            p.x;
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].y =
            p.y;
        interpolated_points[fragment_start_index + y_ind * bb_width + x_ind].x =
            lambda_1 * p1.x + lambda_2 * p2.x + lambda_3 * p3.x;
        interpolated_points[fragment_start_index + y_ind * bb_width + x_ind].y =
            lambda_1 * p1.y + lambda_2 * p2.y + lambda_3 * p3.y;
        interpolated_points[fragment_start_index + y_ind * bb_width + x_ind].z =
            lambda_1 * p1.z + lambda_2 * p2.z + lambda_3 * p3.z;
      }
    }
  }
}

__global__ void interpolate_attr1(float2 *pixels, float4 *points,
                                  uint32_t *faces, int num_faces, int width,
                                  int height, float *face_attribute,
                                  uint32_t *fragment_start_indices,
                                  float2 *interpolated_pixels,
                                  float3 *interpolated_points,
                                  float *interpolated_attribute) {
  int face_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (face_idx >= num_faces) return;

  uint32_t i1 = faces[face_idx * 3 + 0];
  uint32_t i2 = faces[face_idx * 3 + 1];
  uint32_t i3 = faces[face_idx * 3 + 2];

  float2 v1 = pixels[i1];
  float2 v2 = pixels[i2];
  float2 v3 = pixels[i3];

  int fragment_start_index = fragment_start_indices[face_idx];

  int fragments = (int)(fragment_start_indices[face_idx + 1]) -
                  (int)(fragment_start_indices[face_idx]);
  if (fragments == 0) return;

  float4 p1 = points[i1];
  float4 p2 = points[i2];
  float4 p3 = points[i3];

  float a1 = face_attribute[i1];
  float a2 = face_attribute[i2];
  float a3 = face_attribute[i3];

  uint32_t min_x = width;
  uint32_t max_x = 0;
  uint32_t min_y = height;
  uint32_t max_y = 0;

  min_x = min(min_x, int(v1.x));
  min_x = min(min_x, int(v2.x));
  min_x = min(min_x, int(v3.x));
  max_x = max(max_x, int(v1.x));
  max_x = max(max_x, int(v2.x));
  max_x = max(max_x, int(v3.x));
  min_y = min(min_y, int(v1.y));
  min_y = min(min_y, int(v2.y));
  min_y = min(min_y, int(v3.y));
  max_y = max(max_y, int(v1.y));
  max_y = max(max_y, int(v2.y));
  max_y = max(max_y, int(v3.y));

  min_x = max(min_x, 0);
  min_y = max(min_y, 0);
  max_x = min(max_x, width - 1);
  max_y = min(max_y, height - 1);

  int bb_width = (max_x - min_x + 1);
  int bb_height = (max_y - min_y + 1);

  for (int32_t y_ind = 0; y_ind < bb_height; y_ind++) {
    for (int32_t x_ind = 0; x_ind < bb_width; x_ind++) {
      float2 p;
      p.x = min_x + x_ind + 0.5;
      p.y = min_y + y_ind + 0.5;

      float lambda_1 =
          ((v2.y - v3.y) * (p.x - v3.x) + (v3.x - v2.x) * (p.y - v3.y)) /
          ((v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y));
      float lambda_2 =
          ((v3.y - v1.y) * (p.x - v3.x) + (v1.x - v3.x) * (p.y - v3.y)) /
          ((v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y));
      float lambda_3 = 1 - lambda_2 - lambda_1;

      if (lambda_1 < 0 || lambda_2 < 0 || lambda_3 < 0 || isnan(lambda_1) ||
          isnan(lambda_2) || isnan(lambda_3)) {
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].x =
            -1;
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].y =
            -1;
      } else {
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].x =
            p.x;
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].y =
            p.y;
        interpolated_points[fragment_start_index + y_ind * bb_width + x_ind].x =
            lambda_1 * p1.x + lambda_2 * p2.x + lambda_3 * p3.x;
        interpolated_points[fragment_start_index + y_ind * bb_width + x_ind].y =
            lambda_1 * p1.y + lambda_2 * p2.y + lambda_3 * p3.y;
        interpolated_points[fragment_start_index + y_ind * bb_width + x_ind].z =
            lambda_1 * p1.z + lambda_2 * p2.z + lambda_3 * p3.z;
        interpolated_attribute[fragment_start_index + y_ind * bb_width +
                               x_ind] =
            lambda_1 * a1 + lambda_2 * a2 + lambda_3 * a3;
      }
    }
  }
}

__global__ void interpolate_attr3(float2 *pixels, float4 *points,
                                  uint32_t *faces, int num_faces, int width,
                                  int height, float3 *face_attribute,
                                  uint32_t *fragment_start_indices,
                                  float2 *interpolated_pixels,
                                  float3 *interpolated_points,
                                  float3 *interpolated_attribute) {
  int face_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (face_idx >= num_faces) return;

  uint32_t i1 = faces[face_idx * 3 + 0];
  uint32_t i2 = faces[face_idx * 3 + 1];
  uint32_t i3 = faces[face_idx * 3 + 2];

  float2 v1 = pixels[i1];
  float2 v2 = pixels[i2];
  float2 v3 = pixels[i3];

  int fragment_start_index = fragment_start_indices[face_idx];

  int fragments = (int)(fragment_start_indices[face_idx + 1]) -
                  (int)(fragment_start_indices[face_idx]);

  if (fragments == 0) return;

  float4 p1 = points[i1];
  float4 p2 = points[i2];
  float4 p3 = points[i3];

  float3 a1 = face_attribute[i1];
  float3 a2 = face_attribute[i2];
  float3 a3 = face_attribute[i3];

  uint32_t min_x = width;
  uint32_t max_x = 0;
  uint32_t min_y = height;
  uint32_t max_y = 0;

  min_x = min(min_x, int(v1.x));
  min_x = min(min_x, int(v2.x));
  min_x = min(min_x, int(v3.x));
  max_x = max(max_x, int(v1.x));
  max_x = max(max_x, int(v2.x));
  max_x = max(max_x, int(v3.x));
  min_y = min(min_y, int(v1.y));
  min_y = min(min_y, int(v2.y));
  min_y = min(min_y, int(v3.y));
  max_y = max(max_y, int(v1.y));
  max_y = max(max_y, int(v2.y));
  max_y = max(max_y, int(v3.y));

  min_x = max(min_x, 0);
  min_y = max(min_y, 0);
  max_x = min(max_x, width - 1);
  max_y = min(max_y, height - 1);

  int bb_width = (max_x - min_x + 1);
  int bb_height = (max_y - min_y + 1);

  for (int32_t y_ind = 0; y_ind < bb_height; y_ind++) {
    for (int32_t x_ind = 0; x_ind < bb_width; x_ind++) {
      float2 p;
      p.x = min_x + x_ind + 0.5;
      p.y = min_y + y_ind + 0.5;

      float lambda_1 =
          ((v2.y - v3.y) * (p.x - v3.x) + (v3.x - v2.x) * (p.y - v3.y)) /
          ((v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y));
      float lambda_2 =
          ((v3.y - v1.y) * (p.x - v3.x) + (v1.x - v3.x) * (p.y - v3.y)) /
          ((v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y));
      float lambda_3 = 1 - lambda_2 - lambda_1;

      if (lambda_1 < 0 || lambda_2 < 0 || lambda_3 < 0 || isnan(lambda_1) ||
          isnan(lambda_2) || isnan(lambda_3)) {
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].x =
            -1;
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].y =
            -1;
      } else {
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].x =
            p.x;
        interpolated_pixels[fragment_start_index + y_ind * bb_width + x_ind].y =
            p.y;
        interpolated_points[fragment_start_index + y_ind * bb_width + x_ind].x =
            lambda_1 * p1.x + lambda_2 * p2.x + lambda_3 * p3.x;
        interpolated_points[fragment_start_index + y_ind * bb_width + x_ind].y =
            lambda_1 * p1.y + lambda_2 * p2.y + lambda_3 * p3.y;
        interpolated_points[fragment_start_index + y_ind * bb_width + x_ind].z =
            lambda_1 * p1.z + lambda_2 * p2.z + lambda_3 * p3.z;

        interpolated_attribute[fragment_start_index + y_ind * bb_width + x_ind]
            .x = lambda_1 * a1.x + lambda_2 * a2.x + lambda_3 * a3.x;
        interpolated_attribute[fragment_start_index + y_ind * bb_width + x_ind]
            .y = lambda_1 * a1.y + lambda_2 * a2.y + lambda_3 * a3.y;
        interpolated_attribute[fragment_start_index + y_ind * bb_width + x_ind]
            .z = lambda_1 * a1.z + lambda_2 * a2.z + lambda_3 * a3.z;
      }
    }
  }
}

__global__ void projective_error(float4 *points, int num_points,
                                 uint8_t *valid_mask_points, float *depth,
                                 int width, int height,
                                 uint8_t *valid_mask_depth, float *error) {
  int point_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (point_idx >= num_points) return;

  float4 p_orig = points[point_idx];
  float4 p = project_point(p_orig);
  int2 px;

  px.x = floor((p.x / p.w));
  px.y = floor((p.y / p.w));

  if (px.x < width && px.y < height && px.x >= 0 && px.y >= 0) {
    float error_val = (p.z - depth[px.y * width + px.x]);
    // printf("src: %u, dest: %u, error: %f\n", valid_mask_points[point_idx],
    //        valid_mask_depth[px.y * width + px.x], error_val);
    if (valid_mask_points[point_idx] && valid_mask_depth[px.y * width + px.x]) {
      error[point_idx] = error_val;
    } else {
      error[point_idx] = FLT_MAX;
    }
  }
}

__global__ void projective_error_attribute(const float4 *points, int num_points,
                                           const float *src_attribute,
                                           const uint8_t *src_valid,
                                           const float *dest_attribute,
                                           const uint8_t *dest_valid, int width,
                                           int height, float *error) {
  int point_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (point_idx >= num_points) return;

  float4 p_orig = points[point_idx];
  float4 p = project_point(p_orig);
  int2 px;

  px.x = floor((p.x / p.w));
  px.y = floor((p.y / p.w));

  if (px.x < width && px.y < height && px.x >= 0 && px.y >= 0) {
    float error_val =
        (src_attribute[point_idx] - dest_attribute[px.y * width + px.x]);

    if (src_valid[point_idx] != 0 && dest_valid[px.y * width + px.x] != 0) {
      error[point_idx] = error_val;
    } else {
      error[point_idx] = FLT_MAX;
    }
  }
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float vec_length(const float3 &a) {
  return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ __forceinline__ float vec_length_2d(const float3 &a) {
  return sqrt(a.x * a.x + a.y * a.y);
}

__global__ void one_sided_chamfer(float3 *p1, int32_t p1_size,
                                  uint8_t *valid_p1, float3 *p2,
                                  int32_t p2_size, uint8_t *valid_p2,
                                  float *dists) {
  int thread_idx = threadIdx.x % 32;
  int point_idx = (threadIdx.x + blockIdx.x * blockDim.x) / 32;

  if (point_idx >= p1_size) {
    return;
  }

  float3 p = p1[point_idx];
  if (!valid_p1[point_idx]) {
    if (thread_idx == 0) dists[point_idx] = FLT_MAX;
    return;
  }

  float shared_best_dist = FLT_MAX;
  for (unsigned int i = thread_idx; i < p2_size; i += LOCAL_WARP_SIZE) {
    if (i < p2_size) {
      float3 p_ = p2[i];
      float3 d = p - p_;

      float norm = vec_length(d);
      if ((valid_p2[i]) && norm < shared_best_dist) {
        shared_best_dist = norm;
      }
    }
  }

// minimum accross warp
#pragma unroll
  for (int offset = LOCAL_WARP_SIZE / 2; offset > 0; offset /= 2) {
    shared_best_dist =
        fmin(__shfl_down_sync(pow(2, offset) - 1, shared_best_dist, offset),
             shared_best_dist);
  }

  if (thread_idx == 0) dists[point_idx] = shared_best_dist;
}

__global__ void projective_one_sided_chamfer(float3 *p1, int p1_size,
                                             float3 *p2, int p2_size,
                                             float *dists, float thresh) {
  int thread_idx = threadIdx.x;
  int point_idx = threadIdx.y + blockIdx.y * blockDim.y + 1;

  float3 p = p1[point_idx];

  if (p.z == 0) {
    if (thread_idx == 0) dists[point_idx] = 0;
    return;
  }

  __shared__ float shared_best_dist[32][LOCAL_WARP_SIZE];
  __shared__ int shared_best_index[32][LOCAL_WARP_SIZE];

  shared_best_dist[threadIdx.y][thread_idx] = FLT_MAX;
  shared_best_index[threadIdx.y][thread_idx] = -1;
  for (unsigned int i = thread_idx; i < p2_size; i += LOCAL_WARP_SIZE) {
    float3 p_ = p2[i];
    float3 d = p - p_;
    float norm = vec_length_2d(d);
    if (norm > thresh) continue;
    if (i == thread_idx || norm < shared_best_dist[threadIdx.y][thread_idx]) {
      shared_best_dist[threadIdx.y][thread_idx] = norm;
      shared_best_index[threadIdx.y][thread_idx] = i;
    }
  }

  // printf("T%i : %f\n", thread_idx,
  // shared_best_dist[threadIdx.y][thread_idx]);

  // minimum accross warp
  for (int offset = LOCAL_WARP_SIZE / 2; offset > 0; offset /= 2) {
    if (thread_idx < offset) {
      if (shared_best_dist[threadIdx.y][thread_idx] >=
          shared_best_dist[threadIdx.y][thread_idx + offset]) {
        shared_best_dist[threadIdx.y][thread_idx] =
            shared_best_dist[threadIdx.y][thread_idx + offset];
        shared_best_index[threadIdx.y][thread_idx] =
            shared_best_index[threadIdx.y][thread_idx + offset];
      }
    }
  }

  // if (thread_idx == 0)
  //   printf("Best dist: %f\n", shared_best_dist[threadIdx.y][0]);

  if (thread_idx == 0) {
    if (shared_best_index[threadIdx.y][0] >= 0) {
      float metric = vec_length(p - p2[shared_best_index[threadIdx.y][0]]);
      dists[point_idx] = metric;
    }
  }
}

__global__ void one_sided_chamfer_axis(float *p1, int p1_size, float *p2,
                                       int p2_size, int axis, float *dists) {
  int thread_idx = threadIdx.x;
  int point_idx = threadIdx.y + blockIdx.y * blockDim.y + 1;

  float3 p = make_float3(p1[point_idx * 3 + 0], p1[point_idx * 3 + 1],
                         p1[point_idx * 3 + 2]);
  float p_axis = p1[point_idx * 3 + axis];

  if (p.z == 0) {
    if (thread_idx == 0) dists[point_idx] = 0;
    return;
  }

  __shared__ float shared_best_dist[32][LOCAL_WARP_SIZE];
  __shared__ int shared_best_index[32][LOCAL_WARP_SIZE];

  for (unsigned int i = thread_idx; i < p2_size; i += LOCAL_WARP_SIZE) {
    float3 p_ = make_float3(p2[i * 3 + 0], p2[i * 3 + 1], p2[i * 3 + 2]);
    float3 d = p - p_;
    float norm = vec_length(d);
    if (i == thread_idx || norm < shared_best_dist[threadIdx.y][thread_idx]) {
      shared_best_dist[threadIdx.y][thread_idx] = norm;
      shared_best_index[threadIdx.y][thread_idx] = i;
    }
  }

  // printf("T%i : %f\n", thread_idx,
  // shared_best_dist[threadIdx.y][thread_idx]);

  // minimum accross warp
  for (int offset = LOCAL_WARP_SIZE / 2; offset > 0; offset /= 2) {
    if (thread_idx < offset) {
      if (shared_best_dist[threadIdx.y][thread_idx] >=
          shared_best_dist[threadIdx.y][thread_idx + offset]) {
        shared_best_dist[threadIdx.y][thread_idx] =
            shared_best_dist[threadIdx.y][thread_idx + offset];
        shared_best_index[threadIdx.y][thread_idx] =
            shared_best_index[threadIdx.y][thread_idx + offset];
      }
    }
  }

  // if (thread_idx == 0)
  //   printf("Best dist: %f\n", shared_best_dist[threadIdx.y][0]);

  if (thread_idx == 0) {
    dists[point_idx] =
        abs(p_axis - p2[shared_best_index[threadIdx.y][0] * 3 + axis]);
  }
}

__global__ void projective_one_sided_chamfer_axis(float *p1, int p1_size,
                                                  float *p2, int p2_size,
                                                  int axis, float *dists) {
  int thread_idx = threadIdx.x;
  int point_idx = threadIdx.y + blockIdx.y * blockDim.y + 1;

  float3 p = make_float3(p1[point_idx * 3 + 0], p1[point_idx * 3 + 1],
                         p1[point_idx * 3 + 2]);
  float p_axis = p1[point_idx * 3 + axis];

  if (p.z == 0) {
    if (thread_idx == 0) dists[point_idx] = 0;
    return;
  }

  __shared__ float shared_best_dist[32][LOCAL_WARP_SIZE];
  __shared__ int shared_best_index[32][LOCAL_WARP_SIZE];

  for (unsigned int i = thread_idx; i < p2_size; i += LOCAL_WARP_SIZE) {
    float3 p_ = make_float3(p2[i * 3 + 0], p2[i * 3 + 1], p2[i * 3 + 2]);
    float3 d = p - p_;
    float norm = vec_length_2d(d);
    if (i == thread_idx || norm < shared_best_dist[threadIdx.y][thread_idx]) {
      shared_best_dist[threadIdx.y][thread_idx] = norm;
      shared_best_index[threadIdx.y][thread_idx] = i;
    }
  }

  // printf("T%i : %f\n", thread_idx,
  // shared_best_dist[threadIdx.y][thread_idx]);

  // minimum accross warp
  for (int offset = LOCAL_WARP_SIZE / 2; offset > 0; offset /= 2) {
    if (thread_idx < offset) {
      if (shared_best_dist[threadIdx.y][thread_idx] >=
          shared_best_dist[threadIdx.y][thread_idx + offset]) {
        shared_best_dist[threadIdx.y][thread_idx] =
            shared_best_dist[threadIdx.y][thread_idx + offset];
        shared_best_index[threadIdx.y][thread_idx] =
            shared_best_index[threadIdx.y][thread_idx + offset];
      }
    }
  }

  // if (thread_idx == 0)
  //   printf("Best dist: %f\n", shared_best_dist[threadIdx.y][0]);

  if (thread_idx == 0) {
    dists[point_idx] =
        abs(p_axis - p2[shared_best_index[threadIdx.y][0] * 3 + axis]);
  }
}
}