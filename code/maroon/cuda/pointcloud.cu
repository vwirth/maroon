
#include <cstdint>
#define _USE_MATH_DEFINES
#include <cuComplex.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>

extern "C" {

const float SQRT_2PI = 2.506628274631;

__device__ int is_valid(float3* points, int width, int height, int x, int y) {
  if (x < 0 || x >= width) return 0;
  if (y < 0 || y >= height) return 0;
  if (abs(points[y * width + x].z) < 1e-3) return 0;
  return 1;
}

__device__ int is_valid_px(int width, int height, int x, int y) {
  if (x < 0 || x >= width) return 0;
  if (y < 0 || y >= height) return 0;
  return 1;
}
__device__ bool valid_neighbors(float3* points, int index1, int index2,
                                float threshold) {
  if (abs((points[index1].z - points[index2].z)) > threshold) return false;
  if (abs((points[index1].z)) < 1e-3 || abs((points[index2].z) < 1e-3))
    return false;
  return true;
}

__device__ __forceinline__ float vec_length_square(const float2& a) {
  return (a.x * a.x + a.y * a.y);
}

__device__ float gauss_1d(float val, float stddev) {
  return (1.0 / (SQRT_2PI * stddev)) *
         exp(-(val * val) / (2 * stddev * stddev));
}

__device__ float gauss_2d(float2 val, float stddev) {
  return (1.0 / (SQRT_2PI * stddev)) *
         exp(-(vec_length_square(val)) / (2 * stddev * stddev));
}

__device__ float2 operator-(const float2& a, const float2& b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

__global__ void bilateral_filtering(float* attribute_map, int width, int height,
                                    float stddev_space, float stddev_range,
                                    float* attribute_map_out) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= width || y >= height) return;

  float2 px_center = make_float2(x, y);

  float weight = 0;
  float avg = 0;
  for (int offset_x = -stddev_space * 3; offset_x <= stddev_space * 3;
       offset_x++) {
    for (int offset_y = -stddev_space * 3; offset_y <= stddev_space * 3;
         offset_y++) {
      if (!is_valid_px(width, height, x + offset_x, y + offset_y)) continue;
      float2 px_n = make_float2(x + offset_x, y + offset_y);
      float space_weight = gauss_2d(px_center - px_n, stddev_space);
      float range_weight =
          gauss_1d(attribute_map[y * width + x] -
                       attribute_map[(y + offset_y) * width + x + offset_x],
                   stddev_range);
      avg += space_weight * range_weight *
             attribute_map[(y + offset_y) * width + x + offset_x];

      // if (x == 100 && y == 100) {
      //   printf(
      //       "(%i,%i) offset: (%i,%i), space_weight: %f, range_weight: %f, a:"
      //       "%f, avg: "
      //       "%f\n",
      //       x, y, offset_x, offset_y, space_weight, range_weight,
      //       attribute_map[(y + offset_y) * width + x + offset_x], avg);
      // }

      weight = weight + space_weight * range_weight;
    }
  }
  avg = avg / weight;

  // if (x == 100 && y == 100) {
  //   printf("(%i,%i) final avg: %f", x, y, avg);
  // }

  attribute_map_out[y * width + x] = avg;
}

__global__ void outlier_filtering(float3* points, int width, int height,
                                  int kernel_size, float threshold,
                                  uint8_t* valid_mask) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int linear_index = y * width + x;

  if (x >= width || y >= height) return;

  float3 center = points[linear_index];

  int stride = int(kernel_size / 2);
  float mean_offset = 0;
  int valid_points = 0;
  for (int offset_x = -stride; offset_x <= stride; offset_x++) {
    for (int offset_y = -stride; offset_y <= stride; offset_y++) {
      if (offset_x == 0 && offset_y == 0) continue;
      if (is_valid(points, width, height, x + offset_x, y + offset_y)) {
        mean_offset +=
            abs(center.z - points[(y + offset_y) * width + x + offset_x].z);
        valid_points++;
      }
    }
  }
  mean_offset = mean_offset / valid_points;
  if (linear_index < 1000) {
    printf("index: %d, offset: %f, threshold: %f\n", linear_index, mean_offset,
           threshold);
  }

  if (mean_offset <= threshold) {
    valid_mask[linear_index] = 1;
  }
}

__global__ void triangulate(float3* points, int width, int height,
                            float threshold, int32_t* index_list) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int linear_index = y * width + x;

  if (x >= width || y >= height) return;

  uint32_t i00 = y * width + x;
  uint32_t i10 = (y + 1) * width + x;
  uint32_t i01 = y * width + x + 1;
  uint32_t i11 = (y + 1) * width + x + 1;

  int num_vertices = is_valid(points, width, height, x, y) +
                     is_valid(points, width, height, x + 1, y) +
                     is_valid(points, width, height, x, y + 1) +
                     is_valid(points, width, height, x + 1, y + 1);

  if (num_vertices < 3) return;

  if (num_vertices == 4) {
    if (valid_neighbors(points, i00, i10, threshold) &&
        valid_neighbors(points, i10, i01, threshold)) {
      index_list[linear_index * 6 + 0] = i00 + 1;
      index_list[linear_index * 6 + 1] = i10 + 1;
      index_list[linear_index * 6 + 2] = i01 + 1;
    }

    if (valid_neighbors(points, i11, i01, threshold) &&
        valid_neighbors(points, i01, i10, threshold)) {
      index_list[linear_index * 6 + 3] = i11 + 1;
      index_list[linear_index * 6 + 4] = i01 + 1;
      index_list[linear_index * 6 + 5] = i10 + 1;
    }
  } else {
    if (!is_valid(points, width, height, 0, 0)) {
      if (valid_neighbors(points, i11, i10, threshold) &&
          valid_neighbors(points, i10, i01, threshold)) {
        index_list[linear_index * 6 + 0] = i11 + 1;
        index_list[linear_index * 6 + 1] = i10 + 1;
        index_list[linear_index * 6 + 2] = i01 + 1;
      }
    } else if (!is_valid(points, width, height, 1, 0)) {
      if (valid_neighbors(points, i00, i10, threshold) &&
          valid_neighbors(points, i10, i11, threshold)) {
        index_list[linear_index * 6 + 0] = i00 + 1;
        index_list[linear_index * 6 + 1] = i10 + 1;
        index_list[linear_index * 6 + 2] = i11 + 1;
      }
    } else if (!is_valid(points, width, height, 0, 1)) {
      if (valid_neighbors(points, i00, i01, threshold) &&
          valid_neighbors(points, i01, i11, threshold)) {
        index_list[linear_index * 6 + 0] = i00 + 1;
        index_list[linear_index * 6 + 1] = i01 + 1;
        index_list[linear_index * 6 + 2] = i11 + 1;
      }
    } else if (!is_valid(points, width, height, 1, 1)) {
      if (valid_neighbors(points, i00, i10, threshold) &&
          valid_neighbors(points, i10, i01, threshold)) {
        index_list[linear_index * 6 + 0] = i00 + 1;
        index_list[linear_index * 6 + 1] = i10 + 1;
        index_list[linear_index * 6 + 2] = i01 + 1;
      }
    }
  }
}
}