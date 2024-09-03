
#define _USE_MATH_DEFINES
#include <cuComplex.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>

extern "C" {

__constant__ float project[12];

__device__ float3 project_point(float3 point) {
  float3 clip;
  clip.x = project[0] * point.x + project[1] * point.y + project[2] * point.z +
           project[3];
  clip.y = project[4] * point.x + project[5] * point.y + project[6] * point.z +
           project[7];
  clip.z = project[8] * point.x + project[9] * point.y + project[10] * point.z +
           project[11];
  return clip;
}

__global__ void project_depth(float3* points, int num_points, int width,
                              int height, float* out) {
  int point_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (point_idx >= num_points) return;

  float3 p_orig = points[point_idx];
  float3 p = project_point(p_orig);
  int2 px;

  px.x = round((p.x / p.z));
  px.y = round((p.y / p.z));

  if (px.x < width && px.y < height && px.x >= 0 && px.y >= 0) {
    out[px.y * width + px.x] = p.z;
  }
}
}