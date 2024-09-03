// #include <__clang_cuda_builtin_vars.h>
#define _USE_MATH_DEFINES
#include <cuComplex.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>

extern "C" {
const int LOCAL_WARP_SIZE = 32;
__constant__ float frequencies[512];

__device__ __forceinline__ cuComplex comp_exp(float phase) {
  cuComplex result = make_cuComplex(cos(phase), sin(phase));
  return result;
}
__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float vec_length(const float3 &a) {
  return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ __forceinline__ float points_dist(const float3 &a, const float3 &b) {
  float x = a.x - b.x;
  float y = a.y - b.y;
  float z = a.z - b.z;
  return sqrt(x * x + y * y + z * z);
}

__device__ cuComplex operator*(const cuComplex &a, const cuComplex &b) {
  return make_cuComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ cuComplex calculate(const cuComplex &result,
                                               const cuComplex &time_signal,
                                               float frequency, float delay) {
  // float x = cos(frequency * delay);
  // float y = sin(frequency * delay);
  float x;
  float y;
  sincosf(frequency * delay, &y, &x);
  return make_cuComplex(result.x + time_signal.x * x - time_signal.y * y,
                        result.y + time_signal.x * y + time_signal.y * x);

  // float real = result.x + time_signal.x * x - time_signal.y * y;
  // return make_cuComplex(
  //     result.x + 1, result.y + (time_signal.x * y + time_signal.y * x) /
  //     real);
}

__device__ __forceinline__ cuComplex warpReduceSum(cuComplex val) {
#pragma unroll
  for (int offset = LOCAL_WARP_SIZE / 2; offset > 0; offset /= 2) {
    float *a = reinterpret_cast<float *>(&val);
    a[0] += __shfl_down_sync(0xFFFFFFFF, a[0], offset);
    a[1] += __shfl_down_sync(0xFFFFFFFF, a[1], offset);
  }
  return val;
}

__global__ void holo_reco_sfcw_time_domain(
    float3 *tx_antennas, int num_tx_antennas, float3 *rx_antennas,
    int num_rx_antennas, float *reco_positions_x, int num_x_positions,
    float *reco_positions_y, int num_y_positions, float *reco_positions_z,
    int num_z_positions, cuComplex *reco_image, cuComplex *time_signal,
    const int time_signal_length) {
  // int index_x = threadIdx.y + blockIdx.x * blockDim.y;
  // int index_y = blockDim.y;
  // int index_z = blockDim.z;

  int index_x = blockIdx.x;
  int index_y = threadIdx.y + blockIdx.y * blockDim.y;
  int index_z = blockIdx.z;

  if (index_x >= num_x_positions || index_y >= num_y_positions ||
      index_z >= num_z_positions)
    return;

  float3 reco_pos;
  reco_pos.x = reco_positions_x[index_x];
  reco_pos.y = reco_positions_y[index_y];
  reco_pos.z = reco_positions_z[index_z];
  // printf("Thread (%i,%i,%i) Block (%i,%i,%i) -> Position (%i,%i,%i)\n",
  //        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y,
  //        blockIdx.z, index_x, index_y, index_z);

  __shared__ float shared_delay_rx[8][96];
  __shared__ float shared_delay_tx[8][96];

  // 1 warp (32 threads) calculate one point
  // since we have 96 different (RX,point) paths and 96 different (TX, point)
  // paths each thread calculates 3 (Rx,point) and 3 (TX,point)
  shared_delay_rx[threadIdx.y][threadIdx.x] =
      points_dist(reco_pos, rx_antennas[threadIdx.x]);
  shared_delay_rx[threadIdx.y][threadIdx.x + 32] =
      points_dist(reco_pos, rx_antennas[threadIdx.x + 32]);
  shared_delay_rx[threadIdx.y][threadIdx.x + 64] =
      points_dist(reco_pos, rx_antennas[threadIdx.x + 64]);
  shared_delay_tx[threadIdx.y][threadIdx.x] =
      points_dist(reco_pos, tx_antennas[threadIdx.x]);
  shared_delay_tx[threadIdx.y][threadIdx.x + 32] =
      points_dist(reco_pos, tx_antennas[threadIdx.x + 32]);
  shared_delay_tx[threadIdx.y][threadIdx.x + 64] =
      points_dist(reco_pos, tx_antennas[threadIdx.x + 64]);

  cuComplex result = make_cuComplex(1e-3, 1e-3);
  int start_array_index = 0;

  for (int t_index = 0; t_index < time_signal_length; t_index++) {
    float frequency = frequencies[t_index];

    for (int index = threadIdx.x; index < num_rx_antennas * num_tx_antennas;
         index += 32) {
      // we have 96*96 antenna combinations
      // each thread calculates (94*94)/32 antenna combinations
      // with the help of the precomputed (point,RX) and (point,TX) paths
      float delay_rx = shared_delay_rx[threadIdx.y][index / 94];
      float delay = shared_delay_tx[threadIdx.y][index % 94] + delay_rx;
      cuComplex signal =
          time_signal[start_array_index +
                      index]; // time signal order: (frequency, RX, TX)
      // compute cross-correlation and add to result
      result = calculate(result, signal, delay, frequency);
    }

    start_array_index += num_tx_antennas * num_rx_antennas;
  }
  // accumulate result across all threads of a warp
  // -> final cross-correlaion result
  result = warpReduceSum(result);

  if (threadIdx.x == 0) {
    int result_index = index_x * num_z_positions * num_y_positions +
                       index_y * num_z_positions + index_z;
    reco_image[result_index] = result;
    // printf("(%i,%i,%i) Writing into %i\n", index_x, index_y, index_z,
    //        result_index);
  }
}

__global__ void holo_reco_sfcw_time_domain_custom_positions(
    float3 *tx_antennas, int num_tx_antennas, float3 *rx_antennas,
    int num_rx_antennas, float3 *reco_positions, int num_positions,
    cuComplex *reco_image, cuComplex *time_signal, const int time_signal_length,
    float *frequencies) {
  int index_x = threadIdx.x + blockIdx.x * blockDim.x;

  if (index_x >= num_positions)
    return;

  const float c = 299792458.0f;
  float3 reco_pos = reco_positions[index_x];

  cuComplex result = make_cuComplex(1e-3, 1e-3);

  for (int rx_index = 0; rx_index < num_rx_antennas; rx_index++) {
    float3 rx_pos = rx_antennas[rx_index];

    for (int tx_index = 0; tx_index < num_tx_antennas; tx_index++) {
      for (int t_index = 0; t_index < time_signal_length; t_index++) {
        float3 tx_pos = tx_antennas[tx_index];
        float current_frequency = frequencies[t_index];
        float delay =
            (vec_length(reco_pos - tx_pos) + vec_length(reco_pos - rx_pos)) / c;
        float exp_phase = 1.0 * 2 * M_PI * current_frequency * delay;
        cuComplex weight = comp_exp(exp_phase);
        int start_array_index =
            rx_index * num_tx_antennas * time_signal_length +
            tx_index * time_signal_length + t_index;
        cuComplex part_result = time_signal[start_array_index] * weight;

        result.x = result.x + part_result.x;
        result.y = result.y + part_result.y;
      }
    }
  }
  int result_index = index_x;
  reco_image[result_index] = result;
}

__global__ void holo_reco_sfcw_time_domain_custom_positions_noaggregate(
    float3 *tx_antennas, int num_tx_antennas, float3 *rx_antennas,
    int num_rx_antennas, float3 *reco_positions, int num_positions,
    cuComplex *reco_image, cuComplex *time_signal,
    const int time_signal_length) {
  int index_x = threadIdx.y + blockIdx.y * blockDim.y;

  if (index_x >= num_positions)
    return;
  float3 reco_pos = reco_positions[index_x];

  __shared__ float shared_delay_rx[8][96];
  __shared__ float shared_delay_tx[8][96];

  // 1 warp (32 threads) calculate one point
  // since we have 96 different (RX,point) paths and 96 different (TX, point)
  // paths each thread calculates 3 (Rx,point) and 3 (TX,point)
  shared_delay_rx[threadIdx.y][threadIdx.x] =
      points_dist(reco_pos, rx_antennas[threadIdx.x]);
  shared_delay_rx[threadIdx.y][threadIdx.x + 32] =
      points_dist(reco_pos, rx_antennas[threadIdx.x + 32]);
  shared_delay_rx[threadIdx.y][threadIdx.x + 64] =
      points_dist(reco_pos, rx_antennas[threadIdx.x + 64]);
  shared_delay_tx[threadIdx.y][threadIdx.x] =
      points_dist(reco_pos, tx_antennas[threadIdx.x]);
  shared_delay_tx[threadIdx.y][threadIdx.x + 32] =
      points_dist(reco_pos, tx_antennas[threadIdx.x + 32]);
  shared_delay_tx[threadIdx.y][threadIdx.x + 64] =
      points_dist(reco_pos, tx_antennas[threadIdx.x + 64]);

  // int start_array_index = 0;

  for (int index = 0; index < num_rx_antennas * num_tx_antennas; index += 1) {
    int rx_index = index / 94;
    int tx_index = index % 94;

    cuComplex result = make_cuComplex(1e-3, 1e-3);
    for (int t_index = threadIdx.x; t_index < time_signal_length;
         t_index = t_index + 32) {
      float frequency = frequencies[t_index];

      // we have 96*96 antenna combinations
      // each thread calculates (94*94)/32 antenna combinations
      // with the help of the precomputed (point,RX) and (point,TX) paths
      float delay_rx = shared_delay_rx[threadIdx.y][rx_index];
      float delay = shared_delay_tx[threadIdx.y][tx_index] + delay_rx;
      int start_array_index = rx_index * num_tx_antennas * time_signal_length +
                              tx_index * time_signal_length + t_index;

      cuComplex signal =
          time_signal[start_array_index]; // time signal order: (RX, TX,
                                          // frequency) !!!!!

      // compute cross-correlation and add to result
      result = calculate(result, signal, delay, frequency);
    }
    result = warpReduceSum(result);
    result.x -= 31 * 1e-3;
    result.y -= 31 * 1e-3;

    // start_array_index += time_signal_length;
    if (threadIdx.x == 0) {
      reco_image[index_x * num_rx_antennas * num_tx_antennas +
                 rx_index * num_tx_antennas + tx_index] = result;
    }
  }
}

__global__ void holo_reco_sfcw_time_domain_custom_positions_hypotheses(
    float3 *tx_antennas, int num_tx_antennas, float3 *rx_antennas,
    int num_rx_antennas, float3 *reco_positions, int num_positions,
    cuComplex *reco_image, cuComplex *time_signal,
    const int time_signal_length) {
  int index_x = threadIdx.y + blockIdx.y * blockDim.y;

  if (index_x >= num_positions)
    return;
  float3 reco_pos = reco_positions[index_x];

  __shared__ float shared_delay_rx[8][96];
  __shared__ float shared_delay_tx[8][96];

  // 1 warp (32 threads) calculate one point
  // since we have 96 different (RX,point) paths and 96 different (TX, point)
  // paths each thread calculates 3 (Rx,point) and 3 (TX,point)
  shared_delay_rx[threadIdx.y][threadIdx.x] =
      points_dist(reco_pos, rx_antennas[threadIdx.x]);
  shared_delay_rx[threadIdx.y][threadIdx.x + 32] =
      points_dist(reco_pos, rx_antennas[threadIdx.x + 32]);
  shared_delay_rx[threadIdx.y][threadIdx.x + 64] =
      points_dist(reco_pos, rx_antennas[threadIdx.x + 64]);
  shared_delay_tx[threadIdx.y][threadIdx.x] =
      points_dist(reco_pos, tx_antennas[threadIdx.x]);
  shared_delay_tx[threadIdx.y][threadIdx.x + 32] =
      points_dist(reco_pos, tx_antennas[threadIdx.x + 32]);
  shared_delay_tx[threadIdx.y][threadIdx.x + 64] =
      points_dist(reco_pos, tx_antennas[threadIdx.x + 64]);

  // int start_array_index = 0;

  for (int index = 0; index < num_rx_antennas * num_tx_antennas; index += 1) {
    int rx_index = index / 94;
    int tx_index = index % 94;

    cuComplex result = make_cuComplex(1e-3, 1e-3);
    for (int t_index = threadIdx.x; t_index < time_signal_length;
         t_index = t_index + 32) {
      float frequency = frequencies[t_index];

      // we have 96*96 antenna combinations
      // each thread calculates (94*94)/32 antenna combinations
      // with the help of the precomputed (point,RX) and (point,TX) paths
      float delay_rx = shared_delay_rx[threadIdx.y][rx_index];
      float delay = shared_delay_tx[threadIdx.y][tx_index] + delay_rx;
      int start_array_index = rx_index * num_tx_antennas * time_signal_length +
                              tx_index * time_signal_length + t_index;

      cuComplex signal =
          time_signal[start_array_index]; // time signal order: (RX, TX,
                                          // frequency) !!!!!

      // compute cross-correlation and add to result
      float x;
      float y;
      sincosf(frequency * delay, &y, &x);
      result = make_cuComplex(result.x + x, result.y + y);

      // result = calculate(result, signal, delay, frequency);
    }
    result = warpReduceSum(result);
    result.x -= 31 * 1e-3;
    result.y -= 31 * 1e-3;

    // start_array_index += time_signal_length;
    if (threadIdx.x == 0) {
      reco_image[index_x * num_rx_antennas * num_tx_antennas +
                 rx_index * num_tx_antennas + tx_index] = result;
    }
  }
}

__global__ void holo_reco_sfcw_time_domain_custom_positions_signal(
    float3 *tx_antennas, int num_tx_antennas, float3 *rx_antennas,
    int num_rx_antennas, float3 *reco_positions, int num_positions,
    cuComplex *reco_image, cuComplex *time_signal,
    const int time_signal_length) {
  int index_x = threadIdx.y + blockIdx.y * blockDim.y;

  if (index_x >= num_positions)
    return;
  float3 reco_pos = reco_positions[index_x];

  __shared__ float shared_delay_rx[8][96];
  __shared__ float shared_delay_tx[8][96];

  // 1 warp (32 threads) calculate one point
  // since we have 96 different (RX,point) paths and 96 different (TX, point)
  // paths each thread calculates 3 (Rx,point) and 3 (TX,point)
  shared_delay_rx[threadIdx.y][threadIdx.x] =
      points_dist(reco_pos, rx_antennas[threadIdx.x]);
  shared_delay_rx[threadIdx.y][threadIdx.x + 32] =
      points_dist(reco_pos, rx_antennas[threadIdx.x + 32]);
  shared_delay_rx[threadIdx.y][threadIdx.x + 64] =
      points_dist(reco_pos, rx_antennas[threadIdx.x + 64]);
  shared_delay_tx[threadIdx.y][threadIdx.x] =
      points_dist(reco_pos, tx_antennas[threadIdx.x]);
  shared_delay_tx[threadIdx.y][threadIdx.x + 32] =
      points_dist(reco_pos, tx_antennas[threadIdx.x + 32]);
  shared_delay_tx[threadIdx.y][threadIdx.x + 64] =
      points_dist(reco_pos, tx_antennas[threadIdx.x + 64]);

  // int start_array_index = 0;

  for (int index = 0; index < num_rx_antennas * num_tx_antennas; index += 1) {
    int rx_index = index / 94;
    int tx_index = index % 94;

    cuComplex result = make_cuComplex(1e-3, 1e-3);
    for (int t_index = threadIdx.x; t_index < time_signal_length;
         t_index = t_index + 32) {
      float frequency = frequencies[t_index];

      // we have 96*96 antenna combinations
      // each thread calculates (94*94)/32 antenna combinations
      // with the help of the precomputed (point,RX) and (point,TX) paths
      float delay_rx = shared_delay_rx[threadIdx.y][rx_index];
      float delay = shared_delay_tx[threadIdx.y][tx_index] + delay_rx;
      int start_array_index = rx_index * num_tx_antennas * time_signal_length +
                              tx_index * time_signal_length + t_index;

      cuComplex signal =
          time_signal[start_array_index]; // time signal order: (RX, TX,
                                          // frequency) !!!!!

      // compute cross-correlation and add to result

      result = make_cuComplex(result.x + signal.x, result.y + signal.y);

      // result = calculate(result, signal, delay, frequency);
    }
    result = warpReduceSum(result);
    result.x -= 31 * 1e-3;
    result.y -= 31 * 1e-3;

    // start_array_index += time_signal_length;
    if (threadIdx.x == 0) {
      reco_image[index_x * num_rx_antennas * num_tx_antennas +
                 rx_index * num_tx_antennas + tx_index] = result;
    }
  }
}

__global__ void holo_reco_sfcw_time_domain_custom_positions_weighted(
    float3 *tx_antennas, int num_tx_antennas, float3 *rx_antennas,
    int num_rx_antennas, float3 *reco_positions, int num_positions,
    cuComplex *reco_image, cuComplex *time_signal, const int time_signal_length,
    float *frequencies) {
  int index_x = threadIdx.x + blockIdx.x * blockDim.x;

  if (index_x >= num_positions)
    return;

  const float c = 299792458.0f;
  float3 reco_pos = reco_positions[index_x];
  float3 reco_normal = reco_positions[index_x + num_positions];

  cuComplex result = make_cuComplex(1e-3, 1e-3);

  for (int rx_index = 0; rx_index < num_rx_antennas; rx_index++) {
    float3 rx_pos = rx_antennas[rx_index];
    cuComplex rx_result = make_cuComplex(1e-3, 1e-3);
    for (int tx_index = 0; tx_index < num_tx_antennas; tx_index++) {
      for (int t_index = 0; t_index < time_signal_length; t_index++) {
        float3 tx_pos = tx_antennas[tx_index];
        float current_frequency = frequencies[t_index];
        float delay =
            (vec_length(reco_pos - tx_pos) + vec_length(reco_pos - rx_pos)) / c;
        float exp_phase = 1.0 * 2 * M_PI * current_frequency * delay;
        cuComplex weight = comp_exp(exp_phase);
        float3 i = tx_pos - reco_pos;
        float3 r = rx_pos - reco_pos;
        r.x = r.x / vec_length(rx_pos - reco_pos);
        r.y = r.y / vec_length(rx_pos - reco_pos);
        r.z = r.z / vec_length(rx_pos - reco_pos);
        i.x = i.x / vec_length(tx_pos - reco_pos);
        i.y = i.y / vec_length(tx_pos - reco_pos);
        i.z = i.z / vec_length(tx_pos - reco_pos);
        float dot = (i.x * reco_normal.x + i.y * reco_normal.y +
                     i.z * reco_normal.z); // normal = [0,0,1]

        weight.x *= dot;
        weight.y *= dot;

        int start_array_index =
            rx_index * num_tx_antennas * time_signal_length +
            tx_index * time_signal_length + t_index;
        cuComplex part_result = time_signal[start_array_index] * weight;
        result.x = result.x + part_result.x;
        result.y = result.y + part_result.y;
      }
    }
  }
  int result_index = index_x;
  reco_image[result_index] = result;
}
}