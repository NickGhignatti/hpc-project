/***
% Circles intersection
% Nicolò Ghignatti <nicolo.ghignatti@studio.unibo.it> 0001028531

This is a parallel implementation of the circle intersection program
described in the specification, using CUDA.

To compile:

        nvcc cudacircles.cu -o cudacircles -lm

To execute:

        ./cudacircles [ncircles [iterations]]

where `ncircles` is the number of circles, and `iterations` is the
number of iterations to execute.
***/
#include "hpc.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLKDIM 32

typedef struct {
  float x, y;   /* coordinates of center */
  float r;      /* radius */
  float dx, dy; /* displacements due to interactions with other circles */
} circle_t;

/* These constants can be replaced with #define's if necessary */
const float XMIN = 0.0;
const float XMAX = 1000.0;
const float YMIN = 0.0;
const float YMAX = 1000.0;
const float RMIN = 10.0;
const float RMAX = 100.0;
const float EPSILON = 1e-5;
const float K = 1.5;

int ncircles;
circle_t *circles = NULL;

/**
 * Return a random float in [a, b]
 */
float randab(float a, float b) {
  return a + (((float)rand()) / RAND_MAX) * (b - a);
}

/**
 * Create and populate the array `circles[]` with randomly placed
 * circls.
 *
 * Do NOT parallelize this function.
 */
void init_circles(int n) {
  assert(circles == NULL);
  ncircles = n;
  circles = (circle_t *)malloc(n * sizeof(*circles));
  assert(circles != NULL);
  for (int i = 0; i < n; i++) {
    circles[i].x = randab(XMIN, XMAX);
    circles[i].y = randab(YMIN, YMAX);
    circles[i].r = randab(RMIN, RMAX);
    circles[i].dx = circles[i].dy = 0.0;
  }
}

/**
 * Set all displacements to zero.
 */
__global__ void reset_displacements_kernel(circle_t *circles,
                                           int *num_circles) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < *num_circles) {
    circles[i].dx = circles[i].dy = 0.0;
  }
}

/**
 * Compute the force acting on each circle; returns the number of
 * overlapping pairs of circles (each overlapping pair must be counted
 * only once).
 */
__global__ void compute_calculus(int *info, float *constant,
                                 circle_t *dcircles) {
  // calculating the coordinates
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  // check if the coordinates are valid, to avoid out of bounds error
  if (i < info[1] && j < info[1] && i < j) {
    const float deltax = dcircles[j].x - dcircles[i].x;
    const float deltay = dcircles[j].y - dcircles[i].y;
    const float dist = hypotf(deltax, deltay);
    const float Rsum = dcircles[i].r + dcircles[j].r;
    if (dist < Rsum - constant[0]) {
      atomicAdd(&info[0], 1);
      const float overlap = Rsum - dist;
      assert(overlap > 0.0);
      // avoid division by zero
      const float overlap_x = overlap / (dist + constant[0]) * deltax;
      const float overlap_y = overlap / (dist + constant[0]) * deltay;
      atomicAdd(&dcircles[i].dx, -overlap_x / constant[1]);
      atomicAdd(&dcircles[i].dy, -overlap_y / constant[1]);
      atomicAdd(&dcircles[j].dx, overlap_x / constant[1]);
      atomicAdd(&dcircles[j].dy, overlap_y / constant[1]);
    }
  }
}

/**
 * Move the circles to a new position according to the forces acting
 * on each one.
 */
__global__ void move_circles_kernel(circle_t *circles, int *num_circles) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < *num_circles) {
    atomicAdd(&circles[i].x, circles[i].dx);
    atomicAdd(&circles[i].y, circles[i].dy);
  }
}

#ifdef MOVIE
/**
 * Dumps the circles into a text file that can be processed using
 * gnuplot. This function may be used for debugging purposes, or to
 * produce a movie of how the algorithm works.
 *
 * You may want to completely remove this function from the final
 * version.
 */
void dump_circles(int iterno) {
  char fname[64];
  snprintf(fname, sizeof(fname), "circles-%05d.gp", iterno);
  FILE *out = fopen(fname, "w");
  const float WIDTH = XMAX - XMIN;
  const float HEIGHT = YMAX - YMIN;
  fprintf(out, "set term png notransparent large\n");
  fprintf(out, "set output \"circles-%05d.png\"\n", iterno);
  fprintf(out, "set xrange [%f:%f]\n", XMIN - WIDTH * .2, XMAX + WIDTH * .2);
  fprintf(out, "set yrange [%f:%f]\n", YMIN - HEIGHT * .2, YMAX + HEIGHT * .2);
  fprintf(out, "set size square\n");
  fprintf(out, "plot '-' with circles notitle\n");
  for (int i = 0; i < ncircles; i++) {
    fprintf(out, "%f %f %f\n", circles[i].x, circles[i].y, circles[i].r);
  }
  fprintf(out, "e\n");
  fclose(out);
}
#endif

int main(int argc, char *argv[]) {
  int n = 10000;
  int iterations = 20;

  if (argc > 3) {
    fprintf(stderr, "Usage: %s [ncircles [iterations]]\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (argc > 1) {
    n = atoi(argv[1]);
  }

  if (argc > 2) {
    iterations = atoi(argv[2]);
  }

  init_circles(n);
  const double tstart_prog = hpc_gettime();
#ifdef MOVIE
  dump_circles(0);
#endif

  int *device_ncircles;
  circle_t *device_circles;

  // allocating the space on the device to store the variables
  cudaSafeCall(cudaMalloc((void **)&device_ncircles, sizeof(int)));
  cudaSafeCall(cudaMemcpy(device_ncircles, &ncircles, sizeof(int),
                          cudaMemcpyHostToDevice));
  cudaSafeCall(
      cudaMalloc((void **)&device_circles, sizeof(circle_t) * ncircles));
  cudaSafeCall(cudaMemcpy(device_circles, circles, sizeof(circle_t) * ncircles,
                          cudaMemcpyHostToDevice));

  int n_intersections = 0;

  int arr[] = {n_intersections, ncircles};
  int *device_info; // = (int *)(malloc(sizeof(int) * 2));
  float const_arr[] = {EPSILON, K};
  float *device_const; // = (float *)(malloc(sizeof(float) * 2));

  // allocating the space on the device to store the variables
  cudaSafeCall(cudaMalloc((void **)&device_info, sizeof(int) * 2));
  cudaSafeCall(cudaMalloc((void **)&device_const, sizeof(float) * 2));
  cudaSafeCall(
      cudaMemcpy(device_info, arr, sizeof(int) * 2, cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpy(device_const, const_arr, sizeof(float) * 2,
                          cudaMemcpyHostToDevice));

  // calculating the dimension of the grid and the block, to use the
  // bi-dimensionality of CUDA threads
  dim3 grid3((ncircles + BLKDIM - 1) / BLKDIM,
             (ncircles + BLKDIM - 1) / BLKDIM);
  dim3 block3(BLKDIM, BLKDIM);

  for (int it = 0; it < iterations; it++) {
    const double tstart_iter = hpc_gettime();
    // resetting the displacement
    reset_displacements_kernel<<<(ncircles + BLKDIM - 1) / BLKDIM, BLKDIM>>>(
        device_circles, device_ncircles);
    cudaSafeCall(cudaMemcpy(circles, device_circles,
                            sizeof(circle_t) * ncircles,
                            cudaMemcpyDeviceToHost));
    // computing the intersections
    compute_calculus<<<grid3, block3>>>(device_info, device_const,
                                        device_circles);
    cudaCheckError();

    cudaSafeCall(cudaMemcpy(circles, device_circles,
                            sizeof(circle_t) * ncircles,
                            cudaMemcpyDeviceToHost));
    cudaSafeCall(
        cudaMemcpy(arr, device_info, sizeof(int) * 2, cudaMemcpyDeviceToHost));
    // and assigning it
    const int n_overlaps = arr[0];

    // moving circles
    move_circles_kernel<<<(ncircles + BLKDIM - 1) / BLKDIM, BLKDIM>>>(
        device_circles, device_ncircles);
    cudaSafeCall(cudaMemcpy(circles, device_circles,
                            sizeof(circle_t) * ncircles,
                            cudaMemcpyDeviceToHost));
    const double elapsed_iter = hpc_gettime() - tstart_iter;
#ifdef MOVIE
    dump_circles(it + 1);
#endif
    printf("Iteration %d of %d, %d overlaps (%f s)\n", it + 1, iterations,
           n_overlaps, elapsed_iter);
  }

  // Free all the memory
  cudaFree(device_circles);
  cudaFree(device_ncircles);
  cudaFree(device_const);
  cudaFree(device_info);

  const double elapsed_prog = hpc_gettime() - tstart_prog;
  printf("Elapsed time: %f\n", elapsed_prog);

  free(circles);

  return EXIT_SUCCESS;
}
