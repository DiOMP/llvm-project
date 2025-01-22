#ifndef TOOLS_H
#define TOOLS_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef DIOMP_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#define THROW_ERROR(format, ...)                                               \
  do {                                                                         \
    fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);                            \
    fprintf(stderr, format, ##__VA_ARGS__);                                    \
    fprintf(stderr, "\n");                                                     \
    exit(EXIT_FAILURE);                                                        \
  } while (0)

#define GASNET_Safe(fncall)                                                    \
  do {                                                                         \
    int _retval;                                                               \
    if ((_retval = fncall) != GASNET_OK) {                                     \
      fprintf(stderr,                                                          \
              "ERROR calling: %s\n"                                            \
              " at: %s:%i\n"                                                   \
              " error: %s (%s)\n",                                             \
              #fncall, __FILE__, __LINE__, gasnet_ErrorName(_retval),          \
              gasnet_ErrorDesc(_retval));                                      \
      fflush(stderr);                                                          \
      gasnet_exit(_retval);                                                    \
    }                                                                          \
  } while (0)

#ifdef DIOMP_ENABLE_CUDA
#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUCHECK(cmd)                                                           \
  do {                                                                         \
    CUresult err = cmd;                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *errorStr;                                                    \
      cuGetErrorName(err, &errorStr);                                          \
      printf("Failed: CUresult error %s:%d '%s'\n", __FILE__, __LINE__,        \
             errorStr);                                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

#define AM_ACK_LOCK 129
#define AM_LOCK_REQ_IDX 200
#define AM_LOCK_REL_IDX 201
#define AM_PTR_REQ 202
#define AM_REPLY_HIDX 203

#endif
