/*
 * diompcomm.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIOMP_COMM_H
#define DIOMP_COMM_H

#include <cstdint>
#ifndef GASNET_PAR
#define GASNET_PAR
#endif

#include "diomp.h"
#include "diompmem.h"
#include "tools.h"
#include "omptarget.h"
#include <cstddef>
#include <gasnet.h>
#include <gasnet_mk.h>
#include <gasnet_tools.h>
#include <gasnetex.h>
#include <vector>
#include <memory>

extern std::unique_ptr<diomp::MemoryManager> MemManager;

namespace diomp {

#ifdef DIOMP_ENABLE_CUDA\

class CUDAStreamManager {
private:
  std::vector<cudaStream_t> Streams;
  std::mutex StreamMutex;

public:
  cudaStream_t createStream() {
    std::lock_guard<std::mutex> lock(StreamMutex);
    cudaStream_t Stream;
    cudaStreamCreate(&Stream);
    Streams.push_back(Stream);
    return Stream;
  }

  void synchronizeAll() {
    std::lock_guard<std::mutex> lock(StreamMutex);
    for (auto Stream : Streams) {
      cudaStreamSynchronize(Stream);
    }
  }

  void clearStreams() {
    std::lock_guard<std::mutex> lock(StreamMutex);
    for (auto Stream : Streams) {
      cudaStreamDestroy(Stream);
    }
    Streams.clear();
  }

  ~CUDAStreamManager() {
    for (auto Stream : Streams) {
      cudaStreamDestroy(Stream);
    }
  }
};

#endif

// Base communicator class
class DiOMPCommunicator {
public:
  DiOMPCommunicator();
  virtual ~DiOMPCommunicator() = default;

  // Basic communication operations
  virtual void bcast(void *Data, size_t Size, int Root);
  virtual void allreduce(void *Src, void *Dst, size_t Size, 
                        omp_dt_t Dt, omp_op_t Op);
  virtual void reduce(void *Src, void *Dst, size_t Size,
                     omp_dt_t Dt, omp_op_t Op, int Root);
  virtual void barrier();

  // RMA operations
  virtual void get(void *Dest, int Node, void *Src, size_t Size);
  virtual void put(int Node, void *Dest, void *Src, size_t Size);

  // Device collective operations
  virtual void dbcast(void *Data, size_t Size, omp_device_dt_t Dt, 
                     int Node, int DstId) {
    THROW_ERROR("Device broadcast not supported in base communicator");
  }
  virtual void dallreduce(void *Src, void *Dst, size_t Size,
                         omp_device_dt_t Dt, omp_red_op_t Op, int DstId) {
    THROW_ERROR("Device allreduce not supported in base communicator");
  }
  virtual void dreduce(void *Src, void *Dst, size_t Size,
                      omp_device_dt_t Dt, omp_red_op_t Op, 
                      int Root, int DstId) {
    THROW_ERROR("Device reduce not supported in base communicator");
  }

  // RMA operations for device
  virtual void dget(void *Dest, int Node, void *Src, size_t Size, 
                   int DstId, int SrcId) {
    THROW_ERROR("Device get not supported in base communicator");
  }
  virtual void dput(void *Dest, int Node, void *Src, size_t Size,
                   int DstId, int SrcId) {
    THROW_ERROR("Device put not supported in base communicator");
  }

  // Synchronization operations
  virtual void waitAllRMA();
  virtual void waitRMA(omp_event_t Ev);
  virtual void lock(int Rank);
  virtual void unlock(int Rank);

protected:
  gex_TM_t Team;
  int DevicesNum;
  MemoryManager* Mem;
};

#ifdef OPENMP_ENABLE_DIOMP_DEVICE
// Base device communicator class
class DiOMPDeviceCommunicator : public DiOMPCommunicator {
public:
  DiOMPDeviceCommunicator(int DevicesNum = 1);
  virtual ~DiOMPDeviceCommunicator();

  // Generic device communication interfaces
  virtual void dget(void *Dest, int Node, void *Src, size_t Size, 
                   int DstId, int SrcId) override = 0;
  virtual void dput(void *Dest, int Node, void *Src, size_t Size,
                   int DstId, int SrcId) override = 0;

protected:
  int DevicesNum;
  int Mode;
};

#ifdef DIOMP_ENABLE_CUDA
// CUDA device communicator class
class DiOMPCUDACommunicator : public DiOMPDeviceCommunicator {
public:
  DiOMPCUDACommunicator(int DevicesNum = 1, int Mode = 1);
  ~DiOMPCUDACommunicator();

  // CUDA specific implementations
  void dget(void *Dest, int Node, void *Src, size_t Size,
            int DstId, int SrcId) override;
  void dput(void *Dest, int Node, void *Src, size_t Size,
            int DstId, int SrcId) override;

  void waitAllRMA() override;

  // CUDA collective operations
  void dbcast(void *Data, size_t Size, omp_device_dt_t Dt, 
              int Node, int DstId) override;
  void dallreduce(void *Src, void *Dst, size_t Size,
                  omp_device_dt_t Dt, omp_red_op_t Op, int DstId) override;
  void dreduce(void *Src, void *Dst, size_t Size,
               omp_device_dt_t Dt, omp_red_op_t Op, 
               int Root, int DstId) override;

  // // CUDA Memory Allocator
  static void *cuda_device_alloc(size_t Size, int DeviceId){
    return StaticCudaMem->deviceAlloc(Size, DeviceId);
  }
  static void cuda_device_dealloc(){
    StaticCudaMem->deviceDealloc();
  }
  // static void *diomp_device_alloc(size_t Size, int DeviceId);
  // static void diomp_device_dealloc();

  void initNCCL();

private:

  int LocalRank = 0;

  CUDAStreamManager StreamManager;
  CUDAMemoryManager* CudaMem;
  static CUDAMemoryManager* StaticCudaMem;

  ncclComm_t NcclComm;
  cudaStream_t NcclStream;
  // Per process multiple devices
  cudaStream_t *NcclStreams;
  ncclComm_t *NcclComms;
};
#endif

#ifdef DIOMP_ENABLE_HIP
// HIP device communicator class
class DiOMPHIPCommunicator : public DiOMPDeviceCommunicator {
public:
  DiOMPHIPCommunicator(int devicesNum = 1);
  ~DiOMPHIPCommunicator();

  void deviceBcast(void *data, size_t count, omp_device_dt_t dt, int root,
                  int deviceId);
  void deviceAllreduce(void *src, void *dst, size_t count, omp_device_dt_t dt,
                      omp_red_op_t op, int deviceId);
  void deviceReduce(void *src, void *dst, size_t count, omp_device_dt_t dt,
                   omp_red_op_t op, int root, int deviceId);

  void dget(void *Dest, int Node, void *Src, size_t Size, 
            int DstId, int SrcId) override;
  void dput(void *Dest, int Node, void *Src, size_t Size,
            int DstId, int SrcId) override;

private:
  // HIP specific members
};
#endif

#endif // OPENMP_ENABLE_DIOMP_DEVICE
} // namespace diomp

#endif // DIOMP_COMM_H