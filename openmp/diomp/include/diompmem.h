/*
 * diompmem.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIOMP_MEM_H
#define DIOMP_MEM_H

#include <cstdint>
#ifndef GASNET_PAR
#define GASNET_PAR
#endif

#define OPENMP_ENABLE_DIOMP_DEVICE 1
#include "tools.h"
#include <cstddef>
#include <cstdio>
#include <mutex>

#include <gasnet.h>
#include <gasnet_coll.h>
#include <gasnet_mk.h>
#include <gasnet_tools.h>
#include <gasnetex.h>
#include <vector>

#include <cuda_runtime.h>

extern gex_TM_t diompTeam;
extern gex_Client_t diompClient;
extern gex_EP_t diompEp;
extern gex_Segment_t diompSeg;

struct gex_Seginfo_t {
  void *SegStart;
  void *SegRemain;
  size_t SegSize;
};

struct gex_DeviceSeginfo_t {
  void *SegStart;
  void *SegRemain;
  size_t SegSize;
  cudaIpcMemHandle_t IpcHandle;
};

struct MemoryBlock {
  void *Ptr;
  size_t Size;
};

namespace diomp {

// Base memory manager class
class MemoryManager {
public:
  MemoryManager(gex_TM_t gexTeam, int Mode = 1);
  virtual ~MemoryManager() = default;

  // Basic memory operations
  void *globalAlloc(size_t Size);
  size_t getSegmentSpace(int Rank);
  void *getSegmentAddr(int Rank);
  size_t getAvailableSize();
  size_t getOffset(void *Ptr);
  size_t getOffset(void *Ptr, int Rank);
  bool validGlobalAddr(void *Ptr, int Rank);
  void *convertRemotetoLocalAddr(void *Ptr, int Rank);

  // Device operations with default implementations
  virtual void *deviceAlloc(size_t Size, int DeviceId) {
    THROW_ERROR("Device allocation not supported");
    return nullptr;
  }
  virtual void deviceDealloc() {
    THROW_ERROR("Device deallocation not supported");
  }
  virtual void *getDeviceSegmentAddr(int Rank, int DeviceId) {
    THROW_ERROR("Device segment not supported");
    return nullptr;
  }
  virtual size_t getDeviceOffset(void *Ptr) {
    THROW_ERROR("Device offset not supported");
    return 0;
  }
  virtual void *convertLocaltoRemoteAddr(void *Ptr, int Rank, int DeviceId) {
    THROW_ERROR("Device address conversion not supported");
    return nullptr;
  }
  virtual gex_EP_t getEP(int DeviceId) {
    THROW_ERROR("Device endpoint not supported");
    return nullptr;
  }

protected:
  int RanksNum;
  int MyRank;
  int Mode;
  std::vector<MemoryBlock> MemBlocks;
  std::vector<gex_Seginfo_t> SegInfo;
  void *LocalSegStart;
  void *LocalSegRemain;
  size_t LocalSegSize;
};

#ifdef DIOMP_ENABLE_CUDA
class CUDAMemoryManager : public MemoryManager {
public:
  CUDAMemoryManager(gex_TM_t gexTeam, int Mode = 1);
  ~CUDAMemoryManager() = default;

  void *deviceAlloc(size_t Size, int DeviceId) override;
  void deviceDealloc() override;
  void *getDeviceSegmentAddr(int Rank, int DeviceId) override;
  size_t getDeviceOffset(void *Ptr) override;
  void *convertLocaltoRemoteAddr(void *Ptr, int Rank, int DeviceId);
  void *convertRemotetoLocalAddr(void *Ptr, int Rank, int DeviceId);
  size_t getOffset(void *Ptr, int Rank, int DeviceId);
  gex_EP_t getEP(int DeviceId) override;

  cudaIpcMemHandle_t getIpcHandle(int Rank);

private:
  int LocalRank;
  std::vector<gex_EP_t> DeviceEPs;
  std::vector<cudaIpcMemHandle_t> IpcHandles;
  std::vector<std::vector<gex_DeviceSeginfo_t>> DeviceSegInfo;
  uintptr_t tmpRemain = 0;
};
#endif

#ifdef DIOMP_ENABLE_HIP
class HIPMemoryManager : public MemoryManager {
public:
  HIPMemoryManager(gex_TM_t gexTeam, int Mode = 1);
  ~HIPMemoryManager() = default;

  // Override device operations with empty implementations for now
  void *deviceAlloc(size_t Size, int DeviceId) override { return nullptr; }
  void deviceDealloc() override {}
  void *getDeviceSegmentAddr(int Rank, int DeviceId) override { return nullptr; }
  size_t getDeviceOffset(void *Ptr) override { return 0; }
  void *convertLocaltoRemoteAddr(void *Ptr, int Rank, int DeviceId) override { return nullptr; }
  gex_EP_t getEP(int DeviceId) override { return nullptr; }

private:
  // HIP specific members to be added
};
#endif

} // namespace diomp

#endif // DIOMP_MEM_H
