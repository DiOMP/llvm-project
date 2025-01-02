/*
 * diomp.cpp
 */
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "diomp.hpp"
#include "diompmem.h"
#include "tools.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "omptarget.h"
#include <omp.h>

std::unique_ptr<diomp::MemoryManager> MemManager;
gex_TM_t diompTeam;
gex_Client_t diompClient;
gex_EP_t diompEp;
gex_Segment_t diompSeg;
std::atomic<size_t> SegSize{0};
std::atomic<int> LockState{0};
std::vector<int> LockQueues;
std::mutex lockQueueMutex;
std::atomic<int> flag_lock{0}; // 0: initial, 1: locked, 2: occupied

#ifdef DIOMP_ENABLE_CUDA

ncclComm_t NcclComm;
cudaStream_t NcclStream;

// Per process multiple devices

cudaStream_t *NcclStreams;
ncclComm_t *NcclComms;

void CUDACHECK(cudaError_t result, const char *msg) {
  if (result != cudaSuccess) {
    THROW_ERROR("CUDA Error: %s", cudaGetErrorString(result));
    exit(EXIT_FAILURE);
  }
}

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#endif

size_t convertToBytes(const std::string &sizeStr) {
  if (sizeStr.empty()) {
    throw std::invalid_argument("Input string cannot be empty.");
  }

  size_t lastDigitIndex = sizeStr.find_last_of("0123456789");
  if (lastDigitIndex == std::string::npos) {
    throw std::invalid_argument("No numeric part found in the input string.");
  }

  std::string numberPart = sizeStr.substr(0, lastDigitIndex + 1);
  std::string unitPart = sizeStr.substr(lastDigitIndex + 1);

  size_t number = std::stoul(numberPart);

  std::transform(unitPart.begin(), unitPart.end(), unitPart.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  size_t bytes = 0;
  if (unitPart == "kb") {
    if (number > std::numeric_limits<size_t>::max() / 1024) {
      throw std::overflow_error(
          "Value too large to be represented in bytes as size_t.");
    }
    bytes = number * 1024;
  } else if (unitPart == "mb") {
    if (number > std::numeric_limits<size_t>::max() / (1024 * 1024)) {
      throw std::overflow_error(
          "Value too large to be represented in bytes as size_t.");
    }
    bytes = number * 1024 * 1024;
  } else if (unitPart == "gb") {
    if (number > std::numeric_limits<size_t>::max() / (1024 * 1024 * 1024)) {
      throw std::overflow_error(
          "Value too large to be represented in bytes as size_t.");
    }
    bytes = number * 1024 * 1024 * 1024;
  } else {
    bytes = number;
  }
  return bytes;
}

size_t getOMPDistributedSize() {
  const char *envValue = std::getenv("OMP_DISTRIBUTED_MEM_SIZE");
  if (envValue == nullptr) {
    size_t bytes = 16ULL * 1024 * 1024 * 1024; // Default: 16GB
    return bytes;
  }
  std::string valueStr(envValue);
  size_t bytes = convertToBytes(valueStr);
  return bytes;
}

// AM Handlers for DiOMP
void AcknowledgeLockAM(gex_Token_t Token, gex_AM_Arg_t Arg0) {
  flag_lock = Arg0;
  return;
}

void LockRequestAM(gex_Token_t Token) {
  gex_Token_Info_t info;
  int sourceRank = 0;
  gex_TI_t result = gex_Token_Info(Token, &info, GEX_TI_SRCRANK);
  if (result & GEX_TI_SRCRANK) {
    sourceRank = (int)info.gex_srcrank;
  }
  int expected = 0;
  if (LockState.compare_exchange_strong(expected, 1) && LockQueues.empty()) {
    gex_AM_ReplyShort1(Token, AM_ACK_LOCK, 0, 1);
  } else {
    std::lock_guard<std::mutex> lock(lockQueueMutex);
    LockQueues.push_back(sourceRank);
  }
}

void LockReleaseAM(gex_Token_t Token) {
  LockState.store(0);
  int nextRank = -1;
  {
    std::lock_guard<std::mutex> lock(lockQueueMutex);
    if (!LockQueues.empty()) {
      nextRank = LockQueues.front();
      LockQueues.erase(LockQueues.begin());
    }
  }
  if (nextRank != -1) {
    gex_AM_RequestShort1(diompTeam, nextRank, AM_ACK_LOCK, 0, 1);
    LockState.store(1);
  }
}

// AM Handler Table
gex_AM_Entry_t AMTable[] = {{AM_ACK_LOCK, (gex_AM_Fn_t)AcknowledgeLockAM,
                             GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQREP, 0},
                            {AM_LOCK_REQ_IDX, (gex_AM_Fn_t)LockRequestAM,
                             GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST, 1},
                            {AM_LOCK_REL_IDX, (gex_AM_Fn_t)LockReleaseAM,
                             GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST, 0}};

// Initialize DIOMP runtime
void __init_diomp() {
  gex_Client_Init(&diompClient, &diompEp, &diompTeam, "diomp", nullptr, nullptr,
                  0);
  if (SegSize == 0)
    SegSize = getOMPDistributedSize();
  GASNET_Safe(gex_Segment_Attach(&diompSeg, diompTeam, SegSize));
  gex_EP_RegisterHandlers(diompEp, AMTable,
                          sizeof(AMTable) / sizeof(gex_AM_Entry_t));
  MemManager = std::make_unique<diomp::MemoryManager>(diompTeam);
}

void __init_diomp_target() {
  gex_Client_Init(&diompClient, &diompEp, &diompTeam, "diomp", nullptr, nullptr,
                  0);
  if (SegSize == 0)
    SegSize = getOMPDistributedSize();
  GASNET_Safe(gex_Segment_Attach(&diompSeg, diompTeam, SegSize));
  gex_EP_RegisterHandlers(diompEp, AMTable,
                          sizeof(AMTable) / sizeof(gex_AM_Entry_t));
  MemManager = std::make_unique<diomp::MemoryManager>(diompTeam);

  // Setup DiOMP Allocator
  for (int DeviceID = 0; DeviceID < omp_get_num_devices(); DeviceID++) {
    omp_target_setup_diompallocator(DeviceID, (void *)diomp_device_alloc,
                                    (void *)diomp_device_dealloc);
  }

#ifdef DIOMP_ENABLE_CUDA
  ncclUniqueId ncclID;
  int DevicesNum = omp_get_num_devices();
  if (DevicesNum > 1) {
    NcclStreams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * DevicesNum);
    NcclComms = (ncclComm_t *)malloc(sizeof(ncclComm_t) * DevicesNum);
  }

  if (omp_get_rank_num() == 0)
    ncclGetUniqueId(&ncclID);

  // Broadcast NCCL unique ID
  gex_Event_Wait(
      gex_Coll_BroadcastNB(diompTeam, 0, &ncclID, &ncclID, sizeof(ncclID), 0));

  // Initialize NCCL communicator & CUDA stream
  if (DevicesNum == 1) {
    CUDACHECK(cudaSetDevice(0), "Setting CUDA device");
    NCCLCHECK(ncclCommInitRank(&NcclComm, omp_get_num_ranks(), ncclID,
                               omp_get_rank_num()));
    CUDACHECK(cudaStreamCreate(&NcclStream), "Creating CUDA stream");
  } else {
    NCCLCHECK(ncclGroupStart());
    // ncclCommInitRank(comms+i, nRanks*nDev, id, myRank*nDev + i)
    for (int DeviceID = 0; DeviceID < DevicesNum; DeviceID++) {
      CUDACHECK(cudaSetDevice(DeviceID), "Setting CUDA device");
      NCCLCHECK(ncclCommInitRank(&NcclComms[DeviceID],
                                 omp_get_num_ranks() * DevicesNum, ncclID,
                                 omp_get_rank_num() * DevicesNum + DeviceID));
      CUDACHECK(cudaStreamCreate(&NcclStreams[DeviceID]),
                "Creating CUDA stream");
    }
    NCCLCHECK(ncclGroupEnd());
  }
  printf("inited!\n");

#endif
}

void *diomp_device_alloc(size_t Size, int DeviceId) {
  void *MemAddr = nullptr;
  MemAddr = MemManager->deviceAlloc(Size, DeviceId);
  return MemAddr;
}

void diomp_device_dealloc() {
  MemManager->deviceDealloc();
  return;
}

void omp_set_distributed_size(size_t Size) { SegSize = Size; }

// Get the total number of ranks
int omp_get_num_ranks() { return gex_TM_QuerySize(diompTeam); }

// Get the rank number of the current process
int omp_get_rank_num() { return gex_TM_QueryRank(diompTeam); }

// Get the starting address of the memory segment on a node
void *omp_get_space(int node) { return MemManager->getSegmentAddr(node); }

// Get the size of the memory segment on a node
uintptr_t omp_get_length_space(int node) {
  return MemManager->getSegmentSpace(node);
}

// Allocate shared memory accessible by all nodes
void *llvm_omp_distributed_alloc(size_t Size) {
  return MemManager->globalAlloc(Size);
}

// Templated version to allocate typed shared memory
template <typename T> void *diomp_alloc(size_t Size) {
  return MemManager->globalAlloc(Size * sizeof(T));
}

// RMA Operations
// Get data from a remote node
void ompx_get(void *dest, int node, void *src, size_t nbytes) {
  auto Error = gex_RMA_GetNBI(diompTeam, dest, node, src, nbytes, 0);
  if (Error != 0) {
    THROW_ERROR("OpenMP GET Error! Error code is %d", Error);
  }
}

// Put data to a remote node
void ompx_put(int node, void *dest, void *src, size_t nbytes) {
  auto Error =
      gex_RMA_PutNBI(diompTeam, node, dest, src, nbytes, GEX_EVENT_DEFER, 0);
  if (Error != 0) {
    THROW_ERROR("OpenMP PUT Error! Error code is %d", Error);
  }
}

void get_offset(void *Ptr) {
  size_t Offset = MemManager->getDeviceOffset(Ptr);
  printf("Offset: %llu\n", Offset);
}

void ompx_dget(void *dest, int node, void *src, size_t nbytes, int dst_id,
               int src_id) {
  auto LocalEP = MemManager->getEP(dst_id);
  auto RemoteEP = MemManager->getEP(src_id);

  gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(RemoteEP);
  gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);

  void *SrcR = MemManager->convertLocaltoRemoteAddr(src, node, src_id);
  auto Error = gex_RMA_GetNBI(CommTM, dest, node, SrcR, nbytes, 0);
  if (Error != 0) {
    THROW_ERROR("OpenMP Device PUT Error! Error code is %d", Error);
  }

  return;
}

void ompx_dput(void *dest, int node, void *src, size_t nbytes, int dst_id,
               int src_id) {
  auto LocalEP = MemManager->getEP(src_id);
  auto RemoteEP = MemManager->getEP(dst_id);

  gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(RemoteEP);
  gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);

  void *DestR = MemManager->convertLocaltoRemoteAddr(dest, node, dst_id);
  // printf("DestR %p\n", DestR);
  auto Error =
      gex_RMA_PutNBI(CommTM, node, DestR, src, nbytes, GEX_EVENT_DEFER, 0);
  if (Error != 0) {
    THROW_ERROR("OpenMP DPUT Error! Error code is %d", Error);
  }
  return;
}

// End of RMA Operations

// Synchronization Operations
// Barrier synchronization across all nodes
void diomp_barrier() { gex_Event_Wait(gex_Coll_BarrierNB(diompTeam, 0)); }

// Wait for completion of all RMA operations
void diomp_waitALLRMA() { gex_NBI_Wait(GEX_EC_ALL, 0); }

// Wait for completion of a specific RMA operation
void diomp_waitRMA(omp_event_t ev) { gex_NBI_Wait(ev, 0); }

void diomp_lock(int Rank) {
  gex_AM_RequestShort0(diompTeam, Rank, AM_LOCK_REQ_IDX, 0);
  GASNET_BLOCKUNTIL(flag_lock == 1);
}

void diomp_unlock(int Rank) {
  gex_AM_RequestShort0(diompTeam, Rank, AM_LOCK_REL_IDX, 0);
}

// End of Synchronization Operations

// Collective Operations
// Broadcast data from root to all nodes
void omp_bcast(void *data, size_t nbytes, int node) {
  gex_Event_Wait(gex_Coll_BroadcastNB(diompTeam, node, data, data, nbytes, 0));
}

// All-reduce operation across all nodes
void omp_allreduce(void *src, void *dst, size_t count, omp_dt_t dt,
                   omp_op_t op) {
  gex_Event_Wait(gex_Coll_ReduceToAllNB(diompTeam, dst, src, dt, sizeof(dt),
                                        count, op, NULL, NULL, 0));
}

void omp_reduce(void *src, void *dst, size_t count, omp_dt_t dt, omp_op_t op,
                int root) {
  gex_Event_Wait(gex_Coll_ReduceToOneNB(diompTeam, root, dst, src, dt,
                                        sizeof(dt), count, op, NULL, NULL, 0));
}

#ifdef DIOMP_ENABLE_CUDA

void ompx_dbcast(void *data, size_t count, omp_device_dt_t dt, int node,
                 int dst_id) {
  int DevicesNum = omp_get_num_devices();
  if(DevicesNum == 1){
    CUDACHECK(cudaSetDevice(dst_id), "Setting CUDA device");
    NCCLCHECK(
        ncclBcast(data, count, (ncclDataType_t)dt, node, NcclComm, NcclStream));
    CUDACHECK(cudaStreamSynchronize(0), "Synchronizing stream");
    return;
  }
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < DevicesNum; i++){
    void *tmp = MemManager->convertLocaltoRemoteAddr(data, omp_get_rank_num(), dst_id);
    NCCLCHECK(
        ncclBcast(tmp, count, (ncclDataType_t)dt, node * DevicesNum + dst_id, NcclComms[i], NcclStreams[i]));
  }
  NCCLCHECK(ncclGroupEnd());
  for (int i = 0; i < DevicesNum; i++){
    CUDACHECK(cudaStreamSynchronize(NcclStreams[i]), "Synchronizing streams");
  }
}

void ompx_dallreduce(void *src, void *dst, size_t count, omp_device_dt_t dt,
                     omp_red_op_t op, int dst_id) {
  CUDACHECK(cudaSetDevice(dst_id), "Setting CUDA device");
  NCCLCHECK(ncclAllReduce(src, dst, count, (ncclDataType_t)dt, (ncclRedOp_t)op,
                          NcclComm, NcclStream));
  CUDACHECK(cudaStreamSynchronize(0), "Synchronizing stream");
}

void ompx_dreduce(void *src, void *dst, size_t count, omp_device_dt_t dt,
                  omp_red_op_t op, int root, int dst_id) {
  CUDACHECK(cudaSetDevice(dst_id), "Setting CUDA device");
  NCCLCHECK(ncclReduce(src, dst, count, (ncclDataType_t)dt, (ncclRedOp_t)op,
                       root, NcclComm, NcclStream));
  CUDACHECK(cudaStreamSynchronize(0), "Synchronizing stream");
}

#endif
