/*
 * diompcomm.cpp
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "diompcomm.h"

namespace diomp {

// Base communicator implementations
DiOMPCommunicator::DiOMPCommunicator() : Team(diompTeam), DevicesNum(1), Mem(MemManager.get()) {}

void DiOMPCommunicator::bcast(void *Data, size_t Size, int Root) {
  gex_Event_Wait(gex_Coll_BroadcastNB(Team, Root, Data, Data, Size, 0));
}

void DiOMPCommunicator::allreduce(void *Src, void *Dst, size_t Size,
                                 omp_dt_t Dt, omp_op_t Op) {
  gex_Event_Wait(gex_Coll_ReduceToAllNB(Team, Dst, Src, Dt, sizeof(Dt), Size,
                                       Op, NULL, NULL, 0));
}

void DiOMPCommunicator::reduce(void *src, void *dst, size_t count, omp_dt_t dt,
                               omp_op_t op, int root) {
  gex_Event_Wait(gex_Coll_ReduceToOneNB(Team, root, dst, src, dt, sizeof(dt),
                                        count, op, NULL, NULL, 0));
}

void DiOMPCommunicator::barrier() {
  gex_Event_Wait(gex_Coll_BarrierNB(Team, 0));
}

void DiOMPCommunicator::waitAllRMA() {
  gex_NBI_Wait(GEX_EC_ALL, 0);
}

void DiOMPCommunicator::waitRMA(omp_event_t Ev) {
  return;
}

void DiOMPCommunicator::lock(int Rank) {
  return;
}

void DiOMPCommunicator::unlock(int Rank) {
  return;
}

void DiOMPCommunicator::get(void *Dest, int Node, void *Src, size_t Nbytes) {
  auto Error = gex_RMA_GetNBI(Team, Dest, Node, Src, Nbytes, 0);
  if (Error != 0) {
    THROW_ERROR("OpenMP GET Error! Error code is %d", Error);
  }
}

void DiOMPCommunicator::put(int Node, void *Dest, void *Src, size_t Nbytes) {
  auto Error =
      gex_RMA_PutNBI(Team, Node, Dest, Src, Nbytes, GEX_EVENT_DEFER, 0);
  if (Error != 0) {
    THROW_ERROR("OpenMP PUT Error! Error code is %d", Error);
  }
}

#ifdef OPENMP_ENABLE_DIOMP_DEVICE
// Device communicator implementations
DiOMPDeviceCommunicator::DiOMPDeviceCommunicator(int DevicesNum) 
  : DevicesNum(DevicesNum) {}

DiOMPDeviceCommunicator::~DiOMPDeviceCommunicator() = default;

#ifdef DIOMP_ENABLE_CUDA

// CUDA communicator implementations
CUDAMemoryManager* DiOMPCUDACommunicator::StaticCudaMem = nullptr;

DiOMPCUDACommunicator::DiOMPCUDACommunicator(int DevicesNum, int Mode)
    : DiOMPDeviceCommunicator(DevicesNum) {
  CudaMem = dynamic_cast<CUDAMemoryManager*>(Mem);
  StaticCudaMem = CudaMem; 
  this->Mode = Mode;
  if(Mode == 1){
    LocalRank = 0;
    for (int DeviceID = 0; DeviceID < DevicesNum; DeviceID++) {
      omp_target_setup_diompallocator(DeviceID, (void *)diomp_device_alloc,
                                      (void *)diomp_device_dealloc);
    }
  } else {
    LocalRank = omp_get_rank_num() % DevicesNum;
    for (int DeviceID = 0; DeviceID < DevicesNum; DeviceID++) {
      omp_target_setup_diompallocator(DeviceID, (void *)diomp_device_alloc,
                                      (void *)diomp_device_dealloc);
    }
    omp_set_default_device(LocalRank);
  }

  if (!CudaMem) {
    THROW_ERROR("Memory manager is not a CUDA memory manager");
  }
  if (DevicesNum > 1) {
    NcclStreams = new cudaStream_t[DevicesNum];
    NcclComms = new ncclComm_t[DevicesNum];
  }
}

DiOMPCUDACommunicator::~DiOMPCUDACommunicator() {
  if (DevicesNum > 1) {
    delete[] NcclStreams;
    delete[] NcclComms;
    
  }
}

void DiOMPCUDACommunicator::initNCCL() {
  ncclUniqueId NcclId;
  if (omp_get_rank_num() == 0) {
    ncclGetUniqueId(&NcclId);
  }

  gex_Event_Wait(
      gex_Coll_BroadcastNB(Team, 0, &NcclId, &NcclId, sizeof(NcclId), 0));

  if (DevicesNum == 1) {
    CUDACHECK(cudaSetDevice(LocalRank));
    NCCLCHECK(ncclCommInitRank(&NcclComm, omp_get_num_ranks(), NcclId,
                              omp_get_rank_num()));
    CUDACHECK(cudaStreamCreate(&NcclStream));
  } else {
    NCCLCHECK(ncclGroupStart());
    for (int DeviceId = 0; DeviceId < DevicesNum; DeviceId++) {
      CUDACHECK(cudaSetDevice(DeviceId));
      NCCLCHECK(ncclCommInitRank(&NcclComms[DeviceId],
                                omp_get_num_ranks() * DevicesNum, NcclId,
                                omp_get_rank_num() * DevicesNum + DeviceId));
      CUDACHECK(cudaStreamCreate(&NcclStreams [DeviceId]));
    }
    NCCLCHECK(ncclGroupEnd());
  }
}

void DiOMPCUDACommunicator::waitAllRMA(){
  gex_NBI_Wait(GEX_EC_ALL, 0); 
  StreamManager.synchronizeAll();
  StreamManager.clearStreams();
}

void DiOMPCUDACommunicator::dget(void *Dest, int Node, void *Src, size_t Size,
                                int DstId, int SrcId) {
  if (Mode != 1) {
    int TotalDevices = omp_get_num_devices();

    auto LocalEP = CudaMem->getEP(0);
    gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(LocalEP);
    gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);
    void *SrcR = CudaMem->convertLocaltoRemoteAddr(Src, Node, 0);
    if (omp_get_rank_num() / TotalDevices == Node / TotalDevices) {
      int srcDevice = Node % TotalDevices;
      int dstDevice = LocalRank;

      cudaStream_t stream = StreamManager.createStream();

      cudaIpcMemHandle_t IpcHandle = CudaMem->getIpcHandle(Node);
      void *device_ptr = nullptr;
      CUDACHECK(cudaIpcOpenMemHandle(&device_ptr, IpcHandle,
                                 cudaIpcMemLazyEnablePeerAccess));

      size_t offset = CudaMem->getOffset(SrcR, Node, 0);
      char* adjusted_device_ptr = static_cast<char*>(device_ptr) + offset;
      CUDACHECK(
          cudaMemcpyPeerAsync(Dest, dstDevice, adjusted_device_ptr, srcDevice, Size, stream));

      return;
    }
    auto Error = gex_RMA_GetNBI(CommTM, Dest, Node, SrcR, Size, 0);
    if (Error != 0) {
      THROW_ERROR("OpenMP Device Get Error! Error code is %d", Error);
    }
    return;
  }
  auto LocalEP = CudaMem->getEP(DstId);
  auto RemoteEP = CudaMem->getEP(SrcId);

  gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(RemoteEP);
  gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);

  void *SrcR = CudaMem->convertLocaltoRemoteAddr(Src, Node, SrcId);
  auto Error = gex_RMA_GetNBI(CommTM, Dest, Node, SrcR, Size, 0);
  if (Error != 0) {
    THROW_ERROR("OpenMP Device Get Error! Error code is %d", Error);
  }

  return;
}

void DiOMPCUDACommunicator::dput(void *Dest, int Node, void *Src, size_t Size,
                                int DstId, int SrcId) {

}

void DiOMPCUDACommunicator::dbcast(void *Data, size_t Size, omp_device_dt_t Dt,
                                  int Node, int DstId) {
  if (DevicesNum == 1) {
    CUDACHECK(cudaSetDevice(DstId));
    NCCLCHECK(ncclBcast(Data, Size, (ncclDataType_t)Dt, Node, NcclComm,
                        NcclStream));
    CUDACHECK(cudaStreamSynchronize(NcclStream));
    return;
  }

  NCCLCHECK(ncclGroupStart());
  for (int I = 0; I < DevicesNum; I++) {
    void *RemoteData = CudaMem->convertLocaltoRemoteAddr(Data, omp_get_rank_num(), DstId);
    NCCLCHECK(ncclBcast(RemoteData, Size, (ncclDataType_t)Dt,
                        Node * DevicesNum + DstId, NcclComms[I],
                        NcclStreams[I]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int I = 0; I < DevicesNum; I++) {
    CUDACHECK(cudaStreamSynchronize(NcclStreams[I]));
  }
}

void DiOMPCUDACommunicator::dallreduce(void *Src, void *Dst, size_t Size,
                                      omp_device_dt_t Dt, omp_red_op_t Op,
                                      int DstId) {
  if (DevicesNum == 1) {
    CUDACHECK(cudaSetDevice(LocalRank));
    NCCLCHECK(ncclAllReduce(Src, Dst, Size, (ncclDataType_t)Dt,
                            (ncclRedOp_t)Op, NcclComm, NcclStream));
    CUDACHECK(cudaStreamSynchronize(NcclStream));
    return;
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < DevicesNum; i++) {
    NCCLCHECK(ncclAllReduce(Src, Dst, Size, (ncclDataType_t)Dt,
                            (ncclRedOp_t)Op, NcclComms[i], NcclStreams[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < DevicesNum; i++) {
    CUDACHECK(cudaStreamSynchronize(NcclStreams[i]));
  }
}

void DiOMPCUDACommunicator::dreduce(void *Src, void *Dst, size_t Size,
                                   omp_device_dt_t Dt, omp_red_op_t Op,
                                   int Root, int DstId) {
  if (DevicesNum == 1) {
    CUDACHECK(cudaSetDevice(DstId));
    NCCLCHECK(ncclReduce(Src, Dst, Size, (ncclDataType_t)Dt, (ncclRedOp_t)Op,
                         Root, NcclComm, NcclStream));
    CUDACHECK(cudaStreamSynchronize(NcclStream));
    return;
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < DevicesNum; i++) {
    NCCLCHECK(ncclReduce(Src, Dst, Size, (ncclDataType_t)Dt, (ncclRedOp_t)Op,
                         Root * DevicesNum + DstId, NcclComms[i],
                         NcclStreams[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < DevicesNum; i++) {
    CUDACHECK(cudaStreamSynchronize(NcclStreams[i]));
  }
}

void *diomp_device_alloc(size_t Size, int DeviceId) {
  return DiOMPCUDACommunicator::cuda_device_alloc(Size, DeviceId);
}

void diomp_device_dealloc() {
  DiOMPCUDACommunicator::cuda_device_dealloc();
  return;
}

#endif // DIOMP_ENABLE_CUDA

#ifdef DIOMP_ENABLE_HIP
// HIP communicator implementations (empty for now)
DiOMPHIPCommunicator::DiOMPHIPCommunicator(int DevicesNum)
    : DiOMPDeviceCommunicator(DevicesNum) {}

DiOMPHIPCommunicator::~DiOMPHIPCommunicator() = default;

void DiOMPHIPCommunicator::dget(void *Dest, int Node, void *Src, size_t Size,
                               int DstId, int SrcId) {
  // To be implemented
}

void DiOMPHIPCommunicator::dput(void *Dest, int Node, void *Src, size_t Size,
                               int DstId, int SrcId) {
  // To be implemented
}
#endif // DIOMP_ENABLE_HIP

#endif // OPENMP_ENABLE_DIOMP_DEVICE

} // namespace diomp