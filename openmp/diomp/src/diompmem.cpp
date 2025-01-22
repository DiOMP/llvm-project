#include "diompmem.h"
#include <cstddef>

namespace diomp {

// Base MemoryManager implementation
MemoryManager::MemoryManager(gex_TM_t gexTeam, int Mode) {
  RanksNum = gex_TM_QuerySize(gexTeam);
  MyRank = gex_TM_QueryRank(gexTeam);
  this->Mode = Mode;
  
  // Initialize segment info
  SegInfo.resize(RanksNum);
  for (auto &Seg : SegInfo) {
    void *SegBase = 0;
    size_t SegSize = 0;
    gex_Event_Wait(gex_EP_QueryBoundSegmentNB(diompTeam, &Seg - &SegInfo[0],
                                             &SegBase, nullptr, &SegSize, 0));
    Seg.SegStart = SegBase;
    Seg.SegSize = SegSize;
    Seg.SegRemain = SegBase;
  }
  
  LocalSegStart = SegInfo[MyRank].SegStart;
  LocalSegRemain = SegInfo[MyRank].SegRemain;
  LocalSegSize = SegInfo[MyRank].SegSize;
}

void *MemoryManager::globalAlloc(size_t Size) {
  if (Size > getAvailableSize()) {
    return nullptr;
  }

  void *Ptr = LocalSegRemain;
  LocalSegRemain = reinterpret_cast<char *>(LocalSegRemain) + Size;
  MemBlocks.push_back({Ptr, Size});
  return Ptr;
}

size_t MemoryManager::getSegmentSpace(int Rank) {
  return SegInfo[Rank].SegSize;
}

void *MemoryManager::getSegmentAddr(int Rank) {
  return SegInfo[Rank].SegStart;
}

size_t MemoryManager::getAvailableSize(){
  uintptr_t Start = reinterpret_cast<uintptr_t>(LocalSegStart);
  uintptr_t End = Start + LocalSegSize;
  if (End < Start) {
    return 0; // Handle overflow
  }
  return static_cast<size_t>(End - Start);
}

size_t MemoryManager::getOffset(void *Ptr, int Rank) {
  uintptr_t Start = reinterpret_cast<uintptr_t>(SegInfo[Rank].SegStart);
  uintptr_t ptr = reinterpret_cast<uintptr_t>(Ptr);

  if (ptr < Start || ptr > Start + SegInfo[Rank].SegSize) {
    return static_cast<size_t>(-1);
  }

  return static_cast<size_t>(ptr - Start);
}

void *MemoryManager::convertRemotetoLocalAddr(void *Ptr, int Rank) {
  uintptr_t RemoteOffset = reinterpret_cast<uintptr_t>(Ptr) -
                           reinterpret_cast<uintptr_t>(getSegmentAddr(Rank));
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(LocalSegStart) +
                                  RemoteOffset);
}



#ifdef DIOMP_ENABLE_CUDA
// CUDA Memory Manager implementation
CUDAMemoryManager::CUDAMemoryManager(gex_TM_t gexTeam, int Mode) 
    : MemoryManager(gexTeam, Mode) {
  int targetDevicesNum = omp_get_num_devices();
  if (Mode != 1) {
    LocalRank = MyRank % targetDevicesNum;
    targetDevicesNum = 1;
  } else {
    LocalRank = 0;
  }

  DeviceEPs.resize(targetDevicesNum);
  gex_MK_Create_args_t args;
  args.gex_flags = 0;
  args.gex_class = GEX_MK_CLASS_CUDA_UVA;
  
  gex_MK_t mk_array[targetDevicesNum];
  void *tmp = nullptr;
  
  // Create and bind local segments for each device
  for (int DeviceID = 0; DeviceID < targetDevicesNum; DeviceID++) {
    gex_EP_t DeviceEP;
    args.gex_args.gex_class_cuda_uva.gex_CUdevice = DeviceID + LocalRank;
    GASNET_Safe(gex_MK_Create(&mk_array[DeviceID], diompClient, &args, 0));
    
    void *DeviceSegAddr = omp_target_alloc(LocalSegSize, DeviceID + LocalRank);
    tmp = DeviceSegAddr;
    
    gex_Segment_t DeviceSeg = GEX_SEGMENT_INVALID;
    GASNET_Safe(gex_Segment_Create(&DeviceSeg, diompClient, DeviceSegAddr,
                                  LocalSegSize, mk_array[DeviceID], 0));
    
    GASNET_Safe(gex_EP_Create(&DeviceEP, diompClient, GEX_EP_CAPABILITY_RMA, 0));
    GASNET_Safe(gex_EP_BindSegment(DeviceEP, DeviceSeg, 0));
    GASNET_Safe(gex_EP_PublishBoundSegment(diompTeam, &DeviceEP, 1, 0));
    DeviceEPs[DeviceID] = DeviceEP;
  }

  // Initialize device segment info
  DeviceSegInfo.resize(RanksNum, std::vector<gex_DeviceSeginfo_t>(targetDevicesNum));
  for (int Rank = 0; Rank < RanksNum; Rank++) {
    for (int DeviceID = 0; DeviceID < targetDevicesNum; DeviceID++) {
      DeviceSegInfo[Rank][DeviceID].SegStart = nullptr;
      DeviceSegInfo[Rank][DeviceID].SegRemain = nullptr;
      DeviceSegInfo[Rank][DeviceID].SegSize = 0;
    }
  }

  // Setup CUDA IPC
  if (Mode != 1) {
    for (int DeviceID = 0; DeviceID < omp_get_num_devices(); DeviceID++) {
      if (DeviceID == LocalRank) continue;
      
      CUDACHECK(cudaSetDevice(LocalRank));
      cudaDeviceEnablePeerAccess(DeviceID, 0);
      CUDACHECK(cudaSetDevice(DeviceID));
      cudaDeviceEnablePeerAccess(LocalRank, 0);
    }
  }
}

void *CUDAMemoryManager::deviceAlloc(size_t Size, int DeviceId) {
  if (Mode != 1) {
    DeviceId = 0;
  }

  // Assuming tmpRemain is a class member variable, initialized to 0
  static const size_t ALIGNMENT = 16; // Example: 16-byte alignment
  static uintptr_t MaxAddr = 0;       // Tracks the maximum allowed address

  // Initialize tmpRemain if it is uninitialized
  if (tmpRemain == 0) {
    void *Res = getDeviceSegmentAddr(MyRank, DeviceId);
    if (!Res) {
      THROW_ERROR("Failed to get device segment address");
    }
    tmpRemain = reinterpret_cast<uintptr_t>(Res);
    MaxAddr = tmpRemain +
              getSegmentSpace(MyRank); // Assume this gets the segment size
  }

  // Ensure the address is properly aligned
  uintptr_t Res = (tmpRemain + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
  if (Res + Size > MaxAddr) {
    throw std::bad_alloc(); // Allocation exceeds the available memory
  }

  tmpRemain = Res + Size;
  return reinterpret_cast<void *>(Res);
}

void CUDAMemoryManager::deviceDealloc() {
  tmpRemain = reinterpret_cast<uintptr_t>(nullptr);
}

void *CUDAMemoryManager::getDeviceSegmentAddr(int Rank, int DeviceID) {
  if (Mode != 1) {
    DeviceID = 0;
  }
  if (DeviceSegInfo[Rank][DeviceID].SegStart == nullptr) {
    gex_EP_Index_t TargetEPIdx = gex_EP_QueryIndex(DeviceEPs[DeviceID]);
    gex_TM_t TargetTM = gex_TM_Pair(DeviceEPs[0], TargetEPIdx);
    gex_Event_Wait(gex_EP_QueryBoundSegmentNB(
        TargetTM, Rank, &DeviceSegInfo[Rank][DeviceID].SegStart, nullptr,
        &DeviceSegInfo[Rank][DeviceID].SegSize, 0));
    DeviceSegInfo[Rank][DeviceID].SegRemain =
        DeviceSegInfo[Rank][DeviceID].SegStart;
  }
  return DeviceSegInfo[Rank][DeviceID].SegStart;
}

size_t CUDAMemoryManager::getDeviceOffset(void *Ptr) {
  uintptr_t LocalBase =
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(MyRank, 0));
  uintptr_t LocalOffset = reinterpret_cast<uintptr_t>(Ptr) - LocalBase;
  return (size_t)LocalOffset;
}

void *CUDAMemoryManager::convertLocaltoRemoteAddr(void *Ptr, int Rank, int DeviceId) {
    uintptr_t LocalBase =
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(MyRank, DeviceId));
  uintptr_t LocalOffset = reinterpret_cast<uintptr_t>(Ptr) - LocalBase;
  return reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(Rank, DeviceId)) +
      LocalOffset);
}

void *CUDAMemoryManager::convertRemotetoLocalAddr(void *Ptr, int Rank, int DeviceId) {
  uintptr_t RemoteBase =
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(Rank, DeviceId));
  uintptr_t RemoteOffset = reinterpret_cast<uintptr_t>(Ptr) - RemoteBase;
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(LocalSegStart) +
                                  RemoteOffset);
}

size_t CUDAMemoryManager::getOffset(void *Ptr, int Rank, int DeviceId) {
  uintptr_t Start = reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(Rank, DeviceId));
  uintptr_t ptr = reinterpret_cast<uintptr_t>(Ptr);

  if (ptr < Start || ptr > Start + getSegmentSpace(Rank)) {
    return static_cast<size_t>(-1);
  }

  return static_cast<size_t>(ptr - Start);
}

gex_EP_t CUDAMemoryManager::getEP(int DeviceId) {
  return DeviceEPs[DeviceId];
}

cudaIpcMemHandle_t CUDAMemoryManager::getIpcHandle(int Rank) {
  return IpcHandles[Rank];
}

#endif // DIOMP_ENABLE_CUDA

} // namespace diomp