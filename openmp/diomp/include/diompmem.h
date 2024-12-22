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
#include <gasnet.h>
#include <gasnetex.h>
#include <gasnet_tools.h>
#include <gasnet_mk.h>
#include <vector>
#include <cstddef>
#include "tools.h"

extern gex_TM_t diompTeam;
extern gex_Client_t diompClient;
extern gex_EP_t diompEp;
extern gex_Segment_t diompSeg;

struct gex_Seginfo_t{
  void *SegStart;
  void *SegRemain;
  size_t SegSize;
};

struct MemoryBlock{
  void *Ptr;
  size_t Size;
};

namespace diomp{

class MemoryManager {
  private:
    int RanksNum;
    int MyRank;
    std::vector<MemoryBlock> MemBlocks;
    std::vector<gex_Seginfo_t> SegInfo;
    // Device Segment Information
    // DeviceSegInfo[NodeID][DeviceID]
    std::vector<std::vector<gex_Seginfo_t>> DeviceSegInfo;
    std::vector<gex_EP_t> DeviceEPs;

    // Local Segment Information
    void *LocalSegStart;
    void *LocalSegRemain;
    size_t LocalSegSize;

    uintptr_t tmpRemain = reinterpret_cast<uintptr_t>(nullptr);

  public:
    MemoryManager(gex_TM_t gexTeam);
    ~MemoryManager(){};

    size_t getSegmentSpace(int Rank);
    void *getSegmentAddr(int Rank);
    void *getDeviceSegmentAddr(int Rank, int DeviceID);

    void *globalAlloc(size_t Size);
    void *deviceAlloc(size_t Size, int DeviceID);
    void deviceDealloc();

    size_t getAvailableSize() const;
    size_t getDeviceAvailableSize() const;
    size_t getOffset(void *Ptr); // Compute ptr on local memory
    size_t getOffset(void *Ptr, int Rank); // Compute ptr on remote memory
    size_t getDeviceOffset(void *Ptr);

    bool validGlobalAddr(void *Ptr, int Rank);
    void *convertRemotetoLocalAddr(void *Ptr, int Rank, int DeviceID);
    void *convertRemotetoLocalAddr(void *Ptr, int Rank);
    void *convertLocaltoRemoteAddr(void *Ptr, int Rank, int DeviceID);

    gex_EP_t getEP(int DeviceID);

};

} // namespace diomp

#endif // DIOMP_MEM_H
