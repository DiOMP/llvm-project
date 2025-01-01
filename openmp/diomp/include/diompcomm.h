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

#include <gasnet.h>
#include <gasnetex.h>
#include <gasnet_tools.h>
#include <gasnet_mk.h>
#include <vector>
#include <cstddef>
#include "tools.h"

namespace diomp {

class DiOMPCommunicator {
    public:
        void setDevicesNum(int DevicesNum);
        void deviceBcast(void *Date, size_t Count, omp_device_dt_t Dt, int Rank,
                 int DeviceID);

    
    private:
        int DevicesNum;



};

} // namespace diomp