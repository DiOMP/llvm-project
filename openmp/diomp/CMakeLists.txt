# CMakeLists.txt for DiOMP

if(OPENMP_DIOMP_GASNET_ROOT)
  set(GASNET_ROOT "${OPENMP_DIOMP_GASNET_ROOT}")
  set(GASNET_LIB "${GASNET_ROOT}/lib")
else()
  message("GASNET_ROOT is not set.")
  set(GASNET_ROOT "/home/l1065028/gasnet-arm-cuda")
  set(GASNET_LIB "${GASNET_ROOT}/lib")
endif()

if(DEFINED OPENMP_GASNET_API)
  if(OPENMP_GASNET_API STREQUAL "IBV")
    set(GASNET_CONDUIT_IBV ON)
  elseif(OPENMP_GASNET_API STREQUAL "OFI")
    set(GASNET_CONDUIT_OFI ON)
  else()
    message("Invalid GASNET_API value: ${OPENMP_GASNET_API}. Defaulting to GASNET_CONDUIT_IBV.")
    set(GASNET_CONDUIT_IBV ON)
  endif()
else()
  message("OPENMP_GASNET_API is not defined. Defaulting to GASNET_CONDUIT_IBV.")
  set(GASNET_CONDUIT_IBV ON)
endif()

if(GASNET_CONDUIT_IBV)
  set(GASNET_IBV_INCLUDE "${GASNET_ROOT}/include/ibv-conduit/")
  message("Using IBV")
  include_directories(${GASNET_IBV_INCLUDE})
endif()

if(GASNET_CONDUIT_OFI)
  set(GASNET_OFI_INCLUDE "${GASNET_ROOT}/include/ofi-conduit/")
  include_directories(${GASNET_OFI_INCLUDE})
endif()

if(DEFINED OPENMP_HWLOC_LIB)
  set(HWLOC_LIB "${OPENMP_HWLOC_LIB}")
else()
  message("HWLOC_LIB is not set.")
  set(HWLOC_LIB "/lustre/software/hwloc/2.8.0/lib")
endif()


IF(OPENMP_ENABLE_DIOMP_DEVICE)
  set(OPENMP_ENABLE_DIOMP_DEVICE ON)
  MESSAGE(STATUS "Building DiOMP device plugin")
ENDIF()

if(DEFINED ENV{CUDA_HOME})
  set(CUDA_HOME $ENV{CUDA_HOME})
else()
  message("CUDA_HOME is not set.")
  set(CUDA_HOME "/hrtc/apps/cuda/12.4.131/aarch64/rocky9")
endif()

if(DEFINED ENV{NCCL_HOME})
  set(NCCL_HOME $ENV{NCCL_HOME})
else()
  message("NCCL_HOME is not set.")
  set(NCCL_HOME "/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131//Linux_aarch64/24.5/comm_libs/nccl/")
endif()

add_library(diomp SHARED
  src/diomp.cpp
  src/diompmem.cpp
)

if(OPENMP_DIOMP_ENABLE_CUDA)
  add_definitions(-DDIOMP_ENABLE_CUDA)
  set(DIOMP_ENABLE_CUDA ON)
  message(STATUS "Building DiOMP with CUDA support")

  include_directories(${CUDA_HOME}/include ${NCCL_HOME}/include)
  link_directories(${CUDA_HOME}/lib64 ${NCCL_HOME}/lib)

  target_link_libraries(diomp -lcuda -lcudart -lnccl)
endif()

link_directories(${GASNET_LIB} ${HWLOC_LIB})


if(GASNET_CONDUIT_IBV)
  target_link_libraries(diomp -lgasnet-ibv-par -libverbs --offload-arch=sm_90)
endif()

if(GASNET_CONDUIT_OFI)
  target_link_libraries(diomp 
  -I/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/include
  -L/opt/cray/libfabric/1.20.1/lib64 -lfabric -L/opt/cray/pe/pmi/6.1.13/lib -lpmi
  -L/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/lib 
  --offload-arch=sm_80 -lgasnet-ofi-par -lhugetlbfs)
endif()

target_link_libraries(diomp -lmpi -fopenmp -lrt -lhwloc -pthread)

target_include_directories(diomp
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../offload/include
)

install(TARGETS diomp LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
install(FILES include/diomp.h include/diomp.hpp DESTINATION "${LIBOMP_HEADERS_INSTALL_PATH}")
install(PROGRAMS src/diompcxx.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME diompcxx)
install(PROGRAMS src/diompcc.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME diompcc)
