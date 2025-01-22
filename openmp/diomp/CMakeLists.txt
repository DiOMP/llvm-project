# CMakeLists.txt for DiOMP

if(OPENMP_DIOMP_GASNET_ROOT)
  set(GASNET_ROOT "${OPENMP_DIOMP_GASNET_ROOT}")
else()
  set(GASNET_ROOT "/home/l1065028/gasnet-arm-cuda")
endif()
set(GASNET_LIB "${GASNET_ROOT}/lib")

if(DEFINED OPENMP_GASNET_API)
  if(OPENMP_GASNET_API STREQUAL "IBV")
    set(GASNET_CONDUIT_IBV ON)
  elseif(OPENMP_GASNET_API STREQUAL "OFI")
    set(GASNET_CONDUIT_OFI ON)
  else()
    message("Invalid GASNET_API value: ${OPENMP_GASNET_API}. Defaulting to IBV.")
    set(GASNET_CONDUIT_IBV ON)
  endif()
else()
  message("OPENMP_GASNET_API not defined. Defaulting to IBV.")
  set(GASNET_CONDUIT_IBV ON)
endif()

if(GASNET_CONDUIT_IBV)
  set(GASNET_INCLUDE "${GASNET_ROOT}/include/ibv-conduit/")
  message("Using IBV")
elseif(GASNET_CONDUIT_OFI)
  set(GASNET_INCLUDE "${GASNET_ROOT}/include/ofi-conduit/")
  set(LIBFABRIC_ROOT "/opt/cray/libfabric/1.20.1")
  set(LIBFABRIC_LIB "${LIBFABRIC_ROOT}/lib64")
  set(LIBPMI_LIB "/opt/cray/pe/pmi/6.1.13/lib")
  set(MPI_ROOT "/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3")
  set(MPI_INCLUDE "${MPI_ROOT}/include")
  set(LIBMPI_LIB "${MPI_ROOT}/lib")
  link_directories(${LIBFABRIC_LIB} ${LIBPMI_LIB} ${LIBMPI_LIB})
  message("Using OFI")
endif()
include_directories(${GASNET_INCLUDE} ${MPI_INCLUDE})

if(DEFINED OPENMP_HWLOC_LIB)
  set(HWLOC_LIB "${OPENMP_HWLOC_LIB}")
else()
  set(HWLOC_LIB "/lustre/software/hwloc/2.8.0/lib")
endif()

# Device support options
option(OPENMP_ENABLE_DIOMP_DEVICE "Enable device support in DiOMP" OFF)
if(OPENMP_ENABLE_DIOMP_DEVICE)
  message(STATUS "Building DiOMP with device support")
  add_definitions(-DOPENMP_ENABLE_DIOMP_DEVICE)
endif()

# CUDA support
option(OPENMP_DIOMP_ENABLE_CUDA "Enable CUDA support in DiOMP" OFF)
if(OPENMP_DIOMP_ENABLE_CUDA)
  if(NOT OPENMP_ENABLE_DIOMP_DEVICE)
    message(FATAL_ERROR "CUDA support requires OPENMP_ENABLE_DIOMP_DEVICE=ON")
  endif()
  
  message(STATUS "Building DiOMP with CUDA support")
  add_definitions(-DDIOMP_ENABLE_CUDA)
  
  # Find CUDA and NCCL
  if(DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME $ENV{CUDA_HOME})
  else()
    set(CUDA_HOME "/hrtc/apps/cuda/12.4.131/aarch64/rocky9")
  endif()
  
  if(DEFINED ENV{NCCL_HOME})
    set(NCCL_HOME $ENV{NCCL_HOME})
  else()
    set(NCCL_HOME "/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131//Linux_aarch64/24.5/comm_libs/nccl/")
  endif()
  
  include_directories(${CUDA_HOME}/include ${NCCL_HOME}/include)
  link_directories(${CUDA_HOME}/lib64 ${NCCL_HOME}/lib)
endif()

# HIP support 
option(DIOMP_ENABLE_HIP "Enable HIP support in DiOMP" OFF)
if(DIOMP_ENABLE_HIP)
  if(NOT OPENMP_ENABLE_DIOMP_DEVICE)
    message(FATAL_ERROR "HIP support requires OPENMP_ENABLE_DIOMP_DEVICE=ON")
  endif()
  
  message(STATUS "Building DiOMP with HIP support")
  add_definitions(-DDIOMP_ENABLE_HIP)
  # Add HIP related setup here
endif()

add_library(diomp SHARED
  src/diomp.cpp
  src/diompmem.cpp
  src/diompcomm.cpp
)

link_directories(${GASNET_LIB})

set(DIOMP_LIBS
  # Base libraries
  -lmpi
  -fopenmp
  -lrt
  -lhwloc
  -pthread

  # GASNet IBV libraries
  $<$<BOOL:${GASNET_CONDUIT_IBV}>:-lgasnet-ibv-par>
  $<$<BOOL:${GASNET_CONDUIT_IBV}>:-libverbs>
  $<$<BOOL:${GASNET_CONDUIT_IBV}>:--offload-arch=sm_90>

  # GASNet OFI libraries
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:${GASNET_LIB}/libgasnet-ofi-par.a>
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:-lgasnet-ofi-par>
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:-lhugetlbfs>
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:-lfabric>
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:-lpmi>
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:--offload-arch=sm_80>
  # CUDA libraries
  $<$<BOOL:${OPENMP_DIOMP_ENABLE_CUDA}>:-lcudart>
  $<$<BOOL:${OPENMP_DIOMP_ENABLE_CUDA}>:-lcuda>
  $<$<BOOL:${OPENMP_DIOMP_ENABLE_CUDA}>:-lnccl>
)

target_link_libraries(diomp PUBLIC ${DIOMP_LIBS})

target_include_directories(diomp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../offload/include
)

install(TARGETS diomp LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
install(FILES 
  include/diomp.h 
  include/diomp.hpp
  include/diompmem.h
  include/diompcomm.h
  DESTINATION "${LIBOMP_HEADERS_INSTALL_PATH}")
# install(PROGRAMS 
#  src/diompcxx.py
#  src/diompcc.py
#  DESTINATION "${CMAKE_INSTALL_BINDIR}"
#  RENAME diompcxx
# )