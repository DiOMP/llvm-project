# CMakeLists.txt for diomp

IF(NOT GASNET_ROOT)
    MESSAGE(STATUS "GASNET_LIB is not set.")
    set(GASNET_ROOT "/home/l1065028/gasnet-arm-cuda")
ENDIF()

IF(NOT HWLOC_LIB)
    MESSAGE(STATUS "HWLOC_LIB is not set.")
    set(HWLOC_LIB "/lustre/software/hwloc/2.8.0/lib")
ENDIF()

IF(OPENMP_ENABLE_DIOMP_DEVICE)
  add_definitions(-DOPENMP_ENABLE_DIOMP_DEVICE)
  MESSAGE(STATUS "Building DiOMP device plugin")
ENDIF()

set(GASNET_IBV_INCLUDE "${GASNET_ROOT}/include/ibv-conduit/" "/hrtc/apps/cuda/12.4.131/aarch64/rocky9/lib64")
set(GASNET_LIB "${GASNET_ROOT}/lib")
set(CUDA_HOME "/hrtc/apps/cuda/12.4.131/aarch64/rocky9")
set(NCCL_HOME "/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131//Linux_aarch64/24.5/comm_libs/nccl/")




IF(OPENMP_DIOMP_ENABLE_CUDA)
  add_definitions(-DDIOMP_ENABLE_CUDA)
  MESSAGE(STATUS "Building DiOMP with CUDA support")
  include_directories(${CUDA_HOME}/include ${NCCL_HOME}/include)
  link_directories(${CUDA_HOME}/lib64 ${NCCL_HOME}/lib)
ENDIF()


include_directories(${GASNET_IBV_INCLUDE})
link_directories(${GASNET_LIB} ${HWLOC_LIB})


add_library(diomp SHARED
  src/diomp.cpp
  src/diompmem.cpp
)

target_link_libraries(diomp -lcuda -lcudart -lmpi -fopenmp --offload-arch=sm_90
-lrt -lgasnet-ibv-par -lhwloc -pthread -libverbs -lnccl)

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



# clang++ test.cpp -ldiomp -isystem  /home/l1065028/llvm-install/include -I $HOME/gasnet-install/include/ibv-conduit/ -I $MPI_HOME/include -L $MPI_HOME/lib -lmpi  -lgasnet-ibv-par -lhwloc -libverbs -lrt -pthread -fopenmp