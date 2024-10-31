# CMakeLists.txt for diomp

IF(NOT GASNET_ROOT)
    MESSAGE(STATUS "GASNET_LIB is not set.")
    set(GASNET_ROOT "/global/cfs/cdirs/xpress/baodi/software/gasnet-gpu/")
ENDIF()

IF(NOT HWLOC_LIB)
    MESSAGE(STATUS "HWLOC_LIB is not set.")
    set(HWLOC_LIB "/lustre/software/hwloc/2.8.0/lib")
ENDIF()

IF(OPENMP_ENABLE_DIOMP_DEVICE)
  add_definitions(-DOPENMP_ENABLE_DIOMP_DEVICE)
  MESSAGE(STATUS "Building DiOMP device plugin")
ENDIF()

set(GASNET_OFI_INCLUDE "${GASNET_ROOT}/include/ofi-conduit/")
set(GASNET_LIB "${GASNET_ROOT}/lib")
set(CUDA_HOME "/hrtc/apps/cuda/11.2.152/x86_64")
include_directories(${GASNET_OFI_INCLUDE})
link_directories(${GASNET_LIB} ${HWLOC_LIB})


add_library(diomp SHARED
  src/diomp.cpp
  src/diompmem.cpp
)

# #clang++ diomp_lat.cpp -ldiomp -fopenmp -L /global/cfs/cdirs/xpress/baodi/software/llvm/diomp/lib -L /global/cfs/cdirs/xpress/baodi/software/gasnet/lib -I /global/cfs/cdirs/xpress/baodi/software/gasnet/include/ofi-conduit/ -lgasnet-ofi-par -no-pie -lmpi -lhwloc -lrt -pthread -I /opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/include -L /opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/lib -L/opt/cray/libfabric/1.15.2.0/lib64 -lfabric -L /opt/cray/pe/pmi/6.1.13/lib -lpmi -lhugetlbfs -o diomp_lat_put
target_link_libraries(diomp 
-I/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/include
-L/opt/cray/libfabric/1.15.2.0/lib64 -lfabric -L/opt/cray/pe/pmi/6.1.13/lib -lpmi
-L/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/lib 
-lcuda -lmpi -fopenmp --offload-arch=sm_80
-lrt -lgasnet-ofi-par -lhwloc -pthread -lhugetlbfs)

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