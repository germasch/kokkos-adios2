
cmake_minimum_required(VERSION 3.16)

project(Test)

find_package(MPI REQUIRED)
find_package(adios2 REQUIRED)
find_package(Kokkos REQUIRED)

add_executable(test test.cxx)
target_link_libraries(test PRIVATE Kokkos::kokkos adios2::cxx11_mpi MPI::MPI_CXX)
