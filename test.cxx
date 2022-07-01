
#include <Kokkos_Core.hpp>
#include <adios2.h>

int main(int argc, char** argv)
{
  constexpr std::size_t Nx = 6;
  Kokkos::ScopeGuard kokkos(argc, argv);

  MPI_Init(&argc, &argv);
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  adios2::ADIOS adios(MPI_COMM_WORLD);

  adios2::IO io = adios.DeclareIO("Output");
  io.SetEngine("BP4");

  adios2::Variable<double> varGlobalArray =
    io.DefineVariable<double>("GlobalArray", {(size_t)nprocs, Nx});

  adios2::Engine writer = io.Open("globalArray.bp", adios2::Mode::Write);
  writer.BeginStep();

  Kokkos::View<double*> arr("my_view", Nx);
  for (int i = 0; i < Nx; i++) {
    arr(i) = 10. * rank;
  }
  // std::vector<double> arr(Nx, 10. * rank);

  varGlobalArray.SetSelection(
    adios2::Box<adios2::Dims>({size_t(rank), 0}, {1, Nx}));
  writer.Put(varGlobalArray, arr);

  writer.EndStep();
  writer.Close();

  MPI_Finalize();
  return 0;
}
