/****************************************************************/
/*                                                              */
/*  Copyright (C) 2018, M. Andelkovic, L. Covaci, A. Ferreira,  */
/*                    S. M. Joao, J. V. Lopes, T. G. Rappoport  */
/*                                                              */
/****************************************************************/
#include "Generic.hpp"
#include "ComplexTraits.hpp"
#include "Global.hpp"
#include "Coordinates.hpp"
#include "LatticeStructure.hpp"
#include "myHDF5.hpp"
#include "Random.hpp"
template <typename T, unsigned D>
class Hamiltonian;
template <typename T, unsigned D>
class KPM_Vector;
#include "queue.hpp"
#include "Simulation.hpp"
#include "SimulationGlobal.hpp"
#include "Hamiltonian.hpp"
template <typename T,unsigned D>
GlobalSimulation<T,D>::GlobalSimulation( char *name ) : rglobal(name)
{
  debug_message("Entered global_simulation\n");
  Global.ghosts.resize( rglobal.get_BorderSize() );
  std::fill(Global.ghosts.begin(), Global.ghosts.end(), 0);
    
  // Regular quantities to calculate, such as DOS and CondXX
  H5::H5File * file12         = new H5::H5File(name, H5F_ACC_RDONLY);

  // Fetch the energy scale and the magnetic field, if it exists
  get_hdf5<double>(&EnergyScale,  file12, (char *)   "/EnergyScale");
  file12->close();
  delete file12;
  
  omp_set_num_threads(rglobal.n_threads);
  debug_message("Starting parallelization\n");
#pragma omp parallel default(shared)
  {
    Simulation<T,D> simul(name, Global);
  //std::cout << "got to line " << __LINE__ << " in file " << __FILE__ << "\n" << std::flush;

    simul.calc_conddc();
    simul.calc_condopt();
    simul.calc_condopt2();
    simul.calc_singleshot();
    simul.calc_DOS();
    simul.calc_wavepacket();
    simul.calc_LDOS(); 
    simul.calc_ARPES(); // fetches parameters from .h5 file and calculates ARPES

  }
  debug_message("Left global_simulation\n");
};

template class GlobalSimulation<float ,1u>;
template class GlobalSimulation<double ,1u>;
template class GlobalSimulation<long double ,1u>;
template class GlobalSimulation<std::complex<float> ,1u>;
template class GlobalSimulation<std::complex<double> ,1u>;
template class GlobalSimulation<std::complex<long double> ,1u>;

template class GlobalSimulation<float ,3u>;
template class GlobalSimulation<double ,3u>;
template class GlobalSimulation<long double ,3u>;
template class GlobalSimulation<std::complex<float> ,3u>;
template class GlobalSimulation<std::complex<double> ,3u>;
template class GlobalSimulation<std::complex<long double> ,3u>;

template class GlobalSimulation<float ,2u>;
template class GlobalSimulation<double ,2u>;
template class GlobalSimulation<long double ,2u>;
template class GlobalSimulation<std::complex<float> ,2u>;
template class GlobalSimulation<std::complex<double> ,2u>;
template class GlobalSimulation<std::complex<long double> ,2u>;
