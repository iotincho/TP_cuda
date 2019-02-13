//
// Created by sergio on 13/02/19.
//

#include <iostream>
#include <stdint.h>  // Para medir el clock
#include <cstdlib>   // std
#include <iomanip>   // Formateo de datos
#include <string>
#define     CWIDTHLEFT      40
#define     CWIDTHRIGHT     30

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

int main(){
   int devCount = 1;

        if (devCount == 0)
        {
            std::cout << "No se detecto el modulo cuda cargado";
            exit (-1);
        }
        for (int i = 0; i < devCount; i++)
        {
            cudaDeviceProp devProp;
            cudaGetDeviceProperties (&devProp, i);

            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHLEFT) << std::setfill ('*') << " Info de la placa cuda "
                      << i << " ";
            std::cout.unsetf (std::ios::right);
            std::cout.setf (std::ios::left);
            std::cout << std::setw (CWIDTHRIGHT - 2) << std::setfill ('*') << "(todo en bytes) ";
            std::cout << std::endl;
            std::cout.unsetf (std::ios::left);

            std::cout.setf (std::ios::left);
            std::cout << std::setw (CWIDTHLEFT) << std::setfill (' ') << "Nombre:";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << devProp.name << std::endl;
            std::cout.unsetf (std::ios::right);

            std::cout << std::setw (CWIDTHLEFT) << "Total Memoria Global:";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << devProp.totalGlobalMem << std::endl;
            std::cout.unsetf (std::ios::right);

            std::cout << std::setw (CWIDTHLEFT) << "Memoria shared  por bloque (SMM): ";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << devProp.sharedMemPerBlock << std::endl;
            std::cout.unsetf (std::ios::right);
            std::cout << std::setw (CWIDTHLEFT) << "Registros por bloque: ";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << devProp.regsPerBlock << std::endl;
            std::cout.unsetf (std::ios::right);
            std::cout << std::setw (CWIDTHLEFT) << "Wrap size: ";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << devProp.warpSize << std::endl;
            std::cout.unsetf (std::ios::right);
            std::cout << std::setw (CWIDTHLEFT) << "Max threads por bloque: ";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << devProp.maxThreadsPerBlock << std::endl;
            std::cout.unsetf (std::ios::right);
            std::cout << std::setw (CWIDTHLEFT) << "Max threads por SMM: ";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << devProp.maxThreadsPerMultiProcessor
                      << std::endl;
            std::cout.unsetf (std::ios::right);
            //Dimeciones maximas por bloque
            for (int j = 0; j < 3; ++j)
            {
                // c++11
                std::string sstr = "Dimecion maxima " + std::to_string (j) + " por bloque:";
                std::cout << std::setw (CWIDTHLEFT) << sstr;
                std::cout.setf (std::ios::right);
                std::cout << std::setw (CWIDTHRIGHT) << devProp.maxThreadsDim[j] << std::endl;
                std::cout.unsetf (std::ios::right);
            }
            //Dimeciones maximas por grid
            for (int j = 0; j < 3; ++j)
            {
                // c++11
                std::string sstr = "Dimecion maxima " + std::to_string (j) + " por grid:";
                std::cout << std::setw (CWIDTHLEFT) << sstr;
                std::cout.setf (std::ios::right);
                std::cout << std::setw (CWIDTHRIGHT) << devProp.maxGridSize[j] << std::endl;
                std::cout.unsetf (std::ios::right);
            }

            std::cout << std::setw (CWIDTHLEFT) << "Clock rate: ";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << devProp.clockRate << std::endl;
            std::cout.unsetf (std::ios::right);

            std::cout << std::setw (CWIDTHLEFT) << "Total Memoria constante: ";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << devProp.totalConstMem << std::endl;
            std::cout.unsetf (std::ios::right);

            std::cout << std::setw (CWIDTHLEFT) << "Soporta copia y ejecucion concurrente: ";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << (devProp.deviceOverlap ? "Sep" : "No")
                      << std::endl;
            std::cout.unsetf (std::ios::right);

            std::cout << std::setw (CWIDTHLEFT) << "Multiprocesadores: ";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << devProp.multiProcessorCount << std::endl;
            std::cout.unsetf (std::ios::right);

            // Calculo de cuda cores
            int mp = devProp.multiProcessorCount;
            int cores;
            switch (devProp.major)
            {
                case 2: // Fermi
                    if (devProp.minor == 1)
                        cores = mp * 48;
                    else
                        cores = mp * 32;
                    break;
                case 3: // Kepler
                    cores = mp * 192;
                    break;
                case 5: // Maxwell
                    cores = mp * 128;
                    break;
                case 6: // Pascal
                    if (devProp.minor == 1)
                        cores = mp * 128;
                    else if (devProp.minor == 0)
                        cores = mp * 64;
                    else
                        std::cout << " desconocido ";
                    break;
                default:
                    std::cout << " desconocido ";
                    break;
            }
            std::cout << std::setw (CWIDTHLEFT) << "Cuda cores: ";
            std::cout.setf (std::ios::right);
            std::cout << std::setw (CWIDTHRIGHT) << cores << std::endl;
            std::cout.unsetf (std::ios::right);

        }


}

