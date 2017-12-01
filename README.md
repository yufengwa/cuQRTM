# cuQ-RTM

## Overview of cuQ-RTM package

`cuQRTM` is a CUDA-based code package that implements Q-RTM based on a set of stable and efficient strategies, such as streamed CUFFT, checkpointing-assisted time-reversal reconstruction (CATRC) and adaptive stabilization scheme. This package is provided for accelerating conventional CPU-based Q-RTM, and mimicking how a geophysicist writes down a seismic processing modules such as modeling, imaging and inversion in the framework of the `CPU-GPU` heterogeneous computing platform. We provide two package versions: `cuQRTM-Express` for quick execution with 4 shots, which can be done within 3 min under single GPU card (GTX 760); `cuQRTM-Standard` for standard excution with 64 shots, which will take 10 min under 4 GPU cards (Tesla K10).

## The architecture of cuQ-RTM package 

-   `input`: accurate velocity and Q model for Q-RTM:
    - `acc_vp.txt`: quasi-Marmousi velocity model;
    - `acc_Qp.txt`: quasi-Marmousi Q model;
    - `ascii2bin.m`: converting ASCII files to Binary files.
-   `output`: generated results such as seismograms, images of each shot and final stacked images. What we are most interested in is final images, wich includes:
    - `Final_image_cor_type0.dat`: image from acoustic RTM;
    - `Final_image_cor_type1.dat`: image from viscoacoustic RTM with compensation;
    - `Final_image_cor_type2.dat`: image from Q-RTM using low-pass filtering;
    - `Final_image_cor_type3.dat`: image from Q-RTM using adaptive stabilization;
-   `plot`: scripts for plotting figures, which includes:
    - `/madagascar/SConstruct`: plot images, velocity and Q models;
    - `/matlab/martrace`: plot extracted trace from final migrated images for comparison.
-   `Myfunctions.h`: header file;
-   `CUDAQRTM.cu`: cuda code file;
-   `QRTM.cpp`: c++ code file
-   `Makefile`: excution script.

## Prerequisite

`cuQRTM` package is developed under `Linux` system, which should be equipped with the following environments:
- CUDA environment (for example, `-I/usr/local/cuda-8.0/include` `-L/usr/local/cuda-8.0/lib64`);
- MPI environment: (for example, `-I/home/wyf/intel/impi/5.0.1.035/intel64/include` `-L/home/wyf/intel/impi/5.0.1.035/intel64/lib/`);
- matlab;
- madagascar.

## How to run this package

If you want to quick test the package, please use fast version `cuQRTM-Express`; `cuQRTM-Standard` should be excuted on cluster with multi-GPUs (or you can excute on a sigle GPU card within an hour). 

- Step 1: Run the matlab file `ascii2bin.m` in `./input` to convert the ASCII data into binary data;
- Step 2: Confirm the environment in `Makefile`, and replace the folder path with your own enviroment path; 
- Step 3: Run the `Makefile` by the command line: `make`;
- Step 4: View generated files in the folder `./ouput`;
- Step 5: Plot figures by run `/plot/madagascar/SConstruct` and `/plot/matlab/martrace.m`.

## Contact me

I am Yufeng Wang, a PhD candidate from China University of Petroleum, Beijing. If you have any question about this coda package, please feel free to contact me by [Email](hellowangyf@163.com).

## Copyright

  Copyright (C) 
  2017  China University of Petroleum, Beijing (Yufeng Wang) 

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
