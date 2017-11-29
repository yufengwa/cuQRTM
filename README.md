# cuQ-RTM

## Overview of cuQ-RTM package

cuQRTM is a CUDA-based code package that implements Q-RTM based on a set of stable and efficient strategies, such as streamed CUFFT, checkpointing-assisted time-reversal reconstruction (CATRC) and adaptive stabilization.

## The architecture of cuQ-RTM package 

-Data: Original Model data are put in input folder;
-Codes: CUQRTM.cu and QRTM.cpp are cuda and c++ code, respectively, Myfunctions.h is header file;
Makefile and bash file run.sh are provided for compilling our code, 
-Results: The final results will be saved in output folder.

## How to run this package

-Step 1: Run the matlab file ascii2bin.m in ./input to convert the ASCII data into binary data;
-Step 2: Confirm the folder of the environment in Makefile, replace the folder path with your own path, MPI and CUDA should be available on your computer; If all are finished, then you can run the Makefile by the command line: make
-Step 3: Run the bash file run.sh by command line: sh run.sh
-Step 4: The generated file will be placed in the folder ./ouput.

## Contact me

If you have any question about this coda package, please feel free to contact me by [Email] (hellowangyf@163.com).

## Copyright

  Copyright (C) 
  - 2017  China University of Petroleum, Beijing (Yufeng Wang) 

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
