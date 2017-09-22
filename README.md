# cuQRTM
cuQRTM is a CUDA-based code package that implements Q-RTM based on a set of stable and efficient strategies, such as streamed CUFFT, checkpointing-assisted time-reversal reconstruction (CATRC) and adaptive stabilization.

Original Model data are put in input folder;
CUQRTM.cu and QRTM.cpp are cuda and c++ code, respectively, Myfunctions.h is header file;
Makefile and bash file run.sh are provided for compilling our code, which should be redefined by users (change the filepath according to your own computer);
The final results will be saved in output folder.

If you have any question about this coda package, please contact me by Email: hellowangyf@163.com.
