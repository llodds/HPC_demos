An example aims to show how to do single-node multi-GPU 3D R2C/C2R FFT using cuFFT. This example contains 4 tests: 
- R2C/C2R FFT on one GPU, output is very close to input
- R2C/C2R FFT on multiple GPUs within a single node, output is very close to input
- R2C/C2R FFT on one GPU, the intermediate cufftComplex output of R2C FFT is multipled by an array called "lap"
- R2C/C2R FFT on multiple GPUs within a single node, the intermediate cufftComplex output of R2C is multipled by an array called "lap"; output is close to the output of the previous test. The hard part is to figure out data layout of "lap" across multiple GPUs to conform with data layout of the FFT array.

The code is compiled in an environment loaded with nvhpc/20.7 (Nvidia HPCSDK). 
