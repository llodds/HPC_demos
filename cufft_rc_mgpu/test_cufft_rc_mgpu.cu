//test multi-GPU C2R and R2C CUFFT
#include "utils.h"



int main(){
    //1. init
    Array a({n[0]+2, n[1], n[2]}, 0);
#pragma omp parallel for
    for(uint iy = 0; iy < n[2]; iy++)
        for(uint ix = 0; ix < n[1]; ix++)
            for(uint iz = 0; iz < n[0]; iz++)
                a.setVal({iz, ix, iy}, cos(iz) + sin(ix + iy));
    a.getDataStats("a");
    a.dumpData("a");
    
    
    
    //2. single-GPU R2C/C2R FFT
    {
        Array a1(a);
        a1.prefetchToGPU(0);
        cufftHandle planR2C; CUCHK(cufftCreate(&planR2C));
        cufftHandle planC2R; CUCHK(cufftCreate(&planC2R));
        size_t workSize;
        CUCHK(cufftMakePlan3d(planR2C, n[2], n[1], n[0], CUFFT_R2C, &workSize));
        CUCHK(cufftMakePlan3d(planC2R, n[2], n[1], n[0], CUFFT_C2R, &workSize));
        CUCHK(cufftExecR2C(planR2C, (cufftReal*)a1.data, (cufftComplex*)a1.data)); //need (cufftComplex*)!
        mulByFFTScalar <<< 160, 1024 >>> ((float*)a1.data, fft_size, a1.size); //need (float*)!
        CUCHK(cudaGetLastError());
        CUCHK(cufftExecC2R(planC2R, (cufftComplex*)a1.data, (cufftReal*)a1.data));
        CUCHK(cufftDestroy(planR2C));
        CUCHK(cufftDestroy(planC2R));
        cleanPadRegion(a1);
        a1.getDataStats("a1"); //"a1" should be very close to "a"
        //a1.dumpData("a1");
    }
    
    
    
    //3. multi-GPU R2C/C2R FFT
    {
        Array a2(a);
        cufftHandle planR2CmGPU; CUCHK(cufftCreate(&planR2CmGPU));
        cufftHandle planC2RmGPU; CUCHK(cufftCreate(&planC2RmGPU));
        int nGPUs = 3;
        int whichGPUs[4] = {1, 2, 3, 0}; //max accessible GPUs
        size_t workSize[4];
        CUCHK(cufftXtSetGPUs(planR2CmGPU, nGPUs, whichGPUs));
        CUCHK(cufftMakePlan3d(planR2CmGPU, n[2], n[1], n[0], CUFFT_R2C, workSize));
        CUCHK(cufftXtSetGPUs(planC2RmGPU, nGPUs, whichGPUs));
        CUCHK(cufftMakePlan3d(planC2RmGPU, n[2], n[1], n[0], CUFFT_C2R, workSize));
        cudaLibXtDesc *da2; CUCHK(cufftXtMalloc(planR2CmGPU, &da2, CUFFT_XT_FORMAT_INPLACE));
        CUCHK(cufftXtMemcpy(planR2CmGPU, da2, a2.data, CUFFT_COPY_HOST_TO_DEVICE));
        cout << "y-dim = ";
        for(int i = 0; i < nGPUs; i++){
            cout << da2->descriptor->size[i]/4/(n[0]+2)/n[1];
            if(i < nGPUs - 1) cout << ", "; //when n[2]=512, nGPUs=3: y-dim=171,171,170
        }
        cout << endl;
        CUCHK(cufftXtExecDescriptorR2C(planR2CmGPU, da2, da2));
        for(int i = 0; i < nGPUs; i++){
            CUCHK(cudaSetDevice(whichGPUs[i]));
            mulByFFTScalar <<< 160, 1024 >>> ((float*)da2->descriptor->data[i], fft_size,
                                              da2->descriptor->size[i]/4); //need (float*)!
        }
        for(int i = 0; i < nGPUs; i++){
            CUCHK(cudaSetDevice(whichGPUs[i]));
            CUCHK(cudaGetLastError());
            CUCHK(cudaDeviceSynchronize());
        }
        CUCHK(cufftXtExecDescriptorC2R(planC2RmGPU, da2, da2));
        CUCHK(cufftXtMemcpy(planC2RmGPU, a2.data, da2, CUFFT_COPY_DEVICE_TO_HOST)); //planR2CmGPU doesn't work
        CUCHK(cufftXtFree(da2));
        CUCHK(cufftDestroy(planR2CmGPU));
        CUCHK(cufftDestroy(planC2RmGPU));
        cleanPadRegion(a2);
        a2.getDataStats("a2"); //"a2" should be very close to "a"
        //a2.dumpData("a2");
    }
    
    
    
    //4. single-GPU R2C/C2R FFT apply multiplier
    {
        Array a3(a);
        a3.prefetchToGPU(0);
        Array lap({n[0]/2+1, n[1], n[2]});
        computeLapCoeff(lap);
        lap.prefetchToGPU(0);
        cufftHandle planR2C; CUCHK(cufftCreate(&planR2C));
        cufftHandle planC2R; CUCHK(cufftCreate(&planC2R));
        size_t workSize;
        CUCHK(cufftMakePlan3d(planR2C, n[2], n[1], n[0], CUFFT_R2C, &workSize));
        CUCHK(cufftMakePlan3d(planC2R, n[2], n[1], n[0], CUFFT_C2R, &workSize));
        CUCHK(cufftExecR2C(planR2C, (cufftReal*)a3.data, (cufftComplex*)a3.data));
        applyLap <<< 160, 1024 >>> ((float2*)a3.data, lap.data, fft_size, lap.size);
        CUCHK(cudaGetLastError());
        CUCHK(cufftExecC2R(planC2R, (cufftComplex*)a3.data, (cufftReal*)a3.data));
        CUCHK(cufftDestroy(planR2C));
        CUCHK(cufftDestroy(planC2R));
        cleanPadRegion(a3);
        a3.getDataStats("a3");
        //a3.dumpData("a3");
    }
    
    
    
    //5. multi-GPU R2C/C2R FFT apply multiplier
    {
        Array a4(a);
        int nGPUs = 3;
        int whichGPUs[4] = {1, 2, 3, 0}; //max accessible GPUs
        float* lap[4];
        vector<size_t> lap_size(4);
        computeLapCoeff(lap, nGPUs, whichGPUs, lap_size);
        cufftHandle planR2CmGPU; CUCHK(cufftCreate(&planR2CmGPU));
        cufftHandle planC2RmGPU; CUCHK(cufftCreate(&planC2RmGPU));
        size_t workSize[4];
        CUCHK(cufftXtSetGPUs(planR2CmGPU, nGPUs, whichGPUs));
        CUCHK(cufftMakePlan3d(planR2CmGPU, n[2], n[1], n[0], CUFFT_R2C, workSize));
        CUCHK(cufftXtSetGPUs(planC2RmGPU, nGPUs, whichGPUs));
        CUCHK(cufftMakePlan3d(planC2RmGPU, n[2], n[1], n[0], CUFFT_C2R, workSize));
        cudaLibXtDesc *da4; CUCHK(cufftXtMalloc(planR2CmGPU, &da4, CUFFT_XT_FORMAT_INPLACE));
        CUCHK(cufftXtMemcpy(planR2CmGPU, da4, a4.data, CUFFT_COPY_HOST_TO_DEVICE));
        CUCHK(cufftXtExecDescriptorR2C(planR2CmGPU, da4, da4));
        for(int i = 0; i < nGPUs; i++){
            CUCHK(cudaSetDevice(whichGPUs[i]));
            applyLap <<< 160, 1024 >>> ((float2*)da4->descriptor->data[i], lap[i], fft_size, lap_size[i]);
        }
        for(int i = 0; i < nGPUs; i++){
            CUCHK(cudaSetDevice(whichGPUs[i]));
            CUCHK(cudaGetLastError());
            CUCHK(cudaDeviceSynchronize());
        }
        CUCHK(cufftXtExecDescriptorC2R(planC2RmGPU, da4, da4));
        CUCHK(cufftXtMemcpy(planC2RmGPU, a4.data, da4, CUFFT_COPY_DEVICE_TO_HOST)); //planR2CmGPU doesn't work
        CUCHK(cufftXtFree(da4));
        CUCHK(cufftDestroy(planR2CmGPU));
        CUCHK(cufftDestroy(planC2RmGPU));
        for(int i = 0; i < nGPUs; i++) CUCHK(cudaFree(lap[i]));
        cleanPadRegion(a4);
        a4.getDataStats("a4"); //"a4" should be very close to "a3"
        //a4.dumpData("a4");
    }
    
    return 0;
}


//output:
//TMP=/tmp/zhmukc/ OMP_NUM_THREADS=96 KMP_AFFINITY=compact ./test_cufft_rc_mgpu
//=== "a" stats ===
//    max = 1.999991e+00, min=-1.999991e+00
//    mean = -1.331778e-04, std2=1.006595e+09
//=== "a1" stats ===
//    max = 1.999991e+00, min=-1.999992e+00
//    mean = -1.331727e-04, std2=1.006595e+09
//y-dim = 427, 427, 426
//=== "a2" stats ===
//    max = 1.999991e+00, min=-1.999991e+00
//    mean = -1.331727e-04, std2=1.006595e+09
//=== "a3" stats ===
//    max = 4.594342e+02, min=-4.594852e+02
//    mean = 2.153811e-08, std2=8.528039e+13
//=== "a4" stats ===
//    max = 4.594343e+02, min=-4.594852e+02
//    mean = 2.154614e-08, std2=8.528039e+13
