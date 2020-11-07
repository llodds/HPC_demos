#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <typeinfo>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <cfloat>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <iostream>
using std::vector;
using std::string;
using std::to_string;
using std::cout;
using std::endl;
#define MAX(a,b) ( ( (a) > (b) )? (a) : (b) )
#define MIN(a,b) ( ( (a) < (b) )? (a) : (b) )


//=== my abort and CUDA error-checking function
inline void abort(string msg, const char *file, int line){
    fprintf(stderr, "Error in file '%s' at line '%i': %s\n",
            file, line, msg.c_str());
    exit(EXIT_FAILURE);
}
__host__ __device__ const char* cudaGetErrorString(int errCode){
    const char* errStrings[17] = {
        "The cuFFT operation was successful (CUFFT_SUCCESS)",
        "cuFFT was passed an invalid plan handle (CUFFT_INVALID_PLAN)",
        "cuFFT failed to allocate GPU or CPU memory (CUFFT_ALLOC_FAILED)",
        "No longer used (CUFFT_INVALID_TYPE)",
        "User specified an invalid pointer or parameter (CUFFT_INVALID_VALUE)",
        "Driver or internal cuFFT library error (CUFFT_INTERNAL_ERROR)",
        "Failed to execute an FFT on the GPU (CUFFT_EXEC_FAILED)",
        "The cuFFT library failed to initialize (CUFFT_SETUP_FAILED)",
        "User specified an invalid transform size (CUFFT_INVALID_SIZE)",
        "No longer used (CUFFT_UNALIGNED_DATA)",
        "Missing parameters in call (CUFFT_INCOMPLETE_PARAMETER_LIST)",
        "Execution of a plan was on different GPU than plan creation (CUFFT_INVALID_DEVICE)",
        "Internal plan database error (CUFFT_PARSE_ERROR)",
        "No workspace has been provided prior to plan execution (CUFFT_NO_WORKSPACE)",
        "Function does not implement functionality for parameters given (CUFFT_NOT_IMPLEMENTED)",
        "Used in previous versions (CUFFT_LICENSE_ERROR)",
        "Operation is not supported for parameters given (CUFFT_NOT_SUPPORTED)"
    };
    return errStrings[errCode];
}
template<typename T>
inline void checkCudaErrors(T errCode, const char *func, const char *file, int line){
    if(errCode){
        cudaError_t type_cuda;
        cufftResult type_cufft;
        string errType;
        if(typeid(errCode) == typeid(type_cuda)) errType = "General CUDA error";
        else if(typeid(errCode) == typeid(type_cufft)) errType = "CUFFT error";
        else {
            fprintf(stderr, "Unknown error type\n");
            fflush(stderr);
            exit(EXIT_FAILURE);
        }
        fprintf(stderr, "\"%s\", line %i: %s when calling \"%s\": %d (%s)\n",
                file, line, errType.c_str(),
                func, errCode, cudaGetErrorString(errCode));
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}
#define ABORT(msg) abort( msg, __FILE__, __LINE__ )
#define CUCHK(val) checkCudaErrors( (val), #val, __FILE__, __LINE__ )



//=== global vars and read cmd input
typedef unsigned int uint;
vector<uint> n{1024, 768, 1280};
size_t fft_size = (size_t)n[0] * (size_t)n[1] * (size_t)n[2];
vector<float> d{1.1, 2.2, 3.3};
string data_path = "/lustre04/vol0/zhmukc/projects/cuda/test_mgpu/";



//=== ND Array
struct Array{
public:
    vector<uint> shape; //nz = shape[0], nx = shape[1], ny = shape[2]
    size_t size = 0; //product of shape
    float *data = nullptr;
    
    
    Array(vector<uint> _shape, float val = 0){
        shape = _shape;
        size = 1;
        for (uint extent: shape) size *= extent;
        //        printf("QC: size = %zu\n", size);
        //        int i = 0; for(uint extent : shape) printf("QC: shape[%d] = %u \n", i++, extent);
        CUCHK(cudaMallocManaged(&data, sizeof(float)*size, cudaMemAttachGlobal));
        if (val != 0) {
#pragma omp parallel for
            for(size_t i = 0; i < size; i++) data[i] = val;
        }
    };
    Array(Array& b){
        shape = b.shape;
        size = b.size;
        CUCHK(cudaMallocManaged(&data, sizeof(float)*size, cudaMemAttachGlobal));
#pragma omp parallel for
        for(size_t i = 0; i < size; i++)
            data[i] = b.data[i];
    }
    ~Array(){
        if(data) CUCHK(cudaFree(data));
    };
    
    void checkSize() const{
        if(size == 0) ABORT("size == 0!");
    }
    
    size_t getFlatInd(vector<uint> NDInd) const{
        checkSize();
        if (NDInd.size() != shape.size()) ABORT("NDInd.size() != shape.size()!");
        size_t flat_ind = NDInd.back();
        for(uint i = NDInd.size() - 1; i > 0; i--){
            flat_ind *= shape[i-1];
            flat_ind += NDInd[i-1];
        }
        return flat_ind;
    }
    float getVal(vector<uint> NDInd) const{
        return data[getFlatInd(NDInd)];
    }
    void setVal(vector<uint> NDInd, float val){
        data[getFlatInd(NDInd)] = val;
    }
    void prefetchToGPU(int gpu_device){
        CUCHK(cudaMemPrefetchAsync(data, size*sizeof(float), gpu_device));
    }
    void getDataStats(string label="") const{
        double _mean = 0, _max = FLT_MIN, _min = FLT_MAX, _std2 = 0;
#pragma omp parallel for reduction(max:_max) reduction(min:_min) reduction(+:_mean)
        for(size_t i = 0; i < size; i++){
            float val = data[i];
            _mean += val;
            _max = MAX( _max, (double)val);
            _min = MIN( _min, (double)val);
        }
        _mean /= size;
#pragma omp parallel for reduction(+:_std2)
        for(size_t i = 0; i < size; i++){
            _std2 += (data[i] - _mean)*(data[i] - _mean);
        }
        printf("=== \"%s\" stats ===\n", label.c_str());
        printf("    max = %le, min=%le\n", _max, _min);
        printf("    mean = %le, std2=%le\n", _mean, _std2);
    }
    void dumpData(string file_name = "snaps") const{
        std::ofstream ofs;
        //1. dump DDS dictionary
        ofs.open(data_path + file_name, std::ios::out | std::ios::trunc);
        if(!ofs.is_open()) ABORT("fail to open DDS dictionary " + data_path + file_name);
        ofs << "fmt:SAMPLE_TYPE=typedef float4x SAMPLE_TYPE;\n";
        ofs << "fmt:ASP_TRACE= typedef align(1) struct {SAMPLE_TYPE Samples[axis_size(1)];} ASP_TRACE;\n";
        ofs << "format= asp\n";
        ofs << "size.z=" << shape[0] << "\n";
        ofs << "size.x=" << shape[1] << "\n";
        if(shape.size() == 2) ofs << "axis=z x\n";
        else if(shape.size() == 3) {
            ofs << "axis=z x y\n";
            ofs << "size.y=" << shape[2] << "\n";
        }
        ofs << "data=" << data_path << file_name + "@";
        ofs.close();
        //2. dump DDS binary
        ofs.open(data_path + file_name + "@", std::ios::out | std::ofstream::binary | std::ios::trunc);
        if(!ofs.is_open()) ABORT("fail to open DDS binary " + data_path + file_name + "@");
        ofs.write((const char *)data, size*sizeof(float));
        ofs.close();
    }
};



//=== compute laplacian coefficients
void makeMk2(vector<float>& mk2, int n, float d){
    if ( (mk2.size() != n) && (mk2.size() != (n/2+1)) )
        ABORT("weird size!");
    float T = n*d;
    for(int i = 0; i <= n/2; i++){
        float k = 2*M_PI*i/T;
        mk2[i] = - k*k;
    }
    for(int i = n/2+1; i < mk2.size(); i++){
        mk2[i] = mk2[n - i];
    }
}
void computeLapCoeff(Array& lapCoeff){
    vector<float> mkz2(n[0]/2+1);
    makeMk2(mkz2, n[0], d[0]);
    
    vector<float> mkx2(n[1]);
    makeMk2(mkx2, n[1], d[1]);
    
    vector<float> mky2(n[2]);
    makeMk2(mky2, n[2], d[2]);
    
    float fftScalar = 1.0/(n[0]*n[1]*n[2]);
    
#pragma omp parallel for collapse(2)
    for(uint iy = 0; iy < n[2]; iy++)
        for(uint ix = 0; ix < n[1]; ix++){
            float mk2 = mky2[iy] + mkx2[ix];
            for(uint iz = 0; iz < n[0]/2+1; iz++){
                float val = (mk2 + mkz2[iz])*fftScalar;
                lapCoeff.setVal({iz, ix, iy}, val + iz*0.001 - ix*ix*0.002 + iy/30); //not exactly laplacian, adjusted to enable asymmetry across 3 axes
            }
        }
}
void computeLapCoeff(float* lap[4], int nGPUs, int whichGPUs[4], vector<size_t>& lap_size){ //uniformly distributed along x-axis
    vector<float> mkz2(n[0]/2+1);
    makeMk2(mkz2, n[0], d[0]);
    
    vector<float> mkx2(n[1]);
    makeMk2(mkx2, n[1], d[1]);
    
    vector<float> mky2(n[2]);
    makeMk2(mky2, n[2], d[2]);
    
    float fftScalar = 1.0/(n[0]*n[1]*n[2]);
    
    uint ixs = 0;
    for(int i = 0; i < nGPUs; i++){
        uint nx = n[1]/nGPUs;
        if(i < n[1]%nGPUs) nx++;
        lap_size[i] = (n[0]/2+1)*(size_t)nx*(size_t)n[2];
        CUCHK(cudaMallocManaged(&(lap[i]), sizeof(float)*lap_size[i], cudaMemAttachGlobal));
#pragma omp parallel for collapse(2)
        for(uint iy = 0; iy < n[2]; iy++)
            for(uint ix = 0; ix < nx; ix++){
                uint ix0 = ixs + ix;
                float mk2 = mky2[iy] + mkx2[ix0];
                size_t idx = (iy*nx + ix)*(size_t)(n[0]/2+1);
                for(uint iz = 0; iz < n[0]/2+1; iz++){
                    float val = (mk2 + mkz2[iz])*fftScalar;
                    lap[i][idx+iz] = val + iz*0.001 - ix0*ix0*0.002 + iy/30; //not exactly laplacian, adjusted to enable asymmetry across 3 axes
                }
            }
        CUCHK(cudaMemPrefetchAsync(lap[i], sizeof(float)*lap_size[i], whichGPUs[i]));
        ixs += nx;
        //        cout << "lap: size = " << lap_size[i] << endl;
    }
    //    cout << "ixs = " << ixs << ", n[1] = " << n[1] << endl;
}


__global__ void mulByFFTScalar(float* a, size_t fft_size, size_t a_size){
    int nt = gridDim.x * blockDim.x;
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    while(idx < a_size){
        a[idx] /= fft_size;
        idx += nt;
    }
}


void cleanPadRegion(Array& a){
#pragma omp parallel for
    for(uint iy = 0; iy < n[2]; iy++)
        for(uint ix = 0; ix < n[1]; ix++)
            for(uint iz = n[0]; iz < n[0]+2; iz++)
                a.setVal({iz, ix, iy}, 0);
}


__global__ void applyLap(float2* a, float* lap, size_t fft_size, size_t a_size){
    int nt = gridDim.x * blockDim.x;
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    while(idx < a_size){
        a[idx].x *= lap[idx]/fft_size;
        a[idx].y *= lap[idx]/fft_size;
        idx += nt;
    }
}






