EXE = test_cufft_rc_mgpu
CXX = nvc++
CXXFLAGS = -std=c++11 -fast -mp
NVCCFLAGS = -std=c++11 -arch=sm_70 -O3 -use_fast_math -prec-div=true -lineinfo
NVCCFLAGS += -lcufft -L/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/math_libs/lib64
ENV = OMP_NUM_THREADS=96 KMP_AFFINITY=compact
build:
	nvcc -o $(EXE) $(EXE).cu $(NVCCFLAGS) -ccbin $(CXX) -Xcompiler "$(CXXFLAGS)"
run:
	$(ENV) ./$(EXE)
clean:
	rm -rf $(EXE)
