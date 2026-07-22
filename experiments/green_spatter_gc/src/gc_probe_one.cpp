// One process = one 8-SM green context. Prints PID + the physical SMs its
// kernel actually ran on. Run two instances concurrently to see whether two
// processes' green contexts coordinate (expect: they DON'T -> same SMs).
#include <cuda.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

static const char* cn(CUresult r){ const char* n=nullptr; cuGetErrorName(r,&n); return n?n:"?"; }
#define CK(x) do{ CUresult _r=(x); if(_r!=CUDA_SUCCESS){ std::fprintf(stderr,"%s:%s\n",#x,cn(_r)); return 2; } }while(0)

int main(int argc, char** argv){
  int sms = argc>1 ? atoi(argv[1]) : 8;
  int reps = argc>2 ? atoi(argv[2]) : 200;   // repeat launches so two procs overlap
  CK(cuInit(0));
  CUdevice dev; CK(cuDeviceGet(&dev,0));
  CUcontext pctx; CK(cuDevicePrimaryCtxRetain(&pctx,dev)); CK(cuCtxSetCurrent(pctx));
  CUmodule mod; CK(cuModuleLoad(&mod,"smid_probe.ptx"));
  CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"probe"));

  CUdevResource all,g,rem; std::memset(&all,0,sizeof(all)); std::memset(&g,0,sizeof(g)); std::memset(&rem,0,sizeof(rem));
  CK(cuDeviceGetDevResource(dev,&all,CU_DEV_RESOURCE_TYPE_SM));
  unsigned n1=1; CK(cuDevSmResourceSplitByCount(&g,&n1,&all,&rem,0,(unsigned)sms));
  CUdevResourceDesc d; CK(cuDevResourceGenerateDesc(&d,&g,1));
  CUgreenCtx gc; CK(cuGreenCtxCreate(&gc,d,dev,CU_GREEN_CTX_DEFAULT_STREAM));
  CUdevResource act; std::memset(&act,0,sizeof(act));
  CK(cuGreenCtxGetDevResource(gc,&act,CU_DEV_RESOURCE_TYPE_SM));
  CUcontext gctx; CK(cuCtxFromGreenCtx(&gctx,gc));
  CUstream st; CK(cuGreenCtxStreamCreate(&st,gc,CU_STREAM_NON_BLOCKING,0));
  CK(cuCtxSetCurrent(gctx));

  int nb=1024;
  CUdeviceptr out; CK(cuMemAlloc(&out,(size_t)nb*4));
  std::set<int> seen;
  std::vector<int> h(nb);
  for(int r=0;r<reps;r++){
    std::vector<int> init(nb,-1);
    CK(cuMemcpyHtoD(out,init.data(),(size_t)nb*4));
    void* a[]={&out,&nb};
    CK(cuLaunchKernel(fn,nb,1,1,64,1,1,0,st,a,nullptr));
    CK(cuStreamSynchronize(st));
    CK(cuMemcpyDtoH(h.data(),out,(size_t)nb*4));
    for(int v:h) if(v>=0) seen.insert(v);
  }
  std::printf("PID %d | greenctx smCount=%u | physical SMs used: ", (int)getpid(), act.sm.smCount);
  for(int v:seen) std::printf("%d ", v);
  std::printf("\n");
  return 0;
}
