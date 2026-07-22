// Single-path gather runner (device OR zero-copy), for SCF-PMU attribution.
// Runs the gather kernel `repeat` times so SCF counters can be read over a
// clean, kernel-dominated window. Prints achieved GB/s.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char* cn(CUresult r){ const char* n=nullptr; cuGetErrorName(r,&n); return n?n:"?"; }
#define CK(x) do{ CUresult _r=(x); if(_r!=CUDA_SUCCESS){ std::fprintf(stderr,"%s:%s\n",#x,cn(_r)); return 2; } }while(0)
static int ia(int c,char**v,const char*k,int d){ std::string p=std::string("--")+k+"="; for(int i=1;i<c;i++){std::string a=v[i]; if(a.rfind(p,0)==0) return atoi(a.substr(p.size()).c_str());} return d; }
static std::string sa(int c,char**v,const char*k,const char*d){ std::string p=std::string("--")+k+"="; for(int i=1;i<c;i++){std::string a=v[i]; if(a.rfind(p,0)==0) return a.substr(p.size());} return d; }

int main(int argc,char**argv){
  int MB=ia(argc,argv,"mb",16), IT=ia(argc,argv,"iters",8), repeat=ia(argc,argv,"repeat",2000), stride=ia(argc,argv,"stride",17);
  int gc_sms=ia(argc,argv,"gc-sms",0);   // >0: run inside an N-SM green context
  std::string pat=sa(argc,argv,"pat","uniform"), mem=sa(argc,argv,"mem","device");
  int m=(MB*1024*1024)/4; size_t bytes=(size_t)m*4;

  CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev,0));
  CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx,dev)); CK(cuCtxSetCurrent(ctx));
  CUstream stream=nullptr;
  int gc_half=ia(argc,argv,"gc-half",0);   // 0: first partition (SM 0..), 1: remaining (upper SMs)
  if(gc_sms>0){
    CUdevResource all,g,rem; memset(&all,0,sizeof(all)); memset(&g,0,sizeof(g)); memset(&rem,0,sizeof(rem));
    CK(cuDeviceGetDevResource(dev,&all,CU_DEV_RESOURCE_TYPE_SM));
    unsigned n1=1; CK(cuDevSmResourceSplitByCount(&g,&n1,&all,&rem,0,(unsigned)gc_sms));
    CUdevResourceDesc d; CK(cuDevResourceGenerateDesc(&d, gc_half? &rem : &g, 1));
    CUgreenCtx gc; CK(cuGreenCtxCreate(&gc,d,dev,CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext gctx; CK(cuCtxFromGreenCtx(&gctx,gc));
    CK(cuCtxSetCurrent(gctx)); ctx=gctx;
    CK(cuGreenCtxStreamCreate(&stream,gc,CU_STREAM_NON_BLOCKING,0));
  }
  CUmodule mod; CK(cuModuleLoad(&mod,"spatter_gather.ptx"));
  CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"gather_reuse"));

  std::vector<int> idx(m);
  if(pat=="uniform") for(int i=0;i<m;i++) idx[i]=i;
  else if(pat=="strided") for(int i=0;i<m;i++) idx[i]=(int)(((long long)i*stride)%m);
  else for(int i=0;i<m;i++) idx[i]=(int)(((unsigned long long)i*2654435761ull)%m);

  CUdeviceptr in,id,out;
  if(mem=="device"){
    CK(cuMemAlloc(&in,bytes)); CK(cuMemAlloc(&id,bytes)); CK(cuMemAlloc(&out,bytes));
    std::vector<float> h(m,1.0f); CK(cuMemcpyHtoD(in,h.data(),bytes)); CK(cuMemcpyHtoD(id,idx.data(),bytes));
  } else { // zero-copy mapped host memory
    void *inh,*idh; CK(cuMemHostAlloc(&inh,bytes,CU_MEMHOSTALLOC_PORTABLE|CU_MEMHOSTALLOC_DEVICEMAP));
    CK(cuMemHostAlloc(&idh,bytes,CU_MEMHOSTALLOC_PORTABLE|CU_MEMHOSTALLOC_DEVICEMAP));
    float* pin=(float*)inh; for(int i=0;i<m;i++) pin[i]=1.0f;
    int* pid=(int*)idh; for(int i=0;i<m;i++) pid[i]=idx[i];
    CK(cuMemHostGetDevicePointer(&in,inh,0)); CK(cuMemHostGetDevicePointer(&id,idh,0));
    CK(cuMemAlloc(&out,bytes));
  }
  int block=256, grid=(m+block-1)/block;
  auto launch=[&](){ void* a[]={&in,&id,&out,&m,&IT}; return cuLaunchKernel(fn,grid,1,1,block,1,1,0,stream,a,nullptr); };
  for(int w=0;w<5;w++) CK(launch()); CK(cuCtxSynchronize());

  struct timespec t0, t1;
  CUevent e0,e1; CK(cuEventCreate(&e0,0)); CK(cuEventCreate(&e1,0));
  clock_gettime(CLOCK_REALTIME, &t0);
  CK(cuEventRecord(e0,stream));
  for(int r=0;r<repeat;r++) CK(launch());
  CK(cuEventRecord(e1,stream)); CK(cuEventSynchronize(e1));
  clock_gettime(CLOCK_REALTIME, &t1);
  float ms=0; CK(cuEventElapsedTime(&ms,e0,e1));
  double gb=(double)m*4*IT*repeat/(ms/1e3)/1e9;
  std::printf("ONE mem=%s pat=%s mb=%d iters=%d repeat=%d gc_sms=%d time_ms=%.1f GBps=%.1f epoch_start=%.3f epoch_end=%.3f\n",
              mem.c_str(),pat.c_str(),MB,IT,repeat,gc_sms,ms,gb,
              t0.tv_sec+t0.tv_nsec/1e9, t1.tv_sec+t1.tv_nsec/1e9);
  return 0;
}
