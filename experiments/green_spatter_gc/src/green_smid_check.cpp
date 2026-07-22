// Verify (in a SINGLE process) that Green Context partitions physical SMs:
//   full primary context -> all 16 SMs
//   green gc0 (8 SM)      -> 8 specific physical SMs
//   green gc1 (8 SM)      -> the other 8 SMs (disjoint)
// Disproves "Green Context only works across different processes".
#include <cuda.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>
#include <string>

static const char* cn(CUresult r){ const char* n=nullptr; cuGetErrorName(r,&n); return n?n:"?"; }
#define CK(x) do{ CUresult _r=(x); if(_r!=CUDA_SUCCESS){ std::fprintf(stderr,"%s:%s\n",#x,cn(_r)); return 2; } }while(0)

static std::set<int> run(CUcontext ctx, CUstream st, CUfunction fn, int nb, const char* label){
  cuCtxSetCurrent(ctx);
  CUdeviceptr out; cuMemAlloc(&out, (size_t)nb*4);
  std::vector<int> h(nb,-1); cuMemcpyHtoD(out, h.data(), (size_t)nb*4);
  void* a[]={&out,&nb};
  cuLaunchKernel(fn, nb,1,1, 64,1,1, 0, st, a, nullptr);
  cuStreamSynchronize(st);
  cuMemcpyDtoH(h.data(), out, (size_t)nb*4);
  std::set<int> s(h.begin(), h.end()); s.erase(-1);
  std::printf("%-16s -> %zu distinct physical SMs: ", label, s.size());
  for(int v: s) std::printf("%d ", v); std::printf("\n");
  cuMemFree(out);
  return s;
}

int main(int argc, char** argv){
  int dev_sms=8, nb=1024;
  CK(cuInit(0));
  CUdevice dev; CK(cuDeviceGet(&dev,0));
  int smtot=0; CK(cuDeviceGetAttribute(&smtot, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  CUcontext pctx; CK(cuDevicePrimaryCtxRetain(&pctx,dev)); CK(cuCtxSetCurrent(pctx));
  CUmodule mod; CK(cuModuleLoad(&mod,"smid_probe.ptx"));
  CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"probe"));
  std::printf("PID=%d  total SMs=%d  (single process)\n", (int)getpid(), smtot);

  // green split 8 + remaining (8)
  CUdevResource all,dg,rem,zg; std::memset(&all,0,sizeof(all)); std::memset(&dg,0,sizeof(dg));
  std::memset(&rem,0,sizeof(rem)); std::memset(&zg,0,sizeof(zg));
  CK(cuDeviceGetDevResource(dev,&all,CU_DEV_RESOURCE_TYPE_SM));
  unsigned n1=1; CK(cuDevSmResourceSplitByCount(&dg,&n1,&all,&rem,0,(unsigned)dev_sms));
  zg=rem;
  CUdevResourceDesc d0,d1; CK(cuDevResourceGenerateDesc(&d0,&dg,1)); CK(cuDevResourceGenerateDesc(&d1,&zg,1));
  CUgreenCtx g0,g1; CK(cuGreenCtxCreate(&g0,d0,dev,CU_GREEN_CTX_DEFAULT_STREAM));
  CK(cuGreenCtxCreate(&g1,d1,dev,CU_GREEN_CTX_DEFAULT_STREAM));
  CUdevResource a0,a1; std::memset(&a0,0,sizeof(a0)); std::memset(&a1,0,sizeof(a1));
  CK(cuGreenCtxGetDevResource(g0,&a0,CU_DEV_RESOURCE_TYPE_SM));
  CK(cuGreenCtxGetDevResource(g1,&a1,CU_DEV_RESOURCE_TYPE_SM));
  std::printf("green resource: gc0 smCount=%u  gc1 smCount=%u\n", a0.sm.smCount, a1.sm.smCount);
  CUcontext c0,c1; CK(cuCtxFromGreenCtx(&c0,g0)); CK(cuCtxFromGreenCtx(&c1,g1));
  CUstream s0,s1; CK(cuGreenCtxStreamCreate(&s0,g0,CU_STREAM_NON_BLOCKING,0));
  CK(cuGreenCtxStreamCreate(&s1,g1,CU_STREAM_NON_BLOCKING,0));

  CUstream sf; CK(cuCtxSetCurrent(pctx)); CK(cuStreamCreate(&sf, CU_STREAM_NON_BLOCKING));
  auto full = run(pctx, sf, fn, nb, "full primary");
  auto S0 = run(c0, s0, fn, nb, "green gc0(8SM)");
  auto S1 = run(c1, s1, fn, nb, "green gc1(8SM)");

  std::vector<int> inter;
  for(int v: S0) if(S1.count(v)) inter.push_back(v);
  std::printf("\ngc0 ∩ gc1 = %zu (expect 0 = disjoint)  |  gc0 ∪ gc1 = %zu (expect %d)\n",
              inter.size(), [&]{ std::set<int> u=S0; for(int v:S1)u.insert(v); return u.size(); }(), smtot);
  return 0;
}
