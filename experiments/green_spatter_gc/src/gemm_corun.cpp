// cuBLAS GEMM co-run: device-memory GEMM and zero-copy-memory GEMM on a
// Green Context 8:8 split. Same protocol as the gather microbench
// (locked clocks, median, overlap check). Validates that the microbench
// conclusion (high-reuse -> independent co-run) holds for a real AI kernel.
#include <cuda.h>
#include <cublas_v2.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static const char* cn(CUresult r){ const char* n=nullptr; cuGetErrorName(r,&n); return n?n:"?"; }
#define CK(x) do{ CUresult _r=(x); if(_r!=CUDA_SUCCESS){ std::fprintf(stderr,"%s:%s\n",#x,cn(_r)); return 2; } }while(0)
#define BK(x) do{ cublasStatus_t _s=(x); if(_s!=CUBLAS_STATUS_SUCCESS){ std::fprintf(stderr,"cublas err %d @ %s\n",(int)_s,#x); return 3; } }while(0)
static int ia(int c,char**v,const char*k,int d){ std::string p=std::string("--")+k+"="; for(int i=1;i<c;i++){std::string a=v[i]; if(a.rfind(p,0)==0) return atoi(a.substr(p.size()).c_str());} return d; }
static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); size_t n=v.size(); return n? (n%2?v[n/2]:0.5*(v[n/2-1]+v[n/2])):0; }

int main(int argc,char**argv){
  int N=ia(argc,argv,"n",2048), batch=ia(argc,argv,"batch",4), TR=ia(argc,argv,"trials",21);
  size_t bytes=(size_t)N*N*sizeof(float);
  double flop=2.0*N*N*N;                         // per GEMM

  CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev,0));
  int smtot=0; CK(cuDeviceGetAttribute(&smtot,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev));
  CUcontext pctx; CK(cuDevicePrimaryCtxRetain(&pctx,dev)); CK(cuCtxSetCurrent(pctx));

  // green context 8:8
  CUdevResource all,g0,rem,g1; memset(&all,0,sizeof(all));memset(&g0,0,sizeof(g0));memset(&rem,0,sizeof(rem));memset(&g1,0,sizeof(g1));
  CK(cuDeviceGetDevResource(dev,&all,CU_DEV_RESOURCE_TYPE_SM));
  unsigned n1=1; CK(cuDevSmResourceSplitByCount(&g0,&n1,&all,&rem,0,8)); g1=rem;
  CUdevResourceDesc d0,d1; CK(cuDevResourceGenerateDesc(&d0,&g0,1)); CK(cuDevResourceGenerateDesc(&d1,&g1,1));
  CUgreenCtx gc0,gc1; CK(cuGreenCtxCreate(&gc0,d0,dev,CU_GREEN_CTX_DEFAULT_STREAM)); CK(cuGreenCtxCreate(&gc1,d1,dev,CU_GREEN_CTX_DEFAULT_STREAM));
  CUcontext c0,c1; CK(cuCtxFromGreenCtx(&c0,gc0)); CK(cuCtxFromGreenCtx(&c1,gc1));
  CUstream s0,s1; CK(cuGreenCtxStreamCreate(&s0,gc0,CU_STREAM_NON_BLOCKING,0)); CK(cuGreenCtxStreamCreate(&s1,gc1,CU_STREAM_NON_BLOCKING,0));

  // device-path GEMM matrices (all device)
  CK(cuCtxSetCurrent(c0));
  CUdeviceptr dA,dB,dC; CK(cuMemAlloc(&dA,bytes)); CK(cuMemAlloc(&dB,bytes)); CK(cuMemAlloc(&dC,bytes));
  { std::vector<float> h((size_t)N*N,1.0f); CK(cuMemcpyHtoD(dA,h.data(),bytes)); CK(cuMemcpyHtoD(dB,h.data(),bytes)); }
  cublasHandle_t h0; BK(cublasCreate(&h0)); BK(cublasSetStream(h0,(cudaStream_t)s0));

  // zero-copy-path GEMM: A,B host-mapped (read via ZC), C device
  CK(cuCtxSetCurrent(c1));
  void *zAh,*zBh; CK(cuMemHostAlloc(&zAh,bytes,CU_MEMHOSTALLOC_PORTABLE|CU_MEMHOSTALLOC_DEVICEMAP));
  CK(cuMemHostAlloc(&zBh,bytes,CU_MEMHOSTALLOC_PORTABLE|CU_MEMHOSTALLOC_DEVICEMAP));
  { float*a=(float*)zAh,*b=(float*)zBh; for(size_t i=0;i<(size_t)N*N;i++){a[i]=1.0f;b[i]=1.0f;} }
  CUdeviceptr zA,zB,zC; CK(cuMemHostGetDevicePointer(&zA,zAh,0)); CK(cuMemHostGetDevicePointer(&zB,zBh,0)); CK(cuMemAlloc(&zC,bytes));
  cublasHandle_t h1; BK(cublasCreate(&h1)); BK(cublasSetStream(h1,(cudaStream_t)s1));

  const float alpha=1.f,beta=0.f;
  auto gemmDev=[&](){ return cublasSgemm(h0,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,(float*)dA,N,(float*)dB,N,&beta,(float*)dC,N); };
  auto gemmZc =[&](){ return cublasSgemm(h1,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,(float*)zA,N,(float*)zB,N,&beta,(float*)zC,N); };

  CK(cuCtxSetCurrent(c0)); CUevent e0a,e0b; CK(cuEventCreate(&e0a,0)); CK(cuEventCreate(&e0b,0));
  CK(cuCtxSetCurrent(c1)); CUevent e1a,e1b; CK(cuEventCreate(&e1a,0)); CK(cuEventCreate(&e1b,0));

  auto soloDev=[&]()->double{ cuCtxSetCurrent(c0); for(int w=0;w<3;w++) gemmDev(); cuStreamSynchronize(s0);
    std::vector<double> v; for(int t=0;t<TR;t++){ cuEventRecord(e0a,s0); for(int b=0;b<batch;b++) gemmDev(); cuEventRecord(e0b,s0); cuEventSynchronize(e0b);
      float ms=0; cuEventElapsedTime(&ms,e0a,e0b); v.push_back(flop*batch/(ms/1e3)/1e9); } return med(v); };
  auto soloZc=[&]()->double{ cuCtxSetCurrent(c1); for(int w=0;w<3;w++) gemmZc(); cuStreamSynchronize(s1);
    std::vector<double> v; for(int t=0;t<TR;t++){ cuEventRecord(e1a,s1); for(int b=0;b<batch;b++) gemmZc(); cuEventRecord(e1b,s1); cuEventSynchronize(e1b);
      float ms=0; cuEventElapsedTime(&ms,e1a,e1b); v.push_back(flop*batch/(ms/1e3)/1e9); } return med(v); };

  double devSolo=soloDev(), zcSolo=soloZc();

  // co-run
  std::vector<double> dv,zv,ov;
  for(int t=0;t<TR+3;t++){
    auto w0=std::chrono::steady_clock::now();
    cuCtxSetCurrent(c0); cuEventRecord(e0a,s0);
    cuCtxSetCurrent(c1); cuEventRecord(e1a,s1);
    for(int b=0;b<batch;b++){ cuCtxSetCurrent(c0); gemmDev(); cuCtxSetCurrent(c1); gemmZc(); }
    cuCtxSetCurrent(c0); cuEventRecord(e0b,s0);
    cuCtxSetCurrent(c1); cuEventRecord(e1b,s1);
    cuCtxSetCurrent(c0); cuEventSynchronize(e0b);
    cuCtxSetCurrent(c1); cuEventSynchronize(e1b);
    auto w1=std::chrono::steady_clock::now();
    float md=0,mz=0; cuCtxSetCurrent(c0); cuEventElapsedTime(&md,e0a,e0b); cuCtxSetCurrent(c1); cuEventElapsedTime(&mz,e1a,e1b);
    if(t<3) continue;
    double wall=std::chrono::duration<double,std::milli>(w1-w0).count();
    dv.push_back(flop*batch/(md/1e3)/1e9); zv.push_back(flop*batch/(mz/1e3)/1e9); ov.push_back(std::max(md,mz)/wall);
  }
  double devCo=med(dv),zcCo=med(zv),ovl=med(ov);
  double soloSum=devSolo+zcSolo, coSum=devCo+zcCo;
  std::printf("kernel,N,batch,dev_sms,zc_sms,dev_solo_gflops,zc_solo_gflops,dev_co_gflops,zc_co_gflops,solo_sum,co_sum,efficiency_pct,overlap\n");
  std::printf("cublas_sgemm,%d,%d,8,8,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.2f,%.3f\n",
              N,batch,devSolo,zcSolo,devCo,zcCo,soloSum,coSum,coSum/soloSum*100.0,ovl);
  return 0;
}
