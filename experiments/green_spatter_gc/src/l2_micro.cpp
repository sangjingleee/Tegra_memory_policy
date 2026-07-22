// L2 cache-partition CONTROL microbench (we control the addresses, unlike TRT).
// critical = reuse-heavy gather over a `crit-mb` device buffer.
// background = streaming gather over a large buffer (evicts L2).
// --pin 1 : reserve `l2-mb` of L2 as persisting and pin the critical buffer into it.
// Shows the persisting mechanism works when the hot set fits L2, and fails when it doesn't.
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

static const char* cn(CUresult r){ const char* n=nullptr; cuGetErrorName(r,&n); return n?n:"?"; }
#define CK(x) do{ CUresult _r=(x); if(_r!=CUDA_SUCCESS){ std::fprintf(stderr,"%s: %s\n",#x,cn(_r)); return 1; } }while(0)
static int ia(int c,char**v,const char*k,int d){ std::string p=std::string("--")+k+"="; for(int i=1;i<c;i++){std::string a=v[i]; if(a.rfind(p,0)==0) return atoi(a.substr(p.size()).c_str());} return d; }

int main(int argc,char**argv){
  int crit_mb = ia(argc,argv,"crit-mb",2);
  int bg_mb   = ia(argc,argv,"bg-mb",64);
  int iters   = ia(argc,argv,"iters",64);     // critical reuse
  int pin     = ia(argc,argv,"pin",0);
  int l2_mb   = ia(argc,argv,"l2-mb",2);
  int trials  = ia(argc,argv,"trials",50);

  CK(cuInit(0));
  CUdevice dev; CK(cuDeviceGet(&dev,0));
  int apw_max=0; cuDeviceGetAttribute(&apw_max,CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,dev);
  CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx,dev)); CK(cuCtxSetCurrent(ctx));
  CUmodule mod; CK(cuModuleLoad(&mod,"spatter_gather.ptx"));
  CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"gather_reuse"));

  auto mk=[&](int mb, CUdeviceptr&in,CUdeviceptr&idx,CUdeviceptr&out)->int{
    int m=(mb*1024*1024)/4;
    CK(cuMemAlloc(&in,(size_t)m*4)); CK(cuMemAlloc(&idx,(size_t)m*4)); CK(cuMemAlloc(&out,(size_t)m*4));
    std::vector<float> h(m,1.0f); std::vector<int> id(m); for(int i=0;i<m;i++) id[i]=i; // uniform
    CK(cuMemcpyHtoD(in,h.data(),(size_t)m*4)); CK(cuMemcpyHtoD(idx,id.data(),(size_t)m*4));
    return 0;
  };
  CUdeviceptr c_in,c_idx,c_out; if(mk(crit_mb,c_in,c_idx,c_out)) return 1;
  CUdeviceptr b_in,b_idx,b_out; if(mk(bg_mb,b_in,b_idx,b_out)) return 1;
  int c_m=(crit_mb*1024*1024)/4, b_m=(bg_mb*1024*1024)/4;

  CUstream scrit,sbg; CK(cuStreamCreate(&scrit,CU_STREAM_NON_BLOCKING)); CK(cuStreamCreate(&sbg,CU_STREAM_NON_BLOCKING));

  if(pin){
    size_t setaside=(size_t)l2_mb*1024*1024;
    CK(cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE,setaside));
    size_t win=std::min<size_t>((size_t)crit_mb*1024*1024,(size_t)apw_max);
    double hr=std::min(1.0,(double)setaside/(double)win);
    CUstreamAttrValue v; std::memset(&v,0,sizeof(v));
    v.accessPolicyWindow.base_ptr=(void*)c_in; v.accessPolicyWindow.num_bytes=win;
    v.accessPolicyWindow.hitRatio=(float)hr; v.accessPolicyWindow.hitProp=CU_ACCESS_PROPERTY_PERSISTING;
    v.accessPolicyWindow.missProp=CU_ACCESS_PROPERTY_STREAMING;
    CK(cuStreamSetAttribute(scrit,CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,&v));
  }

  int block=256, grid=8;
  auto launch=[&](CUstream s,CUdeviceptr in,CUdeviceptr idx,CUdeviceptr out,int m,int it){
    void* a[]={&in,&idx,&out,&m,&it}; return cuLaunchKernel(fn,grid,1,1,block,1,1,0,s,a,nullptr);
  };
  CUevent e0,e1; CK(cuEventCreate(&e0,0)); CK(cuEventCreate(&e1,0));
  auto bw=[&](bool co)->double{
    // optionally flood background, then time critical
    double best=0;
    for(int t=0;t<trials;t++){
      if(co) for(int q=0;q<4;q++) CK(launch(sbg,b_in,b_idx,b_out,b_m,1));
      CK(cuEventRecord(e0,scrit));
      CK(launch(scrit,c_in,c_idx,c_out,c_m,iters));
      CK(cuEventRecord(e1,scrit)); CK(cuEventSynchronize(e1));
      float ms=0; CK(cuEventElapsedTime(&ms,e0,e1));
      double gb=(double)c_m*4*iters/(ms/1e3)/1e9; if(gb>best) best=gb;
    }
    return best;
  };
  for(int i=0;i<5;i++) CK(launch(scrit,c_in,c_idx,c_out,c_m,iters)); CK(cuStreamSynchronize(scrit));
  double solo=bw(false);
  double co=bw(true);
  std::printf("crit_mb,bg_mb,iters,pin,l2_mb,crit_solo_gbps,crit_co_gbps,preserve_pct\n");
  std::printf("%d,%d,%d,%d,%d,%.3f,%.3f,%.2f\n",crit_mb,bg_mb,iters,pin,l2_mb,solo,co,co/solo*100.0);
  return 0;
}
