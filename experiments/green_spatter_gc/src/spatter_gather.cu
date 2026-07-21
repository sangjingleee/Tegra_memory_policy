// Spatter pattern 이식 gather: idx 배열 기반 (UNIFORM/scatter)
// out += in[idx[(k+r)&(m-1)]] — Spatter gather 의미, reuse로 cache/DRAM 구분
extern "C" __global__ void gather_reuse(const float* in, const int* idx, float* out, int m, int iters){
  int tid=blockIdx.x*blockDim.x+threadIdx.x, stride=blockDim.x*gridDim.x;
  float acc=0.f;
  for(int r=0;r<iters;++r)
    for(int k=tid;k<m;k+=stride)
      acc += in[ idx[(k+r)&(m-1)] ];
  if(tid<m) out[tid]=acc;
}
