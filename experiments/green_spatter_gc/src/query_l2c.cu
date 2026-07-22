#include <cstdio>
#include <cuda_runtime.h>
int main() {
  int l2 = 0, pl2 = 0, apw = 0;
  cudaDeviceGetAttribute(&l2,  cudaDevAttrL2CacheSize, 0);
  cudaDeviceGetAttribute(&pl2, cudaDevAttrMaxPersistingL2CacheSize, 0);
  cudaDeviceGetAttribute(&apw, cudaDevAttrMaxAccessPolicyWindowSize, 0);
  printf("L2 total        : %d bytes (%.3f MB)\n", l2, l2/1048576.0);
  printf("max persisting  : %d bytes (%.3f MB)\n", pl2, pl2/1048576.0);
  printf("=> persisting cap = %.2f%% of L2\n", 100.0*pl2/l2);
  printf("max access window: %d bytes (%.2f MB)\n", apw, apw/1048576.0);
  return 0;
}
