#include <cstdio>
#include <cuda_runtime.h>
int main(){int l2=0,pl2=0,apw=0;cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,0);cudaDeviceGetAttribute(&pl2,cudaDevAttrMaxPersistingL2CacheSize,0);cudaDeviceGetAttribute(&apw,cudaDevAttrMaxAccessPolicyWindowSize,0);printf("GPU L2 cache: 0.00 MB\n",l2/1048576.0);printf("max persisting L2 set-aside: 0.00 MB\n",pl2/1048576.0);printf("max access-policy window: 0.00 MB\n",apw/1048576.0);}
