#include <cuda.h>

#include <cstdio>
#include <cstdlib>

static const char* cn(CUresult r) {
  const char* n = nullptr;
  cuGetErrorName(r, &n);
  return n ? n : "UNKNOWN";
}

#define CK(x)                                                                    \
  do {                                                                           \
    CUresult _r = (x);                                                           \
    if (_r != CUDA_SUCCESS) {                                                    \
      std::fprintf(stderr, "%s:%s\n", #x, cn(_r));                              \
      std::exit(2);                                                              \
    }                                                                            \
  } while (0)

int main() {
  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  int sm_count = 0;
  CK(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  std::printf("%d\n", sm_count);
  return 0;
}

