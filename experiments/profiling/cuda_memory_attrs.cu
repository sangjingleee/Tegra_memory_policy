#include <cuda_runtime.h>
#include <stdio.h>

static void print_attr(cudaDeviceAttr attr, const char* name, int device) {
    int value = -1;
    cudaError_t err = cudaDeviceGetAttribute(&value, attr, device);
    if (err == cudaSuccess) {
        printf("%s=%d\n", name, value);
    } else {
        printf("%s=ERROR:%s\n", name, cudaGetErrorString(err));
    }
}

int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("name=%s\n", prop.name);
    printf("major=%d\n", prop.major);
    printf("minor=%d\n", prop.minor);
    printf("integrated=%d\n", prop.integrated);
    printf("canMapHostMemory=%d\n", prop.canMapHostMemory);
    printf("managedMemory=%d\n", prop.managedMemory);
    printf("concurrentManagedAccess=%d\n", prop.concurrentManagedAccess);
    print_attr(cudaDevAttrHostRegisterSupported, "cudaDevAttrHostRegisterSupported", device);
    print_attr(cudaDevAttrPageableMemoryAccess, "cudaDevAttrPageableMemoryAccess", device);
    print_attr(cudaDevAttrConcurrentManagedAccess, "cudaDevAttrConcurrentManagedAccess", device);
    return 0;
}
