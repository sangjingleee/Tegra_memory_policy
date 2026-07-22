// Each block records the physical SM id it ran on (%smid).
// Used to prove that, within ONE process, two green contexts map to
// disjoint sets of physical SMs.
extern "C" __global__ void probe(int* out, int nb) {
  int b = blockIdx.x;
  volatile float x = 0.f;
  for (int i = 0; i < 3000; ++i) x += i * 0.5f;   // brief spin so blocks co-reside & spread
  if (b < nb && threadIdx.x == 0) {
    unsigned s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    out[b] = (int)s + (x > -1.f ? 0 : 1);          // keep x live
  }
}
