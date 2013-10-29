cudaUVABandwidth
================

Benchmarks of speed of using CUDA UVA-transfer vs DMA-controller (CUDA 5.0 and C++03)

Source code requires: CUDA >= 4.0 and C++03

Recomended: CUDA 5.0 and C++03 in MSVS 2010

Comparison bandwidth for memory transfer for blocks **128 Bytes - 128 MB**: CPU-RAM -> GPU-RAM -> CPU-RAM:

- standard MPP-approach with using DMA-controller
- with GPU-Cores by using Unified Virtual Addressing

Result of benchmarks on GeForce GTX460 SE CC2.1 see in: result.txt