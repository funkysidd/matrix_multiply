# CUDA Testbed

A collection of CUDA C++ examples for testing GPU programming patterns and optimizations.

All matrix multiplication work is inspired from Siboehm's work[1].

# Build Instructions

```
mkdir -p build && cd $-
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

# References

1. [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
