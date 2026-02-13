# CUDA Testbed

A collection of CUDA C++ examples and utilities for testing GPU programming patterns and optimizations.

All matrix multiplication work is inspired from Siboehm's work[1].

## Format Script

`format_cu_files.py` - A Python utility that automatically formats all `.cu` files in the current directory using `clang-format`.

### Usage

```bash
python format_cu_files.py
```

### Requirements

- Python 3.x
- `clang-format` installed and available in PATH

The script will format all CUDA source files in-place and report success or any errors encountered.

# References

[1]: https://siboehm.com/articles/22/CUDA-MMM
