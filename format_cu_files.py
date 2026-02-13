#!/usr/bin/env python3
"""
Format all CUDA (.cu) files in the current directory using clang-format.
"""

import subprocess
import os
import sys


def format_cu_files():
    """Find and format all .cu files in the current directory."""
    cu_files = [f for f in os.listdir(".") if f.endswith(".cu")]
    
    if not cu_files:
        print("No .cu files found in the current directory.")
        return
    
    failed = []
    for cu_file in cu_files:
        try:
            print(f"Formatting {cu_file}...", end=" ")
            subprocess.run(
                ["clang-format", "-i", cu_file],
                check=True,
                capture_output=True
            )
            print("✓")
        except subprocess.CalledProcessError as e:
            print("✗")
            failed.append((cu_file, str(e)))
        except FileNotFoundError:
            print("\n✗ clang-format not found. Please install clang-format and try again.")
            sys.exit(1)
    
    if failed:
        print(f"\n{len(failed)} file(s) failed to format:")
        for file, error in failed:
            print(f"  - {file}: {error}")
        sys.exit(1)
    else:
        print(f"\nSuccessfully formatted {len(cu_files)} file(s).")


if __name__ == "__main__":
    format_cu_files()
