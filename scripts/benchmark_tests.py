#!/usr/bin/env python3
"""
Script to run matrix multiply benchmarks (CuBlas and other variants)
and parse performance metrics from output.
"""

import subprocess
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def find_executables():
    """
    Find and return benchmark executables in the build directory.
    
    Returns:
        List of Path objects to executables
    """
    build_dir = Path(__file__).parent / "build"
    executables = []
    
    if not build_dir.exists():
        print(f"Error: Build directory not found at {build_dir}", file=sys.stderr)
        return executables
    
    # List of executables in desired order
    exe_names = [
        "SharedMemoryCacheBlocking",
        "OneDBlockTiling",
        "TwoDBlockTiling",
        "TwoDBlockTilingTransposed",
        "TwoDBlockTilingTransposedAndVectorized",
        "WarpTiling",
        "CuBlas",
    ]
    
    for exe_name in exe_names:
        exe_path = build_dir / exe_name
        if exe_path.exists():
            executables.append(exe_path)
        else:
            print(f"Warning: {exe_name} executable not found at {exe_path}", file=sys.stderr)
    
    return executables


def run_executable(exe_path, dim_m, dim_n, dim_k, iterations, expected_result):
    """
    Run a benchmark executable with specified matrix dimensions.
    
    Args:
        exe_path: Path to the executable
        dim_m: Rows of matrix A and C
        dim_n: Cols of matrix B and C
        dim_k: Common dimension (cols of A, rows of B)
        iterations: Number of iterations to run
        expected_result: Expected result value
        
    Returns:
        Dictionary containing performance metrics or None if execution fails
    """
    if not exe_path.exists():
        print(f"Error: Executable not found at {exe_path}", file=sys.stderr)
        return None
    
    # Build command with arguments using the new format
    cmd = [str(exe_path), "--dims", str(dim_m), str(dim_n), str(dim_k),
           "--iterations", str(iterations), "--expected-result", str(expected_result)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return None
        
        return parse_output(result.stdout)
        
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None


def parse_output(output):
    """
    Parse CuBlas output to extract performance metrics.
    
    Expected format:
    Duration (milliseconds): 1.2345, Peformance (GFLOPS): 67.8901, Max error: 0.0000
    
    Args:
        output: String output from CuBlas
        
    Returns:
        Dictionary with 'duration_ms', 'gflops', and 'max_error' keys, or None if parsing fails
    """
    metrics = {}
    
    # Pattern to match the output line (handles both "Peformance" and "Performance")
    pattern = r"Best performance \(GFLOPS\):\s*([\d.]+),\s*Max error:\s*([\d.]+)"
    
    for line in output.split('\n'):
        match = re.search(pattern, line)
        if match:
            metrics['gflops'] = float(match.group(1))
            metrics['max_error'] = float(match.group(2))
    
    if not metrics:
        print("Error: Could not parse performance metrics from output", file=sys.stderr)
        print(f"Raw output:\n{output}", file=sys.stderr)
        return None
    
    return metrics


def main():
    """Main entry point."""
    # Define benchmark parameter sets
    # Format: (dim_m, dim_n, dim_k, iterations, expected_result)
    iterations = 25
    parameters = [
        (128, 128, 128, iterations, 128.0),
        (256, 256, 256, iterations, 256.0),
        (512, 512, 512, iterations, 512.0),
        (1024, 1024, 1024, iterations, 1024.0),
        (2048, 2048, 2048, iterations, 2048.0),
        (4096, 4096, 4096, iterations, 4096.0),
    ]
    
    executables = find_executables()
    if not executables:
        print("Error: No benchmark executables found", file=sys.stderr)
        sys.exit(1)
    
    # Run all benchmarks and collect results
    # results[exe_name] = { dimension_str: gflops_value, ... }
    results = {}
    dims_list = [f"{d_m}" for d_m, _, _, _, _ in parameters]
    
    for exe in executables:
        exe_name = exe.name
        results[exe_name] = {}
        print(f"\nRunning {exe_name}...")
        
        for dim_m, dim_n, dim_k, iterations, expected_result in parameters:
            dims_key = f"{dim_m}"
            print(f"  {dims_key} (iterations={iterations})...", end=" ", flush=True)
            
            metrics = run_executable(exe, dim_m, dim_n, dim_k, iterations, expected_result)
            
            if metrics and 'gflops' in metrics:
                results[exe_name][dims_key] = metrics['gflops']
                print(f"GFLOPS: {metrics['gflops']:.4f}")
            else:
                print("Failed")
                results[exe_name][dims_key] = None
    
    if not results:
        print("\nError: No benchmarks completed successfully", file=sys.stderr)
        sys.exit(1)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Colors and markers for different executables - generate based on number of executables
    num_executables = len(results)
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    marker_palette = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'h', '+']
    
    # Extend palettes if needed
    while len(color_palette) < num_executables:
        color_palette.extend(color_palette)
    while len(marker_palette) < num_executables:
        marker_palette.extend(marker_palette)
    
    colors = color_palette[:num_executables]
    markers = marker_palette[:num_executables]
    
    # Extract dimension values as integers for x-axis
    dimensions = [int(d_m) for d_m, _, _, _, _ in parameters]
    
    for idx, exe_name in enumerate(results.keys()):
        exec_results = results[exe_name]
        gflops_values = []
        
        # Collect GFLOPS values in order of dimensions
        for dim_str in dims_list:
            gflops = exec_results.get(dim_str)
            if gflops is not None:
                gflops_values.append(gflops)
            else:
                gflops_values.append(np.nan)
        
        # Plot actual points
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.scatter(dimensions, gflops_values, label=exe_name, color=color, marker=marker, s=100, zorder=3)
        
        # Connect the dots with a line
        plt.plot(dimensions, gflops_values, color=color, linewidth=2, zorder=2)
    
    plt.xlabel('Matrix Dimension (M = N = K)', fontsize=12, fontweight='bold')
    plt.ylabel('Performance (GFLOPS)', fontsize=12, fontweight='bold')
    plt.title('Comparisons (RTX 5060 Ti)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, zorder=1)
    plt.xscale('log')
    plt.xticks(dimensions, dimensions)
    
    plt.tight_layout()
    plt.savefig('results/benchmark_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'benchmark_results.png'")
    plt.show()
    
    sys.exit(0)


if __name__ == "__main__":
    main()
