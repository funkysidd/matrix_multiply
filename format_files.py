"""
Formats C++/CUDA files in the current directory using clang-format.
"""

import subprocess
import os
import sys


def format_files():
    file_extensions = (".cu", ".h")
    files_to_format = [f for f in os.listdir(".") if f.endswith(file_extensions)]
    program_name = "clang-format-20"
    
    if not files_to_format:
        print("No files found in the current directory.")
        return
    
    failed = []
    for file_to_format in files_to_format:
        try:
            print(f"Formatting {file_to_format}...", end=" ")
            subprocess.run(
                [program_name, "-i", file_to_format],
                check=True,
                capture_output=True
            )
            print("✓")
        except subprocess.CalledProcessError as e:
            print("✗")
            failed.append((file_to_format, str(e)))
        except FileNotFoundError:
            print(f"\n✗ {program_name} not found. Please install and try again.")
            sys.exit(1)
    
    if failed:
        print(f"\n{len(failed)} file(s) failed to format:")
        for file, error in failed:
            print(f"  - {file}: {error}")
        sys.exit(1)
    else:
        print(f"\nSuccessfully formatted {len(files_to_format)} file(s).")


if __name__ == "__main__":
    format_files()
