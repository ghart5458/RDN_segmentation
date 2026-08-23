#!/usr/bin/env python3
"""Setup script for installing pytorch_wavelets from source.
This handles the special case dependency that needs to be installed from GitHub.
"""
import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        raise


def install_pytorch_wavelets():
    """Install pytorch_wavelets from GitHub source."""
    print("Installing pytorch_wavelets from GitHub source...")

    # Create temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "pytorch_wavelets"

        # Clone the repository
        run_command([
            "git", "clone",
            "https://github.com/fbcotter/pytorch_wavelets.git",
            str(repo_dir)
        ])

        # Install the package using uv
        run_command([
            "uv", "pip", "install", "."
        ], cwd=repo_dir)

        print("pytorch_wavelets installed successfully!")


def main():
    """Main installation function."""
    try:
        # Check if pytorch_wavelets is already installed
        try:
            import pytorch_wavelets
            print("pytorch_wavelets is already installed, skipping...")
            return
        except ImportError:
            pass

        # Check if we're in a virtual environment or uv project
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )

        if not in_venv:
            print("Warning: Not in a virtual environment. Proceeding anyway...")

        # Install pytorch_wavelets
        install_pytorch_wavelets()

        print("Setup completed successfully!")

    except Exception as e:
        print(f"Error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
