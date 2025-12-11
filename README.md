# PANTHER Challenge Segmentation

This repository contains code for the PANTHER Challenge on biomedical image segmentation using nnUNet and PyTorch.

## Prerequisites

- **Docker** (version 20.10 or later)
- **NVIDIA Docker** (nvidia-docker2) - Required for GPU support
- **NVIDIA GPU** with CUDA support (optional but recommended)

### Installing Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Docker (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Environment Setup

1. **Clone the repository**

   ```bash
   git clone git@github.com:rishuKumar4you/panther.git
   cd panther
   ```

2. **Verify Docker installation**

   ```bash
   docker --version
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

## Building the Docker Container

Build the Docker image using the provided script:

```bash
./do_build.sh
```

Or specify a custom tag:

```bash
./do_build.sh my-custom-tag
```

The default image tag is `panther-task2-baseline`.

## Running the Docker Container

### Test Run

Run the container with test data using the provided script:

```bash
./do_test_run.sh
```

This script will:
- Build the container (if needed)
- Mount `test/input` as `/input` (read-only)
- Mount `test/output` as `/output` (writable)
- Run inference using GPU support

### Manual Run

To run the container manually:

```bash
docker run --rm \
    --platform=linux/amd64 \
    --gpus all \
    --ipc=host \
    --volume /path/to/input:/input:ro \
    --volume /path/to/output:/output \
    panther-task2-baseline
```

**Note:** Ensure your input directory contains the required data files in the expected format.

## Project Structure

```
├── Dockerfile              # Docker container definition
├── requirements.txt        # Python dependencies
├── inference.py           # Main inference script
├── data_utils.py          # Data processing utilities
├── nnUNetTrainer_Xepochs.py  # Custom nnUNet trainer
├── do_build.sh            # Build Docker image script
├── do_test_run.sh         # Test run script
├── do_save.sh             # Save container script
└── README.md              # This file
```

## Saving the Container

To save the Docker image for deployment:

```bash
./do_save.sh
```

This creates a compressed tar.gz file of the container image.

## License
