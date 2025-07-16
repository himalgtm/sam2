# SAM 2.1 – Segment Anything Model (Experimental Setup)

This repository contains scripts, configurations, and test outputs for working with Meta’s Segment Anything Model 2.1 (SAM 2.1). The primary focus is applying SAM 2.1 on both video sequences and high-resolution static imagery for research and geospatial experimentation.

Initial development and testing is being conducted on a Windows 11 machine (CUDA-enabled). Once stable, the environment will be replicated and deployed on a university-managed Mac Studio.

## Features

- SAM 2.1 setup for Windows (manual installation, CUDA 12.6, PyTorch ≥2.3)
- Inference on single images with prompt-based masks
- Support for multi-object segmentation via per-object point sets
- Frame extraction and testing from video input
- Mask overlays and binary mask export (NumPy)
- Output rendering with prompt-point visualization
- Work-in-progress Segment-Geospatial integration


## Status

- Static image segmentation working (single and multi-object)
- Video frame testing complete (via ffmpeg preprocessing)
- Output pipeline stable for overlays and masks
- Segment-Geospatial setup and GIS export in progress
- GitHub repository used for clean transfer to Mac Studio

## Setup Notes

SAM 2.1 does not officially support Windows, so installation requires:

- Visual Studio C++ 2022
- CUDA Toolkit 12.4 or 12.6
- Compatible PyTorch version
- Manual checkpoint downloads
- Environment variable configuration

Details are in the development logs and script headers.

## License

This repository is for research and testing purposes only. Underlying model and training code are maintained by Meta AI (see the [original SAM GitHub](https://github.com/facebookresearch/segment-anything)).
