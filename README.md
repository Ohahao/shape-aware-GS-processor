## CODE IMPLEMENTATION: 3DGS accelerator code in PyTorch

This Github is implementing code for following paper ! 

> [1.78mj/frame 373fps 3d gs processor based on shape-aware hybrid architecture using earlier computation skipping and gaussian cache scheduler](https://ieeexplore.ieee.org/document/10904813)

### Quick start 
``` python
python test_gpu.py -s "source dataset path" -m "model dataset path" --iteration "7000 or 30000" --format_type "colmap or blender" --device "device"
```
