# GHAP: Gaussian Herding Across Pens
An official implementation of "Gaussian Herding Across Pens: an optimal transport perspective on global gaussian reduction for 3DGS"




## Setup
The codebase is based on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

The used datasets, MipNeRF360 and Tank & Temple, are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/).

note: we modified the "arguments" and "scene" packages to adapt to our scenario.


## Ways to Run

GHAP includes **2 ways** to make the 3D Gaussians be compact
<!-- #### Option 0 Run all (currently Prune + SH distillation) -->


#### Option 1 pruning during the process of establishing 3DGS object
Users can construct from scratch and jointly compact redundant Gaussians by GHAP in training using the following command
```
bash run_with_construction.sh <scene_name> <source_path> [<compact_ratio>]
```
#### Option 2 pruning a trained 3DGS object
Users can compact a trained 3DGS object by checkpoint

```
bash run_from_pointcloud.sh <scene_name> <source_path> [<ckp_path>] [<sampling_ratio>]
```

