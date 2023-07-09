# Cheat Code

This folder provides some code to be easily used in deep learning.



### Package `visdom`

You may need to start a server (usually on `localhost`) for `visdom`:
```
python -m visdom.server -p 6677
```
`-p` specifies the port number. After executing this in command line, you can go to `localhost:6677` to see the visualization.



### Image Registration Codes

The codes are in the registration directory:

-  `visualize.py` is for performance visualization, it can generate checkboard or simply merge two images using transparency
-  `jacobian_determinant.py` is for calculation of Jacobian Determinant, obviously:laughing:.
-  `field_generator.py` is for random smooth elastic transformation generation, and also provides a way to visualize deformation field
- `process_CF_FA.py`, `process_CF_OCTA.py` and `process_FIRE.py` are used to do preprocessing for the datasets
- `all_process_gdbicp.py` is for the `gdb-icp` toolkit, the script can enable processing all images in one run.

Note: The `gdb-icp` toolkit has a bad download location. Find it at [GDB-ICP files](https://www.cs.rpi.edu/research/groups/vision/gdbicp/exec/files/) and choose the version you need. `-invert` is needed especially for multi-modal retinal images.
