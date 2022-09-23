# EA_RePOSE

This is the code for our submitted paper to ICRA 2023:

EA-Repose: Efficient and Accurate Feature-metric-based 6D Object Pose Refinement

## Prerequisites
- Python >= 3.6
- Pytorch == 1.9.0
- Torchvision == 0.10.0
- CUDA == 10.1

## Downloads
The dataset parts are from the Downloads part in RePOSE(https://github.com/sh8/RePOSE.git)

## Installation

1. Set up the python environment:
    ```
    $ pip install torch==1.9.0 torchvision==0.10.0
    $ pip install Cython==0.29.17
    $ sudo apt-get install libglfw3-dev libglfw3
    $ pip install -r requirements.txt

    # Install Differentiable Renderer
    $ cd renderer
    $ python3 setup.py install
    ```
2. Compile cuda extensions under `lib/csrc`:
    ```
    ROOT=/path/to/EA_RePOSE
    cd $ROOT/lib/csrc
    export CUDA_HOME="/usr/local/cuda-10.1"
    cd ../camera_jacobian
    python setup.py build_ext --inplace
    cd ../nn
    python setup.py build_ext --inplace
    cd ../fps
    python setup.py
    ```
3. Set up datasets:
    ```
    $ ROOT=/path/to/EA_RePOSE
    $ cd $ROOT/data

    $ ln -s /path/to/linemod linemod
    $ ln -s /path/to/linemod_orig linemod_orig

    $ cd $ROOT/data/model/
    $ unzip pretrained_models.zip

    $ cd $ROOT/cache/LinemodTest
    $ unzip ape.zip benchvise.zip .... phone.zip
 
    ```
