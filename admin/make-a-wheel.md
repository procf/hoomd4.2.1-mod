# Make a wheel to use this software in Google Colab


## Step 0: Check Software

make sure the "wheel" branch of hoomd4.2.1-mod is up-to-date


## Step 1: Install Docker

On your local machine (Linux or macOS), install Docker: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

Download and launch the desktop app. Once it says “Docker Desktop is running,” you’re good to go.

Verify docker in the terminal:
```bash
docker --version

# if the 'docker' command is not recognized, run this 
# (and then add it to your .bashrc or .vimrc)
#export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

# test that docker works:
docker run hello-world
```


## Step 2: clone the software repo and move to the wheel branch

```bash
# cd to a convenient directory and then:
mkdir hoomd-rheoinf-wheel
cd hoomd-rheoinf-wheel
git clone git@github.com:procf/hoomd4.2.1-mod.git
git checkout -b wheel
git push origin wheel
git push --set-upstream origin wheel
git pull

# create a separate copy of the wheel software
cp -r hoomd-rheoinf-wheel/hoomd4.2.1-mod hoomd-rheoinf
```

make sure these lines are at the top of setup.cfg 
```c++
[tool.scikit-build]
cmake.minimum-version = "3.15"
cmake.args = [
      "-DENABLE_GPU=OFF", 
      "-DENABLE_MPI=OFF", 
      "-DBUILD_HPMC=OFF",
      "-DBUILD_METAL=OFF",
      "-DBUILD_TESTING=OFF"
]
```

and that there is a pyproject.toml file:
```bash
[build-system]
requires = ["scikit-build-core", "setuptools", "wheel", "cmake", "ninja"]
build-backend = "scikit_build_core.build"

[project]
name = "hoomdmod"
version = "4.2.1"
description = "Rheoinformatic Modified HOOMD-blue"
requires-python = ">=3.8"
```



## Step 3: setup the docker container

The wheel must be built with the same architecture and Python version as Colab (usually Python 3.11, Ubuntu 20.04/22.04, x86_64)

*NOTE*: you can check your Linux architecture on any system with the command `cat /etc/os-release` and you can check the available python versions in a Docker container with `ls /opt/python/`

```bash
docker run -it -v $PWD:/io quay.io/pypa/manylinux_2_28_x86_64 bash

# install vim
yum install -y vim

# use Python version 3.11 for compatability with 3.11.11 in Colab
/opt/python/cp311-cp311/bin/python3 -m venv /io/hoomd-wheel-venv-311
source /io/hoomd-wheel-venv-311/bin/activate
/opt/python/cp311-cp311/bin/python3 -m ensurepip
pip install --upgrade pip

# Install Ninja for wheel construction 
#/opt/python/cp311-cp311/bin/pip install build ninja
pip install build ninja

# Install GCC 11.1
yum install -y gcc-toolset-11
source /opt/rh/gcc-toolset-11/enable
gcc --version

# Install CMake 3.26.4
CMAKE_VER=3.26.4
CMAKE_SHORT_VER=3.26
PLATFORM=linux-x86_64
cd /tmp
curl -LO https://github.com/Kitware/CMake/releases/download/v$CMAKE_VER/cmake-$CMAKE_VER-$PLATFORM.tar.gz
tar -xzf cmake-$CMAKE_VER-$PLATFORM.tar.gz
mv cmake-$CMAKE_VER-$PLATFORM /opt/cmake-$CMAKE_VER
export PATH=/opt/cmake-$CMAKE_VER/bin:$PATH
cmake --version

# Install OpenBLAS 0.3.29 (this step takes some time)
yum install -y make openblas-devel wget
cd /tmp
wget https://github.com/xianyi/OpenBLAS/archive/refs/tags/v0.3.29.tar.gz
tar -xzf v0.3.29.tar.gz
cd OpenBLAS-0.3.29
# ensure that libopenblas.a is statically linked into HOOMD extensions
# NOTE: this step takes 5-10 min
make USE_OPENMP=0 DYNAMIC_ARCH=1 NO_SHARED=1 USE_LAPACK=1 LAPACKE=1
#make USE_OPENMP=0 DYNAMIC_ARCH=1 NO_SHARED=0 USE_LAPACK=1 LAPACKE=1
make install PREFIX=/opt/openblas
export CMAKE_PREFIX_PATH=/opt/openblas:$CMAKE_PREFIX_PATH
export CPATH=/opt/openblas/include:$CPATH
export LIBRARY_PATH=/opt/openblas/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/openblas/lib:$LD_LIBRARY_PATH

# Install prereqs from HOOMD-blue
cd /io/hoomd-rheoinf
yes | python3 install-prereq-headers.py

# ensure we skip HPMC and METAL for faster builds, and CUDA because Colab does not have GPUs
#export CMAKE_ARGS="-DENABLE_HPMC=OFF -DENABLE_MESH=OFF -DENABLE_CUDA=OFF -DENABLE_METAL=OFF"
export CMAKE_ARGS="$CMAKE_ARGS -DBLA_STATIC=ON -DBLA_VENDOR=OpenBLAS -DENABLE_HPMC=OFF -DENABLE_METAL=OFF -DENABLE_MESH=OFF -DENABLE_TESTING=OFF -DENABLE_CUDA=OFF -DENABLE_MPI=OFF"
#export CMAKE_ARGS="-DENABLE_HPMC=OFF -DENABLE_METAL=OFF -DENABLE_MESH=OFF -DENABLE_TESTING=OFF -DENABLE_CUDA=OFF -DENABLE_MPI=OFF"

# limit parallelization to a single-thread build to avoid running out of memory
export CMAKE_BUILD_PARALLEL_LEVEL=1
```

## Step 4: compile and build the software into a wheel

This will create the file: dist/hoomdmod-4.2.1-cp311-cp311-linux_x86_64.whl

*NOTE*: this can take ~1 hour

```bash
pip install build
/opt/python/cp311-cp311/bin/python3 -m build
```


## Step 5: post-build checks

```bash
cd dist
/opt/python/cp311-cp311/bin/pip install hoomdmod-4.2.1-*.whl
ldd /opt/python/cp311-cp311/lib/python3.11/site-packages/hoomd/md/_md*.so | grep lapack
# you should NOT see liblapacke.so, liblapack.so, or libopenblas.so — your .so should only be dependent on core system libraries like libm, libstdc++, etc.

# test the wheel container
/opt/python/cp311-cp311/bin/python3 -c "import hoomd; print(hoomd.version)"
```

## Step 6: upload the wheel to the interent

To make the wheel available for use in Colab across session, upload it to a public location on line, like here in this repo.
You can then load it in Google Colab with the command

```python
!pip install https://github.com/procf/hoomd4.2.1-mod/tree/main/admin/hoomd*.whl
```
