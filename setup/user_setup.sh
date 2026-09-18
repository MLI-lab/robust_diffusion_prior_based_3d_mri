#!/bin/bash

# warping methods
pip install ninja
export TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"  # A40; set explicitly since build containers may not expose a GPU for torch to auto-detect
pip install --no-binary :all: git+https://github.com/NVlabs/nvdiffrast --no-deps --no-build-isolation
python -c "import nvdiffrast.torch as dr; ctx=dr.RasterizeCudaContext()" # first time compilation takes some time

# install env
pip install -r setup/requirements.txt

# install bart
mkdir -p mri_libs
cd mri_libs
wget https://github.com/mrirecon/bart/archive/refs/tags/v0.6.00.tar.gz
tar -xf v0.6.00.tar.gz
cd bart-0.6.00 && make
cd ../..

# wheels for ismrmd
pip install "git+https://github.com/ismrmrd/ismrmrd-python"
pip install "git+https://github.com/ismrmrd/ismrmrd-python-tools"

FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
FORCE_CUDA=1 pip install "git+https://github.com/samb-t/torchsparse.git"

_ext_build=$(mktemp -d)
for _pkg in simple-knn gaussian-interpolation-complex; do
    cp -r submodules/$_pkg "$_ext_build/$_pkg"
    rm -rf "$_ext_build/$_pkg/build" "$_ext_build/$_pkg"/*.egg-info
    find "$_ext_build/$_pkg" -name "*.so" -delete
    pip install --no-build-isolation --no-deps "$_ext_build/$_pkg"
done
rm -rf "$_ext_build"
python -c "import torch, simple_knn._C, gaussian_rasterizer_complex" \
    || echo "WARNING: local CUDA extensions are not importable" >&2

prefect config set PREFECT_API_URL=http://localhost:4200/api
prefect config set PREFECT_RESULTS_PERSIST_BY_DEFAULT=true

# setup git (required for git-annex -> cc359 download)
# git config --global user.name ???
# git config --global user.email ???