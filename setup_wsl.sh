#!/usr/bin/env bash
# Set up the DiffusionNet backbone inside WSL/Ubuntu, where robust_laplacian and
# potpourri3d import cleanly (they SEGFAULT on native Windows).
#
# Run from Windows:
#   wsl --install -d Ubuntu          # once, if WSL has no distro (no reboot if the
#                                    # WSL feature is already enabled)
#   wsl -d Ubuntu -e bash -lc 'bash /mnt/c/Users/DE00024082/Desktop/code/setup_wsl.sh'
#
# Notes baked in from a real run on this machine:
#  * fresh Ubuntu lacks pip/venv -> apt installs them first
#  * a corporate TLS-inspecting proxy breaks SSL to some hosts -> --trusted-host
#  * diffusion-net has no setup.py -> clone + add src/ to the venv via a .pth file
set -euo pipefail

PTORCH_TRUST="--trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org"

echo "==> [1/7] System packages (venv/pip/build tools)"
# Skip apt (and sudo) when the tools are already present -- a reset distro may run
# as a NON-root user, where `sudo apt-get` blocks on a password and times out in a
# non-interactive launch. We only need venv + git + pip, which are usually shipped.
if python3 -m venv --help >/dev/null 2>&1 && command -v git >/dev/null 2>&1; then
  echo "    venv+git already present -> skipping apt (no sudo needed)"
else
  sudo apt-get update -qq
  sudo apt-get install -y python3-venv python3-pip build-essential git
fi

echo "==> [2/7] Creating venv ~/cpenv"
python3 -m venv ~/cpenv
source ~/cpenv/bin/activate
pip install --upgrade pip

echo "==> [3/7] Installing numpy/scipy/ijson (PyPI works through the proxy)"
pip install numpy scipy ijson scikit-learn

echo "==> [4/7] Installing torch (GPU/CUDA build -- run_wsl.bat trains with --device cuda)"
# GPU (CUDA 12.x) wheels: the cu128 build pulls cuda-toolkit/nvidia-* libs from
# pypi.nvidia.com, so that host must ALSO be trusted. Do NOT add --no-deps -- it
# skips the CUDA runtime libs and you get 'libcudart.so.12 not found' at import.
pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    $PTORCH_TRUST --trusted-host pypi.nvidia.com 2>&1 | tail -3
# CPU-only alternative (no GPU):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu $PTORCH_TRUST

echo "==> [5/7] Installing native geometry libs (these segfault on Windows)"
pip install robust_laplacian potpourri3d

echo "==> [6/7] Installing diffusion-net (no setup.py -> clone + .pth)"
# git through the proxy needs verification off for this clone
git -c http.sslVerify=false clone --depth 1 \
    https://github.com/nmwsharp/diffusion-net.git ~/diffusion-net 2>&1 | tail -1
SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "$HOME/diffusion-net/src" > "$SP/diffusion_net.pth"

echo "==> [7/7] IMPORT CHECK (the proof the segfault is gone)"
python3 -c "import robust_laplacian, potpourri3d, torch, diffusion_net; \
print('IMPORT OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo ""
echo "==> SETUP DONE. Train the real DiffusionNet backbone on the corpus:"
echo "    source ~/cpenv/bin/activate"
echo "    cd /mnt/c/Users/DE00024082/Desktop/code"
echo "    python3 train_cp.py /mnt/c/Users/DE00024082/Desktop/JSON \\"
echo "        --backbone diffusionnet --device cuda \\"
echo "        --op-cache-dir /mnt/c/Users/DE00024082/Desktop/op_cache \\"
echo "        --epochs 200 --eval-every 10 --patience 6 \\"
echo "        --ckpt /mnt/c/Users/DE00024082/Desktop/cp_model.pt \\"
echo "        --out /mnt/c/Users/DE00024082/Desktop/results_full.json"
echo "    # (or simply: python3 run_full.py --eval-every 10 --patience 6)"
