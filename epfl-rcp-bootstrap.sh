#!/bin/bash

SCRATCH_ROOT=/scratch/mnafez

mkdir -p $SCRATCH_ROOT/.cache

if [ ! -L ~/.cache ]; then
    rm -rf ~/.cache
    ln -s $SCRATCH_ROOT/.cache ~/.cache
fi

[ -L ~/scratch ] || ln -s /scratch/mnafez ~/scratch

grep -qxF 'source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh' ~/.bashrc || \
echo 'source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc

grep -qxF 'conda activate dllm' ~/.bashrc || \
echo 'conda activate dllm' >> ~/.bashrc

grep -qxF '[ -f ~/.bashrc ] && source ~/.bashrc' ~/.bash_profile 2>/dev/null || \
echo '[ -f ~/.bashrc ] && source ~/.bashrc' >> ~/.bash_profile

# Keep the pod alive so you can attach to it
# sleep infinity
sleep 8h
