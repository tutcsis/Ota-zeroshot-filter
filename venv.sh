#PBS -I -q gEduq
#PBS -l select=1:ncpus=4:mem=32G:ngpus=1
#PBS -v SINGULARITY_IMAGE=imc.tut.ac.jp/transformers-pytorch-cuda118:4.46.3

cd ${PBS_O_WORKDIR}

TORCH_HOME=/work/${LOGNAME}/.cache/torch
TRANSFORMERS_CACHE=/work/${LOGNAME}/.cache/transformers
HF_HOME=/work/${LOGNAME}/.cache/huggingface
UV_CACHE_DIR=/work/${LOGNAME}/.cache/uv
PIP_CACHE_DIR=/work/${LOGNAME}/.cache/pip
TRITON_CACHE_DIR=/work/${LOGNAME}/.cache/triton
export TORCH_HOME TRANSFORMERS_CACHE HF_HOME UV_CACHE_DIR PIP_CACHE_DIR TRITON_CACHE_DIR

#ADHOC FIX to avoid "undefined symbol: cuModuleGetFunction" error.
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs

#ADHOC FIX to avoid torch._inductor.exc.InductorError
#For more detail, see https://github.com/pytorch/pytorch/issues/119054
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

export PROMPT_COMMAND="PS1='(otazeroshotfilter) [\h:\w]\$ '; unset PROMPT_COMMAND"
bash --rcfile .venv/bin/activate