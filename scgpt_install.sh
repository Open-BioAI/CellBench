# conda create -n scgpt-env python=3.10 pip
# conda create -n scgpt-env python=3.10 pip -c conda-forge -y
conda create -n scgpt python=3.11 -y

conda activate scgpt-env
python --version

conda install conda-forge::cudatoolkit=11.7 conda-forge::cudatoolkit-dev=11.7 conda-forge::cudnn=8.9.7.29
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install packaging cellxgene-census GitPython transformers datasets tensorboard wandb rich scib ipython torchtext scvi-tools
pip install flash-attn'<'1.0.5 --no-build-isolation
