## Création de l'environnement virtuel
```bash
module load python
python3 -m venv venv
source venv/bin/activate
pip3 install --prefix=/work/imvia/in156281/dlbcl/venv -r requirements.txt
export PYTHONPATH=/work/imvia/in156281/dlbcl/venv/lib/python3.9/site-packages:$PYTHONPATH
pip3 list
```
## Alias 'venv'
```bash
mkdir -p /work/imvia/in156281/.cache/matplotlib
mkdir -p /work/imvia/in156281/.cache/wandb
mkdir -p /work/imvia/in156281/.config/wandb
alias venv='module load python && source venv/bin/activate 
                               && export PYTHONPATH=venv/lib/python3.9/site-packages:$PYTHONPATH
                               && export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib
                               && export WANDB_CACHE_DIR=/work/imvia/in156281/.cache/wandb 
                               && export WANDB_CONFIG_DIR=/work/imvia/in156281/.config/wandb'
```

## Same github repo but on lung cancer
https://github.com/DeepMicroscopy/Cox_AMIL?tab=readme-ov-file
