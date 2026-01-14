#!/bin/ksh 
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N dlbclv1
cd $WORKDIR
source /beegfs/data/work/imvia/in156281/dlbcl/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/dlbcl/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib
python /beegfs/data/work/imvia/in156281/dlbcl/train.py