#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(rtx8000|a100_80)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=168:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/

~/miniconda3/envs/pytorch1.4/bin/python MainSup3D.py -c config/airway1.yaml