#$ -l tmem=40G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=168:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/EMSSL/

~/miniconda3/envs/pytorch1.4/bin/python Main.py -c configs/exp9.yaml