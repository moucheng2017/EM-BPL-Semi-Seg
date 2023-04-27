#$ -l tmem=48G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -wd /SAN/medic/PerceptronHead/codes/EMSSL/

~/miniconda3/envs/pytorch1.4/bin/python Main.py -c configs/exp15.yaml