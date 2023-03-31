#$ -l tmem=48G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=96:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/EMSSL/

git checkout 02_exps_brain

~/miniconda3/envs/pytorch1.4/bin/python Main.py -c configs/brain2.yaml