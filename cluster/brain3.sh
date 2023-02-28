#$ -l tmem=16G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=48:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/

~/miniconda3/envs/pytorch1.4/bin/python Main.py -c config_brain/brain3.yaml