#$ -l tmem=16G
#$ -l gpu=true,gpu_type=!(gtx1080ti|rtx2080ti)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=168:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/

~/miniconda3/envs/pytorch1.4/bin/python MainSup3D.py -c config/sup3d3c.yaml