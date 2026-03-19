#!/bin/bash
#SBATCH -p gpu17,gpu22,gpu24
#SBATCH --gres=gpu:1
#SBATCH -o logs/inner_fvis_%j.out
#SBATCH -e logs/inner_fvis_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=64G

python -u imagenet2txt.py 