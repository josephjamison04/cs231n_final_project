#!/usr/bin/env bash

conda create -n cs231n python=3.8
conda activate cs231n

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
<<<<<<< HEAD
=======
pip install torch
pip install sckit-learn
>>>>>>> c783246b8d4fb8ce0eaf509a73857654d7a06d76
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install sklearn==0.0
pip install tokenizers==0.10.1
pip install explainaboard_client==0.0.7
