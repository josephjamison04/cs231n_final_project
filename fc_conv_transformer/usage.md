## when use fc.py to run, please first activate the env using the following

    conda activate cs231n

## run code 

## for conv layers  

    python train.py --use_gpu --batch_size 64 --lr 1e-3 --epochs 3 --option conv 

## for fc layers

    python train.py --use_gpu --batch_size 64 --lr 1e-6 --epochs 10 --option fc
## for transformer layers

    python train.py --use_gpu --batch_size 64 --lr 3e-5 --epochs 10 --option trans
## for pretrained vit_b16

    
## for AlexNet model

    python train.py --use_gpu --batch_size 64 --lr 1e-5 --epochs 3 --option alex

## for ResNet18 model

    python train.py --use_gpu --batch_size 64 --lr 1e-5 --epochs 3 --option resnet18

## if want to add other stuff, go to fc.py bottom add more arguments, and also see more arguments