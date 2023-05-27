## when use fc.py to run, please first activate the env using the following

    conda activate cs231n

## run code 

## for conv layers  

    python train.py --use_gpu --batch_size 64 --lr 1e-3 --epochs 3 --option conv 

## for fc layers

    python train.py --use_gpu --batch_size 64 --lr 1e-6 --epochs 10 --option fc

## if want to add other stuff, go to fc.py bottom add more arguments, and also see more arguments