## first activate the env using the following

    conda activate cs231n

## run code 

## for convNext model  

    python train.py --use_gpu --batch_size 64 --lr 1e-3 --epochs 3 --option convNext --small_data

## with pretrained weights
    python train.py --use_gpu --batch_size 64 --lr 1e-4 --epochs 10 --option convNext --from_pretrain