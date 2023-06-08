echo "Beginning HPO training script with hyperparamters specified in train.py..."


# python train.py --use_gpu --batch_size 32 --epochs 5 --option swin --from_pretrain
# python train.py --use_gpu --batch_size 32 --epochs 10 --option swin
python train.py --use_gpu --batch_size 32 --epochs 12 --lr 1e-5 --option swin --from_filepath


echo "WARNING! SYSTEM WILL SHUTDOWN IN 60 SECONDS UNLESS CTRL-C INTERRUPT..."
sleep 60
sudo shutdown now
