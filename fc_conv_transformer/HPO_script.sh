echo "Beginning HPO training script with hyperparamters specified in train.py..."


python train.py --use_gpu --batch_size 32 --epochs 5 --option vit_b16 --from_pretrain
# python train.py --use_gpu --batch_size 32 --epochs 10 --option vit_b16


echo "WARNING! SYSTEM WILL SHUTDOWN IN 60 SECONDS UNLESS CTRL-C INTERRUPT..."
sleep 60
sudo shutdown now
