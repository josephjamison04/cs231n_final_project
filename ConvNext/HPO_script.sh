echo "Beginning training script..."


python train.py --use_gpu --batch_size 64 --epochs 1 --option convNext --from_pretrain --small_data


echo "WARNING! SYSTEM WILL SHUTDOWN IN 60 SECONDS UNLESS CTRL-C INTERRUPT..."
sleep 60
# sudo shutdown now
