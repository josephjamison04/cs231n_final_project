echo "Beginning HPO training script with hyperparamters specified in train.py..."


python train.py --use_gpu --batch_size 64 --epochs 6 --option convNext 


echo "WARNING! SYSTEM WILL SHUTDOWN IN 60 SECONDS UNLESS CTRL-C INTERRUPT..."
sleep 60
sudo shutdown now
