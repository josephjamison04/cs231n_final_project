## 20 min total training with code following
**Training ACC: Got 16775 / 64000 correct (26.21)**  
**Val ACC: Got 2920 / 16000 correct (18.25)** 

    python train.py --use_gpu --batch_size 64 --lr 3e-5 --epochs 10 --option trans
    '''model = VisionTransformer(embed_dim = 512,
            hidden_dim = 1024,
            num_channels = 3,
            num_heads = 8,
            num_layers = 6,
            num_classes =100,
            patch_size = 16,
            num_patches = 64,
            dropout=0.2,)'''

## 50 min total training with code following          
## Top-1 Training ACC: Got 21191 / 64000 correct (33.11)
## Top-5 Training ACC: Got 40781 / 64000 correct (63.72)
## Top-1 Val ACC: Got 2911 / 16000 correct (18.19)
## Top-5 Val ACC: Got 6913 / 16000 correct (43.21)

    python train.py --use_gpu --batch_size 64 --lr 3e-5 --epochs 10 --option trans
    '''model = VisionTransformer(embed_dim = 768,
            hidden_dim = 768,
            num_channels = 3,
            num_heads = 8,
            num_layers = 6,
            num_classes =100,
            patch_size = 16,
            num_patches = 64,
            dropout=0.2,)'''

## AlexNet Finetune Experiment (with hyperparameters specified below)
## Training time: ~10 min
## Top-1 Training ACC: Got 18662 / 64000 correct (29.16)
## Top-5 Training ACC: Got 37478 / 64000 correct (58.56)
## Top-1 Val ACC: Got 3296 / 16000 correct (20.60)
## Top-5 Val ACC: Got 7632 / 16000 correct (47.70)

python train.py --use_gpu --batch_size 64 --lr 1e-5 --epochs 3 --option alex