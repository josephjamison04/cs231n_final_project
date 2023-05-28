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

## 30 min total training with code following          
**Training ACC: Got 21191 / 64000 correct (33.11)**  
**Val ACC: Got 2911 / 16000 correct (18.19)**

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
