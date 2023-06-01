# Experiment Results

# Vision Transformers

## Experiment 1

20 min total training with code following

- Top-1 Training ACC: Got 16775 / 64000 correct (26.21)
- Top-5 Training ACC: Got 35209 / 64000 correct (55.01)
- Top-1 Val ACC: Got 2920 / 16000 correct (18.25)
- Top-5 Val ACC: Got 6943 / 16000 correct (43.39)

    ```python train.py --use_gpu --batch_size 64 --lr 3e-5 --epochs 10 --option trans```
    model = VisionTransformer(embed_dim = 512,
            hidden_dim = 1024,
            num_channels = 3,
            num_heads = 8,
            num_layers = 6,
            num_classes =100,
            patch_size = 16,
            num_patches = 64,
            dropout=0.2)

## Experiment 2

50 min total training with code following          
 - Top-1 Training ACC: Got 21191 / 64000 correct (33.11)
 - Top-5 Training ACC: Got 40781 / 64000 correct (63.72)
 - Top-1 Val ACC: Got 2911 / 16000 correct (18.19)
 - Top-5 Val ACC: Got 6913 / 16000 correct (43.21)

    ```python train.py --use_gpu --batch_size 64 --lr 3e-5 --epochs 10 --option trans```
    model = VisionTransformer(embed_dim = 768,
            hidden_dim = 768,
            num_channels = 3,
            num_heads = 8,
            num_layers = 6,
            num_classes =100,
            patch_size = 16,
            num_patches = 64,
            dropout=0.2)

---

# AlexNet

## Experiment 1 - Finetune

Training time: ~10 min

 - Top-1 Training ACC: Got 18662 / 64000 correct (29.16)
 - Top-5 Training ACC: Got 37478 / 64000 correct (58.56)
 - Top-1 Val ACC: Got 3296 / 16000 correct (20.60)
 - Top-5 Val ACC: Got 7632 / 16000 correct (47.70)

python train.py --use_gpu --batch_size 64 --lr 1e-5 --epochs 3 --option alex

---

# ConvNext

## Experiment 1 - without finetuning

Training time ~ 30 minutes 

 - Top-1 Training ACC: Got 3836 / 64000 correct (5.99)
 - Top-5 Training ACC: Got 13670 / 64000 correct (21.36)
 - Top-1 Val ACC: Got 916 / 16000 correct (5.73)
 - Top-5 Val ACC: Got 3259 / 16000 correct (20.37)

```python train.py --use_gpu --batch_size 64 --lr 1e-5 --epochs 3 --option convNext```

## Experiment 2 - without finetuning

Training Time ~ 30 min

 - Top-1 Training ACC: Got 9700 / 64000 correct (15.16)
 - Top-5 Training ACC: Got 25373 / 64000 correct (39.65)
 - Top-1 Val ACC: Got 2240 / 16000 correct (14.00)
 - Top-5 Val ACC: Got 5933 / 16000 correct (37.08)

```python train.py --use_gpu --batch_size 64 --lr 1e-4 --epochs 3 --option convNext```

## Experiment 3 - without finetuning

Training Time ~70 minutes

 - Top-1 Training ACC: Got 28021 / 64000 correct (43.78)
 - Top-5 Training ACC: Got 47840 / 64000 correct (74.75) 
 - Top-1 Val ACC: Got 3373 / 16000 correct (21.08)
 - Top-5 Val ACC: Got 7696 / 16000 correct (48.10)

- Val accuracy decreased slightly after 7 epochs while training still increased -> overfitting

```python train.py --use_gpu --batch_size 64 --lr 1e-4 --epochs 10 --option convNext```

## Experiment 4 - from pretrained weigths, with finetuning

Training Time ~60 minutes

### At epoch 3:
- Top-1 Training ACC: Got 50590 / 64000 correct (79.05)
- Top-5 Training ACC: Got 61443 / 64000 correct (96.00)
- Top-1 Val ACC: Got 8236 / 16000 correct (51.48)
- Top-5 Val ACC: Got 12998 / 16000 correct (81.24)

### At epoch 10:
 - Top-1 Training ACC: Got 62708 / 64000 correct (97.98)
 - Top-5 Training ACC: Got 63982 / 64000 correct (99.97)
 - Top-1 Val ACC: Got 7975 / 16000 correct (49.84)
 - Top-5 Val ACC: Got 12532 / 16000 correct (78.33)

```python train.py --use_gpu --batch_size 64 --lr 1e-4 --epochs 10 --option convNext --from_pretrain```

---

# ResNet

## Experiment 1 - without finetuning

Training time: 7 mins.

- Top-1 Training ACC: Got 31760 / 64000 correct (49.62)
- Top-5 Training ACC: Got 50808 / 64000 correct (79.39)
- Top-1 Val ACC: Got 6732 / 16000 correct (42.08)
- Top-5 Val ACC: Got 11648 / 16000 correct (72.80)

```python train.py --use_gpu --batch_size 64 --lr 1e-5 --epochs 3 --option resnet```

## Experiment 2 - without finetuning

```python train.py --use_gpu --batch_size 64 --lr 1e-5 --epochs 10 --option resnet```

