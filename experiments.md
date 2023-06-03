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

## Experiment 1 - start with pretrained weights then finetune

Training time: ~10 min

 - Top-1 Training ACC: Got 18662 / 64000 correct (29.16)
 - Top-5 Training ACC: Got 37478 / 64000 correct (58.56)
 - Top-1 Val ACC: Got 3296 / 16000 correct (20.60)
 - Top-5 Val ACC: Got 7632 / 16000 correct (47.70)

```python train.py --use_gpu --batch_size 64 --lr 1e-5 --epochs 3 --option alex```

## Experiment 2 - start with pretrained weights then finetune

Training time: 1 min per epoch

- Top-1 Training ACC: Got 30151 / 64000 correct (47.11)
- Top-5 Training ACC: Got 49277 / 64000 correct (77.00)
- Top-1 Val ACC: Got 4304 / 16000 correct (26.90)
- Top-5 Val ACC: Got 9073 / 16000 correct (56.71)

- Overfitting after epoch 8

```python train.py --use_gpu --batch_size 64 --lr 1e-4 --epochs 10 --option alex```

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

## Experiment 4 - from pretrained weights, with finetuning

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

# ResNet18

## Experiment 1 - start with pretrained weights then finetune

Training time: ~3 mins. per epoch

### At epoch 4:
- Top-1 Training ACC: Got 35121 / 64000 correct (54.88)
- Top-5 Training ACC: Got 53314 / 64000 correct (83.30)
- Top-1 Val ACC: Got 7160 / 16000 correct (44.75)
- Top-5 Val ACC: Got 12075 / 16000 correct (75.47)

### At epoch 7:
- Top-1 Training ACC: Got 43268 / 64000 correct (67.61)
- Top-5 Training ACC: Got 58188 / 64000 correct (90.92)
- Top-1 Val ACC: Got 7698 / 16000 correct (48.11)
- Top-5 Val ACC: Got 12541 / 16000 correct (78.38)

### At epoch 10:
- Top-1 Training ACC: Got 50704 / 64000 correct (79.22)
- Top-5 Training ACC: Got 61210 / 64000 correct (95.64)
- Top-1 Val ACC: Got 7753 / 16000 correct (48.46)
- Top-5 Val ACC: Got 12610 / 16000 correct (78.81)

- No overfitting even after 10 epochs but val accuracy increases very slowly

```python train.py --use_gpu --batch_size 64 --lr 1e-5 --epochs 10 --option resnet18```

## Experiment 2

### At epoch 2: 
- Top-1 Training ACC: Got 47846 / 64000 correct (74.76)
- Top-5 Training ACC: Got 60269 / 64000 correct (94.17)
- Top-1 Val ACC: Got 7917 / 16000 correct (49.48)
- Top-5 Val ACC: Got 12714 / 16000 correct (79.46)

### At epoch 4:
- Top-1 Training ACC: Got 60517 / 64000 correct (94.56)
- Top-5 Training ACC: Got 63853 / 64000 correct (99.77)
- Top-1 Val ACC: Got 7609 / 16000 correct (47.56)
- Top-5 Val ACC: Got 12380 / 16000 correct (77.38)

- Early stop at epoch 4: val accuracy falling after epoch 3, overfitting

```python train.py --use_gpu --batch_size 64 --lr 1e-4 --epochs 10 --option resnet18```

## Experiment 3

### At epoch 3: 
- Top-1 Training ACC: Got 51760 / 64000 correct (80.88)
- Top-5 Training ACC: Got 61552 / 64000 correct (96.17)
- Top-1 Val ACC: Got 7958 / 16000 correct (49.74)
- Top-5 Val ACC: Got 12726 / 16000 correct (79.54)

### At epoch 5:
- Top-1 Training ACC: Got 62215 / 64000 correct (97.21)
- Top-5 Training ACC: Got 63886 / 64000 correct (99.82)
- Top-1 Val ACC: Got 7708 / 16000 correct (48.18)
- Top-5 Val ACC: Got 12420 / 16000 correct (77.62)

- Early stop at epoch 5: val accuracy falling after epoch 4, overfitting

```python train.py --use_gpu --batch_size 64 --lr 5e-5 --epochs 10 --option resnet18```

---

# ResNet50

## Experiment 1 - start with pretrained weights then finetune

Training time: 6 mins. per epoch

### At epoch 9:
- Top-1 Training ACC: Got 53790 / 64000 correct (84.05)
- Top-5 Training ACC: Got 62198 / 64000 correct (97.18)
- Top-1 Val ACC: Got 8682 / 16000 correct (54.26)
- Top-5 Val ACC: Got 13346 / 16000 correct (83.41)

- Overfitting after 9 epochs 

```python train.py --use_gpu --batch_size 64 --lr 1e-5 --epochs 10 --option resnet50```

## Experiment 2 - start with pretrained weights then finetune

Training time: 6 mins. per epoch

### At epoch 3:
- Top-1 Training ACC: Got 56847 / 64000 correct (88.82)
- Top-5 Training ACC: Got 63024 / 64000 correct (98.47)
- Top-1 Val ACC: Got 8995 / 16000 correct (56.22)
- Top-5 Val ACC: Got 13592 / 16000 correct (84.95)

- Overfitting starting epoch 4, early stop

```python train.py --use_gpu --batch_size 64 --lr 5e-5 --epochs 10 --option resnet50```