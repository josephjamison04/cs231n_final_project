import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, img_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.zeros(1, img_size // patch_size * img_size // patch_size + 1, embed_dim))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.projection(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positions
        return x

class ImageTransformer(nn.Module):
    def __init__(self, patch_size, img_size, in_chans, embed_dim, num_classes):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size, img_size, in_chans, embed_dim)
        self.transformer = nn.Transformer(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.classifier(x[:, 0])
        return x
    
    