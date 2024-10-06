import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt, pi
import math
try: 
    from ViT.x_transformers import AttentionLayers
except:
    from x_transformers import AttentionLayers

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(sqrt(2 / pi) * (x + 0.044715 * torch.pow(x, 3))))
    
def SinousoidalPositionalEmbedding(seq_len, embedding_dim):
    position = torch.arange(seq_len).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -math.log(10000.0) / embedding_dim)
    pos_embedding = torch.zeros(seq_len, embedding_dim)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    # return pos_embedding.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))4
    pos_embedding = pos_embedding.unsqueeze(0)
    assert pos_embedding.shape == torch.Size((1, seq_len, embedding_dim))
    return pos_embedding

class tranformer_layer(nn.Module):
    def __init__(self, embed_dim, n_heads, attn_dim, mlp_dim = None, dropout=0.0, mlp_dropout=0.0):
        super(tranformer_layer, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = attn_dim // n_heads
        self.attn_dim = attn_dim
        self.QKV = nn.Linear(embed_dim, attn_dim * 3, bias=False)
        self.fc = nn.Linear(attn_dim, embed_dim, bias=False)
        if mlp_dim is None:
            mlp_dim = embed_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(mlp_dropout)

    def forward(self, x):
        assert x.shape == torch.Size((x.shape[0], x.shape[1], self.embed_dim))
        res = x
        x = self.norm1(x)
        qkv = self.QKV(x)
        q, k, v = qkv.chunk(3, dim=-1)
        assert q.shape == torch.Size((x.shape[0], x.shape[1], self.attn_dim))
        assert k.shape == torch.Size((x.shape[0], x.shape[1], self.attn_dim))
        assert v.shape == torch.Size((x.shape[0], x.shape[1], self.attn_dim))
        q = q.reshape(q.shape[0], q.shape[1], self.n_heads, self.head_dim)
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.head_dim)
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.head_dim)

        attn = torch.einsum('bqnc,bknc->bnqk', q, k) / sqrt(self.head_dim) # c=head_dim
        attn = F.softmax(attn, dim=-1)
        attn_val = torch.einsum('bnqk,bknc->bqnc', attn, v)
        attn_val = attn_val.reshape(attn_val.shape[0], attn_val.shape[1], self.attn_dim)

        x = res + self.dropout1(self.fc(attn_val))
        res = x
        x = self.norm2(x)
        x = res + self.dropout2(self.mlp(x))
        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, n_layers, heads, attn_dim, mlp_dim=None, channels = 3, dropout=0.0, mlp_dropout=0.0, embedding="learnable", use_x_transformers=False):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_dim = patch_dim

        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        if embedding == "learnable":
            self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        elif embedding == "sinousoidal":
            self.positional_embedding = SinousoidalPositionalEmbedding(num_patches + 1, embed_dim)
        else:
            raise NotImplementedError("embedding must be either 'learnable' or 'sinousoidal'")
        self.dropout = nn.Dropout(0.1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        if not use_x_transformers:
            self.transformer = nn.Sequential(*[tranformer_layer(embed_dim, heads, attn_dim, mlp_dim, dropout=dropout, mlp_dropout=mlp_dropout) for _ in range(n_layers)])
        else:
            self.transformer = AttentionLayers(
                dim=embed_dim,
                heads=heads,
                depth=n_layers,
                attn_dim_head=attn_dim // heads,
                attn_dropout=dropout,
                ff_dropout=mlp_dropout,
            )
        
        self.LN = nn.LayerNorm(embed_dim)
        self.to_cls_token = nn.Identity()
        self.fc = nn.Linear(embed_dim, num_classes)
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(embed_dim, mlp_dim),
        #     GELU(),
        #     nn.Linear(mlp_dim, num_classes)
        # )

    def forward(self, img):
        p = self.patch_size
        x = img
        # # test
        # x = torch.arange(3*10*10).reshape(1, 3, 10, 10)
        # p=2
        # self.image_size = 10
        # self.patch_size = 2
        # self.num_patches = 25
        # self.patch_dim = 4*3
        # print("0",x)
        assert x.shape == torch.Size((x.shape[0], 3, self.image_size, self.image_size)), 'Input tensor shape does not match image size'
        x = x.reshape(x.shape[0], 3, self.image_size // p, p, self.image_size // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(-1, self.num_patches, self.patch_dim)
        assert x.shape == torch.Size((x.shape[0], self.num_patches, self.patch_dim))
        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        assert x.shape == torch.Size((x.shape[0], self.num_patches + 1, self.embed_dim))
        self.positional_embedding = self.positional_embedding.cuda()
        x += self.positional_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        assert x.shape == torch.Size((x.shape[0], self.num_patches + 1, self.embed_dim))
        x = x[:, 0]
        assert x.shape == torch.Size((x.shape[0], self.embed_dim))
        x = self.LN(x)
        assert x.shape == torch.Size((x.shape[0], self.embed_dim))

        x = self.to_cls_token(x)
        assert x.shape == torch.Size((x.shape[0], self.embed_dim))
        x = self.fc(x)
        # x = self.mlp_head(x)
        assert x.shape == torch.Size((x.shape[0], 10))

        return x