A template for a training block

```python
# Here is the hyperparameters

epochs = 50
patch_size = 8
embed_dim = 256
n_layers = 6
heads = 8
attn_dim = 512
mlp_dim = None # default to 4*embed_dim
pool = 'cls'
dropout = 0.0

model = ViT(image_size=32, patch_size=patch_size, num_classes=10, embed_dim=embed_dim, n_layers=n_layers, heads=heads, attn_dim=attn_dim, mlp_dim=mlp_dim, pool=pool, dropout=dropout)

last_time_str = TODO

# load model
model.load_state_dict(torch.load(f"ViT/models{last_time_str}.pth"))

print(f"models loaded from ViT/models{last_time_str}.pth")

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=5e-4)

time_str = timestr()

print(f"Time string: {time_str}")

# print the model and the number of parameters
# print(model.transformer)
print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

train(epochs=epochs, model=model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), 
      train_loader=train_loader, val_loader=val_loader, outdir=get_outdir(time_str))

torch.save(model.state_dict(), f"ViT/models/{time_str}.pth")

print(f"models saved to ViT/models/{time_str}.pth")
```