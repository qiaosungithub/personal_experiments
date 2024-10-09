# improve NCSN version 1

we focus on the following aspects:

1. We adjust the model, to use learnable parameters in conditioned instance normalization (CIN) layers, instead of catting the noise scale.

2. We use more channels.

基本上成功了!

- discovery:

both when sampling or denoising from a pretrained model, using `eps=2e-5, T=100` and `eps=8e-5, T=20` have similar results. So we may just use the latter one for efficiency.