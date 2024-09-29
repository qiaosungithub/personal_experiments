# [Diffusion Models in Vision: A survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10081412)

- Diffusion 的特点：forward adding Gaussian noise + reverse recover process

- Diffusion 的分类：

1. DDPM. 利用了 non-equilibrium thermodynamics theory, 是一种 latent variable model. 可以认为像 VAE

2. NCSN. 





# [DDPM](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)


# [NCSN](https://proceedings.neurips.cc/paper_files/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf)

- challenge of score matching: score __undefined__ in ambient space & bad langevin in low density region (slow mixing)

- 核心思想：__perturb__ the data with random Gaussian noise of various magnitudes; annealed version of Langevin dynamics