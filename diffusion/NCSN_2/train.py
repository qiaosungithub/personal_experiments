try:
    from NCSN.utils import *
    from NCSN.model import *
except:
    from utils import *
    from model import *
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import os

def cal_noise_level(init, final, steps):
    sigmas = []
    init = np.log(init)
    final = np.log(final)
    for i in range(steps):
        sigmas.append(np.exp(init + (final - init) * i / (steps - 1)))
    # print("noise levels: ", sigmas)
    return sigmas

def save_image(images, filename):
    images = images.clamp(0, 1)
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2, pad_value=1)
    torchvision.utils.save_image(grid, filename)

def train(epochs, model, optimizer, criterion, train_loader, val_loader, sigmas, eps, T, outdir, eval_freq=5, sample_dir='./NCSN/samples/'):
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists("NCSN/models"):
        os.makedirs("NCSN/models")
    if not os.path.exists("NCSN/samples"):
        os.makedirs("NCSN/samples")
    if not os.path.exists("NCSN/training_data"):
        os.makedirs("NCSN/training_data")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    with open(outdir, 'a') as f:
        f.write(str(model) + '\n')
        f.write(str(optimizer) + '\n')
        f.write("sigmas: " + str(sigmas) + '\n')
        f.write("eps: " + str(eps) + '\n')
        f.write("T: " + str(T) + '\n')


    best_matching_loss = float('inf')
    train_losses = []
    # sigmas = cal_noise_level(init_sigma, final_sigma, n_sigma)
    n_sigma = len(sigmas)
    sigmas_t = torch.tensor(sigmas, device=device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for b in progress_bar:
            # print(images)
            # print(images[0])
            # print(images[1])
            # assert False
            images, _ = b
            # print(images.shape)
            # assert False
            sigma_indices = torch.randint(0, n_sigma, (images.size(0),), device=device)
            # sigma_batch_no_extend = sigmas_t[sigma_indices]
            sigma_batch = sigmas_t[sigma_indices].view(-1, 1, 1, 1)

            # add noise to images
            images = images.to(device)
            noise = torch.randn_like(images) * sigma_batch
            images_noise = images + noise
            # print(images_noise.shape)
            # assert False

            optimizer.zero_grad()
            outputs = model(images_noise, sigma_indices)
            outputs = outputs * sigma_batch
            target = - noise / sigma_batch
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Train Loss': running_loss / (progress_bar.n + 1)})

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        print("Epoch {i}: train loss {loss}".format(i=epoch, loss=train_loss))
        # save it to outdir (which is a file)
        with open(outdir, 'a') as f:
            f.write("Epoch {i}: train loss {loss}\n".format(i=epoch, loss=train_loss))

        # save model
        if train_loss < best_matching_loss:
            best_matching_loss = train_loss
        torch.save(model.state_dict(), "NCSN/models/{i}.pth".format(i=epoch))

        if epoch % eval_freq == 0:
            # generate samples
            model.eval()
            with torch.no_grad():
                x = torch.rand(64, 1, 28, 28).to(device)
                samples = langevin(model, x, sigmas, eps, T, clamp=False)
                save_image(samples, sample_dir + "{i}.png".format(i=epoch))


@torch.no_grad()
def make_dataset(model, sigmas, eps, T, n_samples=1000, save=True, save_dir='./NCSN/generated/'):
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    x = torch.rand(n_samples, 1, 28, 28).to(device)
    samples = langevin(model, x, sigmas, eps, T)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if save:
        for j in tqdm(range(n_samples)): 
            torchvision.utils.save_image(samples[j], os.path.join(save_dir, "{:>03d}.png".format(j)))

def langevin(score_model, x, sigmas, eps, T, save=False, epochs=None, clamp=True):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bs = x.shape[0]
    all_samples = []
    
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        alpha = eps * (sigma ** 2) / (sigmas[-1] ** 2)
        for t in range(T):
            noise = torch.randn_like(x).to(device)
            indices = i * torch.ones(bs, dtype=torch.long, device=device)
            assert indices.shape == torch.Size([bs,])
            x = x + alpha / 2 * score_model(x, indices) + np.sqrt(alpha) * noise
            if clamp:
                x = torch.clamp(x, 0, 1)
        if save:
            if i % (len(sigmas) // 10) == (len(sigmas) - 1) % (len(sigmas) // 10):
                all_samples.append(x.clone().cpu())

    if save:
        assert x.shape[0] == 10
        # assert len(sigmas) == 10
        assert epochs is not None
        save_dir = './NCSN/denoising_process/'
        filename = '{:>03d}.png'.format(epochs)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # concatenate all samples
        all_samples = all_samples[-10:]
        all_samples = torch.cat(all_samples, dim=0)
        assert all_samples.shape == torch.Size([100, 1, 28, 28])
        # save the image
        grid = torchvision.utils.make_grid(all_samples, nrow=10, padding=2, pad_value=1)
        torchvision.utils.save_image(grid, os.path.join(save_dir, filename))

    return x
