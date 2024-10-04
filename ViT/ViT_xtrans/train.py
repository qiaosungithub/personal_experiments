try:
    from ViT.utils import *
    from ViT.model import *
except:
    from utils import *
    from model import *
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import os

def train(epochs, model, optimizer, criterion, train_loader, val_loader, outdir):
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    with open(outdir, 'a') as f:
        f.write("the model")
        f.write(str(model) + '\n')
        f.write("the transformer")
        f.write(str(model.transformer) + '\n')
        f.write(str(optimizer) + '\n')

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model.cuda()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # # 使用 DataParallel 将模型并行化
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = torch.nn.DataParallel(model)
    
    model.to(device)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            # labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            progress_bar.set_postfix({'Train Loss': running_loss / (progress_bar.n + 1)})

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # labels = labels.to(device)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
        with open(outdir, 'a') as f:
            f.write("Epoch {i}: train loss {loss}, val loss {vloss}, val accuracy {vacc}, train acc {trainacc}\n".format(i=epoch, loss=train_loss, vloss=val_loss, vacc=val_accuracy, trainacc=train_accuracy))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # return train_losses, val_losses

