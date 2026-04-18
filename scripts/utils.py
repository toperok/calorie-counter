import os
import random
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def set_requires_grad(
        module: nn.Module,
        unfreeze_pattern="",
        verbose=False
):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return
    
    pattern = unfreeze_pattern.split("|")
    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False

def calculate_metrics(preds, targets):
    mae = torch.mean(torch.abs(preds - targets)).item()
    rmse = torch.sqrt(torch.mean((preds - targets)**2)).item()
    # в MAPE я добавляю 1e-7 в знаменатель, чтобы не стремилось к бесконечности при делении на 0.
    mape = torch.mean(torch.abs((targets - preds) / (targets + 1e-7))).item() * 100
    return mae, rmse, mape

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, leave=False, desc="Training")
    
    for batch in pbar:
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        mass = batch['mass'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, mask, image, mass).squeeze()
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.2f}"})
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            mass = batch['mass'].to(device)
            labels = batch['label'].to(device)

            outputs = model(image, input_ids, mask, mass).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    mae, rmse, mape = calculate_metrics(preds, targets)
    avg_loss = total_loss / len(loader)
    
    return avg_loss, mae, rmse, mape