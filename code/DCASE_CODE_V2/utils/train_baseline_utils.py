from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    task_id,
    ):
    model.train() # turn to train mode
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    for x, y ,paths in pbar:
        x = x.to(device)    # shape = [B, duartion * samplerate = 128000]
        y = y.to(device)    # shape = [B]

        optimizer.zero_grad()
        logits = model(x, task_id)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct_nums = torch.sum(predictions == y).item()

        total_loss += loss.item() * batch_size
        total_correct += correct_nums
        total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        current_lr = optimizer.param_groups[0]["lr"]

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{avg_acc:.4f}",
            "lr": f"{current_lr:.2e}",
        })

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
        "lr": optimizer.param_groups[0]["lr"],
    }

@torch.no_grad()
def eval_one_epoch(
    model,
    test_loader,
    device,
    epoch,
    task_id,
    ):
    model.eval() # turn to eval mode
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(test_loader, desc=f"Test Epoch {epoch}")
    for x, y ,paths in pbar:
        x = x.to(device)    # shape = [B, duartion * samplerate = 128000]
        y = y.to(device)    # shape = [B]

        # optimizer.zero_grad()
        logits = model(x, task_id)
        loss = criterion(logits, y)
        # loss.backward()
        # optimizer.step()

        batch_size = x.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct_nums = torch.sum(predictions == y).item()

        total_loss += loss.item() * batch_size
        total_correct += correct_nums
        total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{avg_acc:.4f}",
        })

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }

def save_checkpoint_as_state_dict(
        model,
        save_dir,
        domain_id,
    ):
    """
    Strictly follow official DCASE Task 7 baseline checkpoint format.

    Saved file:
        checkpoint_D2.pth
        checkpoint_D3.pth

    Saved content:
        model.state_dict()
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"checkpoint_D{domain_id}.pth"

    torch.save(model.state_dict(), str(save_path))
    print("Checkpoint saved : ", save_path)

def load_checkpoint_as_state_dict(
        model,
        load_dir,
        domain_id,
        device
    ):
    load_path = Path(load_dir) / f"checkpoint_D{domain_id}.pth"
    state_dict = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    print("Checkpoint loaded : ", load_path)
    return model