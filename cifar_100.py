import os
import random
import argparse
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR100
import torch.backends.cudnn as cudnn
import torch.utils.checkpoint as checkpoint_util
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm


def set_all_seeds(seed=42):
    torch.manual_seed(seed)                   # CPU seed
    torch.cuda.manual_seed_all(seed)          # GPU seed(s)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True                # Enforce determinism if desired
    cudnn.benchmark = False

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def get_model(name="resnet18", num_classes=100, use_checkpoint=False):
    # Use torchvision resnets; you can also use timm for more options
    model = getattr(torchvision.models, name)(weights=None)
    # Replace the final head for CIFAR-100 (resnets expect 1000-class)
    if hasattr(model, 'fc'):
        featin = model.fc.in_features
        model.fc = nn.Linear(featin, num_classes)
    else:
        # for other architectures adjust accordingly
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # Optionally apply gradient checkpointing to save memory
    if use_checkpoint:
        # simple wrapper: apply torch.utils.checkpoint to each block if available
        for name, module in model.named_children():
            # naive: only wrap a few known submodules in many architectures (customize)
            pass
        # For torchvision resnet from v0.12+, use model.layerX.__call__ wrappers or use timm with checkpoint_cfg.
    return model

# Example forward hook: capture activations
activations = {}
def forward_hook(name):
    def hook(module, inp, out):
        activations[name] = out.detach()
    return hook


# Training Loop
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Repro
    set_all_seeds(args.seed)

    # Data transforms (Data Handling)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # RandAugment or AutoAugment could be added
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
    ])

    # Datasets & DataLoader (Data Handling)
    root = args.data_dir
    train_ds = CIFAR100(root, train=True, transform=train_transform, download=True)
    test_ds  = CIFAR100(root, train=False, transform=test_transform, download=True)

    # show usage: Subset + ConcatDataset
    if args.debug_subset:
        train_ds = Subset(train_ds, list(range(2000)))  # quick dev

    # Create DataLoader with pin_memory and worker_init_fn
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=(args.num_workers>0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=(args.num_workers>0),
    )

    # Model
    model = get_model(name=args.model, num_classes=100, use_checkpoint=args.grad_checkpoint)
    # move to GPU and use channels_last for speed on conv nets
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)  # Performance hack

    # Optional hook example
    # model.layer4.register_forward_hook(forward_hook("layer4"))  # uncomment for debugging

    # Mixed precision
    scaler = GradScaler(enabled=args.use_amp)

    # Loss (label smoothing option)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    # Optimizer (SGD + weight decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # LR scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_acc = 0.0

    # optionally compile model for speed (PyTorch 2.x)
    try:
        if args.use_compile:
            print("Compiling model with torch.compile()")
            model = torch.compile(model)
    except Exception as e:
        print("torch.compile failed or not available:", e)

    # Load checkpoint if resume
    ckpt_path = Path(args.ckpt_dir)/"latest.pth"
    if args.resume and ckpt_path.exists():
        print("Resuming from checkpoint", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['opt_state'])
        scheduler.load_state_dict(ckpt['sched_state'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)

    writer = SummaryWriter(log_dir=args.logdir)

    # Training + evaluation
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad(set_to_none=True)  # faster gradient zeroing (Training Hacks)

        for i, (images, targets) in pbar:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)

            # Gradient accumulation to emulate larger batch
            with autocast(enabled=args.use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets) / args.accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.accum_steps == 0:
                # optional gradient clipping
                if args.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * args.accum_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(f"Epoch {epoch}/{args.epochs} loss={running_loss/(i+1):.4f} acc={100.*correct/total:.2f}")

        scheduler.step()

        # Save checkpoint (Debugging & Monitoring / Performance Hacks)
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'best_acc': best_acc,
        }
        os.makedirs(args.ckpt_dir, exist_ok=True)
        torch.save(ckpt, ckpt_path)
        if epoch % args.save_every == 0:
            torch.save(ckpt, Path(args.ckpt_dir)/f"epoch_{epoch}.pth")

        # Evaluate
        test_acc = evaluate(model, test_loader, device, args)
        writer.add_scalar("test/acc", test_acc, epoch)
        writer.add_scalar("train/loss", running_loss / len(train_loader), epoch)

        # Keep best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'best_acc': best_acc}, Path(args.ckpt_dir)/"best.pth")

        # Memory summary / debug (Debugging & Monitoring)
        print(torch.cuda.memory_summary(device=None, abbreviated=True))
        if args.clear_cache:
            torch.cuda.empty_cache()

    writer.close()
    print("Training finished. Best acc:", best_acc)

def evaluate(model, loader, device, args):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():  # Faster eval than no_grad
        for images, targets in loader:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    print(f"Validation acc: {acc:.4f}")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--grad-clip', type=float, default=None)
    parser.add_argument('--ckpt-dir', type=str, default='./checkpoints')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-compile', action='store_true')
    parser.add_argument('--grad-checkpoint', action='store_true')
    parser.add_argument('--debug-subset', action='store_true')
    parser.add_argument('--clear-cache', action='store_true')
    args = parser.parse_args()
    train(args)