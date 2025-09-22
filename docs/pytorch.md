# PyTorch

[<img align="right" width=150px src='../res/rackete_2.png'></img>](../README.md)

## Table of Contents
* [Fundamentals](#fundamentals)
* [Workflow](#workflow)
* [Custom Data Loading](#custom-data-loading)
* [Classification](#classification)
* [Computer Vision](#computer-vision)
* [Training Loops & Evaluation](#training-loops--evaluation)
* [Transfer Learning](#transfer-learning)
* [Experiment Tracking](#experiment-tracking)
* [Paper Replicating](#paper-replicating)
* [Deployment](#deployment)
* [Advanced Tips & Tricks](#advanced-tips--tricks)
* [Common Errors & Debugging](#common-errors--debugging)
* [Cheatsheet](#cheatsheet)

<br>

> First base implementation. Add more content/details in future.

<br><br>

---

## Fundamentals

* Tensors, operations, device management, autograd, saving/loading models.
* Example snippets for **tensor creation, basic math, gradients, and conversions to NumPy**.
* Installation:

```bash
pip install torch torchvision torchaudio
```

Basic Code:
```python
import torch
import numpy as np

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tensor creation
x = torch.tensor([[1., 2.], [3., 4.]], device=device)
y = torch.randn(2, 2, requires_grad=True, device=device)

# basic ops
z = x * y + torch.sin(x)

# autograd
loss = z.sum()
loss.backward()  # populates y.grad
print('y.grad:', y.grad)

# convert to numpy
npy = x.cpu().numpy()
# and numpy -> tensor
t = torch.from_numpy(npy).to(device)

# save / load
torch.save({'model_state': {'dummy': 1}, 'optimizer_state': None}, 'checkpoint.pt')
ckpt = torch.load('checkpoint.pt', map_location=device)
```

<br><br>

---

## Workflow

Typical PyTorch workflow:

1. **Data Preparation** – `Dataset` + `DataLoader`.
2. **Model Definition** – subclass `nn.Module`.
3. **Loss & Optimizer** – `nn.CrossEntropyLoss()`, `Adam`, etc.
4. **Training Loop** – iterate batches, compute gradients, optimizer step.
5. **Evaluation** – metrics computation.

Code:
```python
import torch
from torch import nn, optim

# toy dataset
X = torch.randn(1000, 1, 28, 28)
y = torch.randint(0, 10, (1000,))

dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# model
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, n_classes)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# train loop
for epoch in range(3):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch} loss: {running_loss/len(loader):.4f}")

# save
torch.save(model.state_dict(), 'simple_cnn.pt')
# load
model.load_state_dict(torch.load('simple_cnn.pt', map_location=device))
```

<br><br>

---

## Custom Data Loading

* Subclass `Dataset` for custom data:

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

dataset = MyDataset(img_paths, labels, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

More comprehensive example:
```python
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import os

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, extensions={'.jpg', '.png', '.jpeg'}):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of (path, label)
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if os.path.splitext(fname)[1].lower() in extensions:
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# use transforms and dataloader
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# dataset = ImageFolderDataset('data/train', transform=transform)
# compute class weights for balanced sampling example
# labels = [lab for _, lab in dataset.samples]
# class_sample_count = np.bincount(labels)
# weights = 1.0 / class_sample_count
# sample_weights = [weights[lab] for lab in labels]
# sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
# loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
```

<br><br>

---

## Classification

* Binary & multi-class examples.
* Define `SimpleNN` or `CNN` models.
* Example **training loop**:

```python
for epoch in range(epochs):
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
```

Another practical example: training a small ResNet-like model on CIFAR-10 using Torchvision utilities.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# model (transfer-learned tiny ResNet)
model = models.resnet18(pretrained=False, num_classes=10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# training loop (single epoch snippet)
model.train()
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    out = model(xb)
    loss = criterion(out, yb)
    loss.backward()
    optimizer.step()

# evaluate using the evaluate function (below)
```

<br><br>

---

## Computer Vision

* Use `torchvision.datasets` and `transforms`.
* CNN architecture examples.
* Data augmentation with `transforms.Compose`.
* Image segmentation / detection references.

Data & Transform:
```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

CNN Example:
```python
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*15*15, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
```

Another example:
```python
# IoU function for two boxes in xyxy format
def iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

# example pseudo-model usage: forward an image, get boxes and scores
# (for detection, prefer torchvision.models.detection.fasterrcnn_resnet50_fpn)
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model_det = fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()

# run inference snippet
# images = [transform_test(Image.open('img.jpg')).to(device)]
# preds = model_det(images)
# preds[0]['boxes'], preds[0]['scores']
```

<br><br>

---

## Training Loops & Evaluation

* Separate **training** and **evaluation** functions.
* Compute metrics (accuracy, IoU, F1, etc.).
* Use `torch.no_grad()` for evaluation.

Training:
```python
for epoch in range(10):
    for X, y in loader:
        optimizer.zero_grad()
        y_pred = model(X).squeeze()
        loss = criterion(y_pred, y.float())
        loss.backward()
        optimizer.step()
```

Evaluation:
```python
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            total_loss += criterion(y_pred, y).item()
            correct += (y_pred.argmax(dim=1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)
```

Provide robust training & evaluation functions (supports mixed precision and checkpointing).

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        if scaler:
            with autocast():
                out = model(xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total_loss += criterion(out, yb).item()
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return total_loss / len(loader), correct / total

# checkpoint helper
def save_checkpoint(model, optimizer, scheduler, epoch, path='ckpt.pth'):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'sched_state': scheduler.state_dict() if scheduler else None
    }, path)

# usage example in training script
# for epoch in range(start, epochs):
#     train_loss = train_one_epoch(..., scaler=scaler)
#     val_loss, val_acc = evaluate(...)
#     save_checkpoint(...)
```

<br><br>

---

## Transfer Learning

* Freeze pretrained layers.
* Replace classifier head for custom tasks.
* Fine-tuning strategies.

```python
from torchvision import models
resnet = models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
```

Freezing layers, replacing heads, and a small fine-tuning recipe.

```python
from torchvision import models
resnet = models.resnet50(pretrained=True)
# freeze all
for param in resnet.parameters():
    param.requires_grad = False

# replace head
resnet.fc = nn.Linear(resnet.fc.in_features, 5)  # e.g., 5 classes
resnet = resnet.to(device)
# only params in resnet.fc are trainable
optimizer = optim.Adam(resnet.fc.parameters(), lr=1e-3)

# small finetune: unfreeze last block
for name, param in resnet.named_parameters():
    if 'layer4' in name:
        param.requires_grad = True

# lower lr for pretrained params when using two groups
optimizer = optim.SGD([
    {'params': [p for n,p in resnet.named_parameters() if 'layer4' in n], 'lr': 1e-4},
    {'params': resnet.fc.parameters(), 'lr': 1e-3}
], momentum=0.9)
```

<br><br>

---

## Experiment Tracking

Track experiments using:

* **TensorBoard**:
    ```python
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="./logs")
    writer.add_scalar("Loss/train", loss.item(), epoch)
    ```

* **Weights & Biases** integration for automatic logging.
    ```python
    import wandb
    wandb.init(project="my-project")
    wandb.log({"loss": loss.item()})
    ```

TensorBoard and Weights & Biases usage examples.

```python
# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs/exp1')
# log scalars
writer.add_scalar('Loss/train', 0.5, 1)
writer.add_scalar('Accuracy/val', 0.82, 1)
# log model graph (optional, on CPU or small batches)
# writer.add_graph(model, torch.zeros(1,3,224,224))
writer.close()

# Weights & Biases (wandb)
# pip install wandb
import wandb
wandb.init(project='my-project', name='run-1')
# inside training loop
# wandb.log({'train_loss': loss.item(), 'val_acc': val_acc})

# also example: save model artifact
# wandb.save('model.pt')
```

<br><br>

---

## Paper Replicating

Steps to replicate results from research papers:
1. Read the paper carefully.
2. Implement the architecture in PyTorch.
3. Preprocess data exactly as described.
4. Use same hyperparameters if available.
5. Compare metrics to verify replication.

Notice:
* Ensure dataset preprocessing matches the paper.
* Compare metrics against published results.


Checklist + small reproducible training scaffold: set seeds, log hyperparams, freeze randomness.

```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1234)

# Save hyperparams
hparams = {
    'batch_size': 128,
    'lr': 0.1,
    'optimizer': 'SGD',
}
# save hparams to json or log with wandb/tensorboard
```

Tips: always include exact dataset split code and preprocessing. Provide a script to export the random seed and final metric.

<br><br>

---

## Deployment

* **TorchScript** for mobile/production:

```python
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

* **ONNX export**:

```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

* **TorchServe** or REST API with Flask/FastAPI.

TorchScript and ONNX export examples.

```python
# TorchScript
model.eval()
example = torch.randn(1, 3, 224, 224).to(device)
traced = torch.jit.trace(model.cpu(), example.cpu())
traced.save('model_traced.pt')

# ONNX
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model.cpu(), dummy, 'model.onnx', opset_version=12,
                  input_names=['input'], output_names=['output'], dynamic_axes={'input':{0:'batch_size'}})
```

For REST APIs: small FastAPI example to serve a classification model.

```python
# server.py (FastAPI)
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import uvicorn

app = FastAPI()

# load model once
# model.load_state_dict(torch.load('simple_cnn.pt', map_location='cpu'))
# model.eval()

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    # preprocess -> tensor -> model -> postprocess
    return {'pred': 0}

# run: uvicorn server:app --reload --port 8000
```

<br><br>

---

## Advanced Tips & Tricks

* Mixed-precision training (`torch.cuda.amp`).
* Gradient accumulation for large batches.
* Multi-GPU & DistributedDataParallel.
* Optimizer tricks: learning rate schedulers, warmup, weight decay.
* Efficient DataLoader usage (`num_workers`, `pin_memory`).
* Profiling and debugging GPU usage.

```python
# gradient accumulation
accum_steps = 4
optimizer.zero_grad()
for i, (xb, yb) in enumerate(loader):
    out = model(xb)
    loss = criterion(out, yb) / accum_steps
    loss.backward()
    if (i+1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# profiler (simple)
from torch.profiler import profile, record_function, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(torch.randn(1,3,224,224).to(device))
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
```

DDP hint: use `torch.nn.parallel.DistributedDataParallel` with `torch.distributed.launch` or `torchrun`; wrap model and use `DistributedSampler` for DataLoader.

<br><br>

---

## Common Errors & Debugging

* Dimension mismatches (`view`, `reshape` issues).
* GPU memory overflow (`CUDA out of memory`).
* Autograd mistakes (forgetting `requires_grad`).
* Tips for `RuntimeError` debugging.

Short list of mistakes and fixes with code illustrations.

*Dimension mismatch*
```python
# wrong: flatten without keeping batch dim
x = torch.randn(8, 3, 32, 32)
# flatten to (8, 3072) correct:
x_flat = x.view(x.size(0), -1)
```

*CUDA out of memory*
```python
# try reducing batch size or use torch.cuda.empty_cache()
# use fp16
from torch.cuda.amp import autocast
with autocast():
    out = model(x)
```

*Autograd mistakes*
```python
# ensure requires_grad for tensors you want gradients for
w = torch.randn(10, requires_grad=True)
```

<br><br>

---

## Cheatsheet

* Most important topics:
    * Tensor creation & manipulation
    * Common layers (`Conv2d`, `Linear`, `LSTM`)
    * Optimizers & schedulers
    * Loss functions
    * Training / evaluation patterns
    * Saving / loading models
    * Device management & GPU handling

<br><br>

```text
# create tensor on GPU: torch.randn(3,3, device=device)
# move model to GPU: model.to(device)
# zero grad: optimizer.zero_grad()
# detach: x.detach()
# inference: model.eval(); with torch.no_grad(): out = model(x)
```

---

