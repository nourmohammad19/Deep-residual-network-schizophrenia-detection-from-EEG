# src/engine.py
import torch
from tqdm import tqdm 

def train_one_epoch(model, dataloader, optimizer, criterion, device, batch_size):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, total=len(dataloader)):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        image = inputs.to(torch.float32)
        label = targets.to(torch.float32)
        
    
        if image.dim() == 5:
            b, seq, c, h, w = image.shape
            image = image.view(b * seq, c, h, w)

            # Duplicate the patient label for each of the Y images
            label = label.repeat_interleave(seq)

        mini_batch_size = batch_size
        image_batches = torch.split(image, mini_batch_size)
        label_batches = torch.split(label, mini_batch_size)

        # Iterate through the chunks and run the steps
        for img_batch, lbl_batch in zip(image_batches, label_batches):
            
            output = model(img_batch)
            
            loss = criterion(output, lbl_batch.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
  
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, batch_size):
    model.eval()
    running_loss = 0.0

    correct = 0
    total = 0
    
    with torch.no_grad():
        for image, label in dataloader:
            
            # 1. Flatten the 5D tensor into a 4D sequence tensor
            if image.dim() == 5:
                b, seq, c, h, w = image.shape
                image = image.view(b * seq, c, h, w)
                label = label.repeat_interleave(seq)
                
            # 2. Split the sequence into mini-batches to prevent OOM
            mini_batch_size = batch_size
            image_batches = torch.split(image, mini_batch_size)
            label_batches = torch.split(label, mini_batch_size)
            
            # 3. Process each mini-batch
            for img_batch, lbl_batch in zip(image_batches, label_batches):
                
                img_batch = img_batch.to(device)
                lbl_batch = lbl_batch.to(device).long()
                
                output = model(img_batch)
                loss = criterion(img_batch, lbl_batch)
                running_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                
                total += lbl_batch.size(0)
                correct += (lbl_batch == predicted).sum().item()

    return running_loss / len(dataloader), correct / total if total > 0 else 0
