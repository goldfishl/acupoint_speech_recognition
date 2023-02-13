import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch import nn
import time
import random
import os
import pandas as pd
import math

from acpsr.train.dataset import Audio_Embedding, Acupuncture_Prescription, collate_batch, label_padding_value, embedding
from acpsr.train.model import RNN


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        embed, seq_len, trg = batch
        
        optimizer.zero_grad()
        
        output = model(embed, seq_len, batch_infer=True)
        
        output = output.view(-1, output.shape[-1])
        trg = trg.view(-1)
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        if i % 50 == 0:
            loss, current = loss.item(), i
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(iterator):>5d}]")
        
    return epoch_loss / len(iterator)
    

def comput_accuracy(model_out, trg, seq_len=[], batch=True):
    if batch:
        out = torch.argmax(model_out, dim=2)
        acc = 0
        batch_size = model_out.shape[0]
        for _out, _trg, _len in zip(out, trg, seq_len):
            _len = _len.item()
            _out = _out[:_len]
            _trg = _trg[:_len]
            acc += ((_trg == _out).sum() / _len).item()
        return acc / batch_size
    else:
        out = torch.argmax(model_out, dim=1)
        return ((trg == out).sum() / len(out)).item()

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            embed, seq_len, trg = batch

            output = model(embed, seq_len, batch_infer=True)
            acc = comput_accuracy(output, trg, seq_len, batch=True)
            output = output.view(-1, output.shape[-1])
            trg = trg.view(-1)
            loss = criterion(output, trg)

            epoch_loss += loss.item()
            
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def start_train():
    path = os.path.join('./data', 'combined_prescription')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = Acupuncture_Prescription(path, "training")
    val_dataset = Acupuncture_Prescription(path, "validation")
    test_dataset = Acupuncture_Prescription(path, "testing")

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=30, collate_fn=collate_batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=30, collate_fn=collate_batch, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=30, collate_fn=collate_batch, shuffle=True)


    # initalizate model
    model = RNN().to(device)

    # optimizer & criterion
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = label_padding_value)
    
    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_dataloader, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, './models/rnn_segmentator.pth')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')