"""
Author: Michael Kurilko
Date: 6/3/2025 (Torch version 6/5/2025)
Description: This module contains the TransformerModel class, which provides
transformer-based similarity scorer using TORCH!
"""
import collections
import collections.abc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import islice
from torch.utils.data import Dataset, DataLoader, IterableDataset

collections.Container = collections.abc.Container

class PreEncodedDataset(IterableDataset):
    def __init__(self, subject_pairs, labels, model, max_samples=None):
        self.model = model
        self.subject_pairs = subject_pairs
        self.labels = labels
        self.max_samples = max_samples

    def __iter__(self):
        pairs_iter = iter(self.subject_pairs)
        labels_iter = iter(self.labels)
        
        generator = (
            (
                self.model._byte_encode(subj1),
                self.model._byte_encode(subj2),
                label
            )
            for idx, ((subj1, subj2), label) in enumerate(zip(pairs_iter, labels_iter))
            if self.max_samples is None or idx < self.max_samples
         )
        return islice(generator, self.max_samples) if self.max_samples else generator

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, embedding_dim=64, vocab_size=259, nhead=4, num_layers=2, max_len=128):
        super().__init__()
        self.feature_list = [
            "first_name", "middle_name", "last_name", "dob", "dod",
            "email", "phone_number", "birth_city",
        ]
        self.embedding_dim = embedding_dim
        self.cls_token_id = vocab_size - 2
        self.sep_token_id = vocab_size - 1
        self.max_len = max_len
        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def _byte_encode(self, subject):
        tokens = [self.cls_token_id]
        for feature in self.feature_list:
            value = str(getattr(subject, feature, "")).lower().strip()
            if value:
                value_bytes = list(value.encode('utf-8'))
                tokens.extend(b for b in value_bytes)
                tokens.append(self.sep_token_id)
        tokens = tokens[:self.max_len]
        padding = [0] * (self.max_len - len(tokens))
        tokens.extend(padding)
        return torch.tensor(tokens, dtype=torch.long)

    def transformer_similarity(self, subject1, subject2):
        model = self.module if isinstance(self, nn.DataParallel) else self
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            arr1 = model._byte_encode(subject1)
            arr2 = model._byte_encode(subject2)
            batch = torch.stack([arr1, arr2]).to(device)
            embeddings = model(batch)
            vec1 = embeddings[0].unsqueeze(0)
            vec2 = embeddings[1].unsqueeze(0)
            similarity = F.cosine_similarity(vec1, vec2)
        return similarity.item()

    def forward(self, arr):
        embedded = self.embeddings(arr) * math.sqrt(self.embedding_dim)
        pos_encoded = self.pos_encoder(embedded)
        transformer_output = self.transformer_encoder(pos_encoded)
        cls_embedding = transformer_output[:, 0, :]
        return cls_embedding

    def train_transformer(
        self, train_pairs, train_labels, val_pairs, val_labels,
        epochs=10, lr=1e-4, device="cuda", batch_size=512
    ):
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(self)
        else:
            model = self
        
        model.to(device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        train_dataset = PreEncodedDataset(train_pairs, train_labels, self)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
        
        val_dataset = PreEncodedDataset(val_pairs, val_labels, self)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)

        print("Starting training...")
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            train_batch_count = 0
            for arr1_batch, arr2_batch, label_batch in train_loader:
                arr1_batch = arr1_batch.to(device)
                arr2_batch = arr2_batch.to(device)
                target = (label_batch * 2 - 1).float().to(device)

                optimizer.zero_grad()

                combined_batch = torch.cat([arr1_batch, arr2_batch], dim=0)
                all_embeddings = model(combined_batch)
                vec1, vec2 = torch.chunk(all_embeddings, 2, dim=0)

                similarity = F.cosine_similarity(vec1, vec2, dim=1)
                
                positive_loss = (1 - similarity)[target == 1]
                negative_loss = F.relu(similarity[target == -1] - 8)
                
                loss_pos = positive_loss.mean() if positive_loss.numel() > 0 else torch.tensor(0.0, device=device)
                loss_neg = negative_loss.mean() if negative_loss.numel() > 0 else torch.tensor(0.0, device=device)
                
                loss = loss_pos + (10.0 * loss_neg)

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                train_batch_count += 1

            avg_train_loss = total_train_loss / train_batch_count if train_batch_count > 0 else 0
            
            model.eval()
            total_val_loss = 0.0
            val_batch_count = 0
            total_tp, total_fp, total_fn = 0, 0, 0
            with torch.no_grad():
                for arr1_batch, arr2_batch, label_batch in val_loader:
                    arr1_batch = arr1_batch.to(device)
                    arr2_batch = arr2_batch.to(device)
                    target = (label_batch * 2 - 1).float().to(device)
                    
                    combined_batch = torch.cat([arr1_batch, arr2_batch], dim=0)
                    all_embeddings = model(combined_batch)
                    vec1, vec2 = torch.chunk(all_embeddings, 2, dim=0)
                    
                    similarity = F.cosine_similarity(vec1, vec2, dim=1)
                    
                    positive_loss = (1 - similarity)[target == 1]
                    negative_loss = F.relu(similarity[target == -1] - 0.8)
                    
                    loss_pos = positive_loss.mean() if positive_loss.numel() > 0 else torch.tensor(0.0, device=device)
                    loss_neg = negative_loss.mean() if negative_loss.numel() > 0 else torch.tensor(0.0, device=device)

                    val_loss = loss_pos + (10.0 * loss_neg)
                    total_val_loss += val_loss.item()
                    val_batch_count += 1
                    
                    labels_on_device = label_batch.to(device)
                    preds = (similarity > 0.90).long()
                    total_tp += ((preds == 1) & (labels_on_device == 1)).sum().item()
                    total_fp += ((preds == 1) & (labels_on_device == 0)).sum().item()
                    total_fn += ((preds == 0) & (labels_on_device == 1)).sum().item()
            
            avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else 0
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Precision: {precision:.4f} | Val Recall: {recall:.4f}")

    def save(self, path):
        if isinstance(self, nn.DataParallel):
            torch.save(self.module.state_dict(), path)
        else:
            torch.save(self.state_dict(), path)    

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location or "cpu"))