"""
Author: Michael Kurilko
Date: 6/3/2025 (Torch version 6/5/2025)
Description: This module contains the TransformerModel class, which provides
transformer-based similarity scorer using TORCH!
"""

import collections
import collections.abc
import math

collections.Container = collections.abc.Container

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import islice
from torch.utils.data import Dataset, DataLoader, IterableDataset


class PreEncodedDataset(IterableDataset):
    def __init__(self, subject_pairs, labels, model, max_samples=None):
        self.model = model
        self.subject_pairs = subject_pairs  # store the raw subject pairs
        self.labels = labels
        self.max_samples = max_samples

    def __iter__(self):
        # create fresh iterators for each epoch
        pairs_iter = iter(self.subject_pairs)
        labels_iter = iter(self.labels)
        
        generator = (
            (
                self.model._byte_encode(" ".join(str(getattr(subj1, f, "")) for f in self.model.feature_list)),
                self.model._byte_encode(" ".join(str(getattr(subj2, f, "")) for f in self.model.feature_list)),
                label
            )
            for idx, ((subj1, subj2), label) in enumerate(zip(pairs_iter, labels_iter))
            if self.max_samples is None or idx < self.max_samples
        )

        return islice(generator, self.max_samples) if self.max_samples else generator

class TransformerModel(nn.Module):
    def __init__(self, embedding_dim=16, vocab_size=257):
        super().__init__()
        self.feature_list = [
            "first_name",
            "middle_name",
            "last_name",
            "dob",
            "dod",
            "email",
            "phone_number",
            "birth_city",
        ]
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.attn_wq = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attn_wk = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attn_wv = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.ffn_w1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.ffn_w2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gb_weights = nn.Parameter(torch.ones(7))

    def _byte_encode(self, s, max_len=32):
        arr = list(s.encode("utf-8")[:max_len])
        arr = arr + [0] * (max_len - len(arr))
        return torch.tensor(arr, dtype=torch.long)

    def _embed(self, arr):
        return self.embeddings(arr)

    def _self_attention(self, x):
        Q = self.attn_wq(x)
        K = self.attn_wk(x)
        V = self.attn_wv(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embedding_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)

    def _ffn(self, x):
        return self.ffn_w2(F.relu(self.ffn_w1(x)))

    def _transformer_encode(self, arr):
        device = arr.device
        x = self._embed(arr).to(device)
        x = self._self_attention(x).to(device)
        x = self._ffn(x).to(device)
        return x.mean(dim=0)

    def transformer_similarity(self, subject1, subject2):
        s1 = " ".join([str(getattr(subject1, f, "")) for f in self.feature_list])
        s2 = " ".join([str(getattr(subject2, f, "")) for f in self.feature_list])
        if device is None:
            device = self.embeddings.weight.device
        arr1 = self._byte_encode(s1).to(device)
        arr2 = self._byte_encode(s2).to(device)
        vec1 = self._transformer_encode(arr1)
        vec2 = self._transformer_encode(arr2)
        dot = (vec1 * vec2).sum()
        norm1 = vec1.norm()
        norm2 = vec2.norm()
        similarity = dot / (norm1 * norm2 + 1e-8)
        return float(((similarity + 1) / 2).item())

    def _transformer_encode_batch(self, arr_batch):
        # arr_batch: (B, L)
        x = self.embeddings(arr_batch)  # (B, L, D)
        Q = self.attn_wq(x)
        K = self.attn_wk(x)
        V = self.attn_wv(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embedding_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        x = torch.matmul(attn_weights, V)  # (B, L, D)
        x = self.ffn_w2(F.relu(self.ffn_w1(x)))  # (B, L, D)
        return x.mean(dim=1)  # (B, D)

    def forward(self, arr1, arr2):
        vec1 = self._transformer_encode_batch(arr1)
        vec2 = self._transformer_encode_batch(arr2)
        return vec1, vec2

    def transformer_similarity_tensor(self, subject1, subject2):
        s1 = " ".join([str(getattr(subject1, f, "")) for f in self.features_list])
        s2 = " ".join([str(getattr(subject2, f, "")) for f in self.feature_list])
        arr1 = self._byte_encode(s1).to(self.embeddings.weight.device)
        arr2 = self._byte_encode(s2).to(self.embeddings.weight.device)
        vec1 = self._transformer_encode(arr1)
        vec2 = self._transformer_encode(arr2)
        dot = (vec1 * vec2).sum()
        norm1 = vec1.norm()
        norm2 = vec2.norm()
        similarity = dot / (norm1 * norm2 + 1e-8)
        return (similarity + 1) / 2  # Tensor in [0,1]

    def train_transformer(
        self, subject_pairs, labels, epochs=10, lr=1e-3, device="cuda", batch_size=10000, max_samples=10000
    ):
        use_multi_gpu = torch.cuda.device_count() > 1
        dp_model = nn.DataParallel(self) if use_multi_gpu else self
        dp_model.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        eps = 1e-7

        dataset = PreEncodedDataset(subject_pairs, labels, self, max_samples=max_samples)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            # shuffle=True,
            num_workers=16,
            pin_memory=True,
            prefetch_factor=10,
        )

        print("Training with batch size:", batch_size)

        for epoch in range(epochs):
            dp_model.train()
            total_loss = 0.0
            batch_count = 0

            for i, (arr1_batch, arr2_batch, label_batch) in enumerate(loader):
                arr1_batch = arr1_batch.to(device)
                arr2_batch = arr2_batch.to(device)
                label_batch = label_batch.unsqueeze(1).to(device)

                vec1, vec2 = dp_model(arr1_batch, arr2_batch)

                dot = (vec1 * vec2).sum(dim=1, keepdim=True)
                norm1 = vec1.norm(dim=1, keepdim=True)
                norm2 = vec2.norm(dim=1, keepdim=True)
                similarity = dot / (norm1 * norm2 + 1e-8)
                similarity = (similarity + 1) / 2
                similarity = torch.clamp(similarity, eps, 1 - eps)

                loss = -(
                    label_batch * torch.log(similarity)
                    + (1 - label_batch) * torch.log(1 - similarity)
                ).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item() * arr1_batch.size(0)
                batch_count += 1
                
                if i % 100 == 0:  # Print progress periodically
                    expected_batches = math.ceil(max_samples / batch_size)
                    print(f"  Batch {i}/{expected_batches}, Current Loss: {loss.item():.4f}")
                
                if batch_count >= math.ceil(max_samples / batch_size):
                    print("🔴 Hit max_batches limit, breaking early")
                    break
                
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location or "cpu"))


# Example usage:
# model = TransformerModel()
# gb_score = model.gradient_boosted_score(subject1, subject2)
# transformer_score = model.transformer_similarity(subject1, subject2)

"""
# Train your model
model = TransformerModel()
model.train_transformer(subject_pairs, labels, epochs=20, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu")

# Save the trained weights
model.save("transformer_weights.npz")


# Later, or in a new script/session
model = TransformerModel()
model.load("transformer_weights.npz")

# Now you can use the trained model for inference
score = model.transformer_similarity(subject1, subject2)
print("Similarity score:", score)
"""
