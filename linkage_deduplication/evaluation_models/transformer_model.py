"""
Author: Michael Kurilko
Date: 6/3/2025 (Torch version 6/5/2025)
Description: This module contains the TransformerModel class, which provides
transformer-based similarity scorer using TORCH!
"""

import os
import sys

import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# debug, force multiprocessing using spawn
if __name__ != '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

class LazyDataset(Dataset):
    """
    lazy dataset on demand encoding
    """

    def __init__(self, subject_pairs, labels, feature_list, max_len=32):
        self.subject_pairs = subject_pairs
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.feature_list = feature_list
        self.max_len = max_len

        # pre-cache strings
        self.processed_pairs = []
        for i, (subj1, subj2) in enumerate(subject_pairs):
            if i % 100000 == 0:
                print(f"cached {i}/{len(subject_pairs)} pairs...")

            subj1_data = {f: getattr(subj1, f, "") for f in self.feature_list}
            subj2_data = {f: getattr(subj2, f, "") for f in self.feature_list}

            self.processed_pairs.append((subj1_data, subj2_data))

        print(f"pre-processing complete: {len(self.processed_pairs)} pairs ready")

    def __len__(self):
        return len(self.processed_pairs)

    def _get_subject_string_from_dict(self, subject_dict):
        return " ".join([str(subject_dict.get(f, "")) for f in self.feature_list])

    def _byte_encode(self, s):
        """encode string to byte tensor"""
        arr = list(s.encode("utf-8")[: self.max_len])
        arr = arr + [0] * (self.max_len - len(arr))
        return torch.tensor(arr, dtype=torch.long)

    def __getitem__(self, idx):
        subj1_data, subj2_data = self.processed_pairs[idx]

        s1 = self._get_subject_string_from_dict(subj1_data)
        s2 = self._get_subject_string_from_dict(subj2_data)

        arr1 = self._byte_encode(s1)
        arr2 = self._byte_encode(s2)

        return arr1, arr2, self.labels[idx]


# old PreEncodedDataset (commented for reference)
# class PreEncodedDataset(Dataset):
#     """
#     pre-encoded dataset - MEMORY INTENSIVE, avoid for large datasets
#     """
#
#     def __init__(self, subject_pairs, labels, model):
#         self.model = model
#         self.inputs = []
#         self.labels = torch.tensor(labels, dtype=torch.float32)
#         for subj1, subj2 in subject_pairs:
#             arr1 = model._byte_encode(
#                 " ".join([str(getattr(subj1, f, "")) for f in model.feature_list])
#             )
#             arr2 = model._byte_encode(
#                 " ".join([str(getattr(subj2, f, "")) for f in model.feature_list])
#             )
#             self.inputs.append((arr1, arr2))
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, idx):
#         return self.inputs[idx][0], self.inputs[idx][1], self.labels[idx]


class TransformerModel(nn.Module):
    """
    transformer model
    """

    def __init__(self, embedding_dim=16, vocab_size=257):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.attn_wq = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attn_wk = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attn_wv = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.ffn_w1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.ffn_w2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gb_weights = nn.Parameter(torch.ones(7))

        # feature list for consistent encoding
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

    def _get_subject_string(self, subject):
        """convert subject to string representation"""
        return " ".join([str(getattr(subject, f, "")) for f in self.feature_list])

    def _transformer_encode(self, arr):
        device = arr.device
        x = self._embed(arr).to(device)
        x = self._self_attention(x).to(device)
        x = self._ffn(x).to(device)
        return x.mean(dim=0)

    def _transformer_encode_batch(self, arr_batch):
        """process a batch of encodings at once"""
        # arr_batch: (B, L)
        x = self.embeddings(arr_batch)
        Q = self.attn_wq(x)
        K = self.attn_wk(x)
        V = self.attn_wv(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embedding_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        x = torch.matmul(attn_weights, V)
        x = self.ffn_w2(F.relu(self.ffn_w1(x)))
        return x.mean(dim=1)

    def _compute_similarity(self, vec1, vec2):
        """compute cosine similarity between two vectors"""
        dot = (vec1 * vec2).sum()
        norm1 = vec1.norm()
        norm2 = vec2.norm()
        similarity = dot / (norm1 * norm2 + 1e-8)
        return (similarity + 1) / 2  # scale to [0,1]

    def forward(self, arr1, arr2):
        """forward pass for batch processing"""
        vec1 = self._transformer_encode_batch(arr1)
        vec2 = self._transformer_encode_batch(arr2)
        return vec1, vec2

    def transformer_similarity(self, subject1, subject2, device=None):
        """compute similarity between two subjects"""
        s1 = self._get_subject_string(subject1)
        s2 = self._get_subject_string(subject2)

        if device is None:
            device = self.embeddings.weight.device

        arr1 = self._byte_encode(s1).to(device)
        arr2 = self._byte_encode(s2).to(device)

        vec1 = self._transformer_encode(arr1)
        vec2 = self._transformer_encode(arr2)

        similarity = self._compute_similarity(vec1, vec2)
        return float(similarity.item())

    def transformer_similarity_tensor(self, subject1, subject2):
        """compute similarity between two subjects, returning tensor"""
        s1 = self._get_subject_string(subject1)
        s2 = self._get_subject_string(subject2)

        device = self.embeddings.weight.device
        arr1 = self._byte_encode(s1).to(device)
        arr2 = self._byte_encode(s2).to(device)

        vec1 = self._transformer_encode(arr1)
        vec2 = self._transformer_encode(arr2)

        return self._compute_similarity(vec1, vec2)

    def train_transformer(
        self, subject_pairs, labels, epochs=10, lr=1e-3, device="cuda", batch_size=32
    ):
        # multi-gpu setup
        use_multi_gpu = torch.cuda.device_count() > 1
        dp_model = nn.DataParallel(self) if use_multi_gpu else self
        dp_model.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        eps = 1e-7

        # create dataset/dataloader with lazy loading
        print(f"creating lazy dataset for {len(subject_pairs)} pairs...")
        dataset = LazyDataset(subject_pairs, labels, self.feature_list)

        # debug info
        print(f"system info:")
        print(f" - cpu count: {mp.cpu_count()}")
        print(f" - platform: {sys.platform}")
        print(f" - mp start method: {mp.get_start_method()}")

        # calculate number of workers and batch size
        total_pairs = len(subject_pairs)

        if total_pairs > 1000000:
            batch_size = max(batch_size, 64)
            num_workers = min(mp.cpu_count() - 2, 24)
            prefetch_factor = 8
        else:
            num_workers = min(mp.cpu_count(), 20)
            prefetch_factor = 2

        print(f"settings:")
        print(f" - batch size: {batch_size}")
        print(f" - workers: {num_workers}")
        print(f" - prefetch: {prefetch_factor}")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            drop_last=True
        )

        # original training loop (commented for reference)
        # for epoch in range(epochs):
        #     total_loss = 0.0
        #     for (subj1, subj2), label in zip(subject_pairs, labels):
        #         pred = self.transformer_similarity_tensor(subj1, subj2)
        #         pred = torch.clamp(pred, eps, 1 - eps)
        #         target = torch.full(pred.shape, float(label), device=pred.device)
        #         loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).mean()
        #         opt.zero_grad()
        #         loss.backward()
        #         opt.step()
        #         total_loss += float(loss.item())
        #         if torch.isnan(loss):
        #             print(f"NaN detected in loss! pred={pred.cpu().detach().numpy()}, target={target.cpu().detach().numpy()}")
        #             break
        #     print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(subject_pairs):.4f}")

        # training loop with DataLoader
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_arr1, batch_arr2, batch_labels in loader:
                batch_arr1 = batch_arr1.to(device)
                batch_arr2 = batch_arr2.to(device)
                batch_labels = batch_labels.to(device)

                # get batch embeddings
                vec1_batch, vec2_batch = dp_model(batch_arr1, batch_arr2)

                # compute batch similarities
                batch_preds = []
                for i in range(len(vec1_batch)):
                    similarity = self._compute_similarity(vec1_batch[i], vec2_batch[i])
                    batch_preds.append(similarity)

                batch_preds = torch.stack(batch_preds)
                batch_preds = torch.clamp(batch_preds, eps, 1 - eps)

                # calculate loss for the batch
                loss = -(
                    batch_labels * torch.log(batch_preds)
                    + (1 - batch_labels) * torch.log(1 - batch_preds)
                ).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += float(loss.item())
                num_batches += 1

                if torch.isnan(loss):
                    print(
                        f"NaN detected in loss! pred={batch_preds.cpu().detach().numpy()}, target={batch_labels.cpu().detach().numpy()}"
                    )
                    break

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")

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
