"""
Author: Michael Kurilko
Date: 6/3/2025 (Torch version 6/5/2025)
Description: This module contains the TransformerModel class, which provides
transformer-based similarity scorer using TORCH!
"""

import os
import sys
import time
import logging

import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# debug, force multiprocessing using spawn for better cross-platform compatibility
# Only set if not already set to avoid conflicts
try:
    if mp.get_start_method(allow_none=True) is None:
        if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
            mp.set_start_method("spawn", force=True)
        elif sys.platform.startswith("win"):
            mp.set_start_method("spawn", force=True)
except RuntimeError as e:
    print(f"Warning: Could not set multiprocessing start method: {e}")

# Configure logging to work with multiprocessing
logging.basicConfig(level=logging.INFO, format="%(processName)s - %(message)s")
logger = logging.getLogger(__name__)


class PreEncodedDataset(Dataset):
    """
    Pre-encoded dataset for efficient training - optimized for Tesla K80s
    """

    def __init__(self, subject_pairs, labels, model, debug_mode=False):
        self.model = model
        self.inputs = []
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.debug_mode = debug_mode
        
        print(f"Pre-encoding {len(subject_pairs)} subject pairs for efficient training...")
        
        # Pre-encode all data for maximum efficiency
        for idx, (subj1, subj2) in enumerate(subject_pairs):
            if debug_mode and idx % 10000 == 0:
                print(f"  Encoding pair {idx}/{len(subject_pairs)}")
                
            arr1 = self.model._byte_encode(
                " ".join([str(getattr(subj1, f, "")) for f in self.model.feature_list])
            )
            arr2 = self.model._byte_encode(
                " ".join([str(getattr(subj2, f, "")) for f in self.model.feature_list])
            )
            self.inputs.append((arr1, arr2))
        
        print("Pre-encoding complete!")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx][0], self.inputs[idx][1], self.labels[idx]


class LazyDataset(Dataset):
    """
    lazy dataset on demand encoding (fallback for memory-constrained scenarios)
    """

    def __init__(
        self, subject_pairs, labels, feature_list, max_len=32, debug_mode=False
    ):
        # Store raw subject pairs and labels
        self.subject_pairs = subject_pairs
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.feature_list = feature_list
        self.max_len = max_len
        self.debug_mode = debug_mode

        # Use logging instead of print for multiprocessing compatibility
        logger.info(
            f"Dataset initialized with {len(subject_pairs)} raw pairs. Processing will happen on demand."
        )
   
    def __len__(self):
        return len(self.subject_pairs)

    def _get_subject_string_from_dict(self, subject_obj):
        # This function now takes the raw subject object and extracts features
        return " ".join([str(getattr(subject_obj, f, "")) for f in self.feature_list])

    def _byte_encode(self, s):
        """encode string to byte tensor"""
        arr = list(s.encode("utf-8")[: self.max_len])
        arr = arr + [0] * (self.max_len - len(arr))
        return torch.tensor(arr, dtype=torch.long)

    def __getitem__(self, idx):
        # Only add timing for debug mode and use logging for multiprocessing compatibility
        if self.debug_mode:
            start_getitem_time = time.time()

        subj1, subj2 = self.subject_pairs[idx]
        label = self.labels[idx]

        if self.debug_mode:
            start_string_time = time.time()
        
        s1 = self._get_subject_string_from_dict(subj1)
        s2 = self._get_subject_string_from_dict(subj2)
        
        if self.debug_mode:
            end_string_time = time.time()
            # Use logging instead of print for multiprocessing
            logger.info(
                f"Stringification for item {idx}: {end_string_time - start_string_time:.6f} seconds"
            )

        if self.debug_mode:
            start_encode_time = time.time()
        
        arr1 = self._byte_encode(s1)
        arr2 = self._byte_encode(s2)
        
        if self.debug_mode:
            end_encode_time = time.time()
            logger.info(
                f"Byte encoding for item {idx}: {end_encode_time - start_encode_time:.6f} seconds"
            )

        if self.debug_mode:
            end_getitem_time = time.time()
            logger.info(
                f"Total __getitem__ time for item {idx}: {end_getitem_time - start_getitem_time:.6f} seconds"
            )

        return arr1, arr2, label


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

    def _monitor_gpu_utilization(self, epoch, batch_idx, num_batches):
        """Monitor and report GPU utilization"""
        if torch.cuda.is_available():
            print(f"\nGPU Utilization Report (Epoch {epoch+1}, Batch {batch_idx}/{num_batches}):")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                utilization = allocated / cached * 100 if cached > 0 else 0
                print(f"  GPU {i}: {allocated:.2f}GB / {cached:.2f}GB ({utilization:.1f}% utilized)")

    def train_transformer(
        self,
        subject_pairs,
        labels,
        epochs=10,
        lr=1e-3,
        device="cuda",
        batch_size=32,
        debug_mode=False,
        use_preencoded=True,
    ):
        # Enhanced multi-GPU setup for Tesla K80s
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"CUDA available with {num_gpus} GPUs")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Optimize batch size for Tesla K80s (12GB each)
            if num_gpus >= 4:  # Multi-GPU setup
                batch_size = max(batch_size, 128)  # Larger batches for multi-GPU
                if use_preencoded:
                    batch_size = max(batch_size, 50000)  # Very large batches for pre-encoded data
            
            use_multi_gpu = num_gpus > 1
            if use_multi_gpu:
                print(f"Using DataParallel across {num_gpus} GPUs")
                dp_model = nn.DataParallel(self)
            else:
                dp_model = self
            dp_model.to(device)
        else:
            print("CUDA not available, falling back to CPU")
            device = "cpu"
            dp_model = self.to(device)
            use_multi_gpu = False
            
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        eps = 1e-7

        # Choose dataset type based on memory availability and preference
        if use_preencoded and torch.cuda.is_available():
            print(f"Using pre-encoded dataset for maximum efficiency")
            dataset = PreEncodedDataset(subject_pairs, labels, self, debug_mode=debug_mode)
            # Use fewer workers for pre-encoded data since it's already in memory
            num_workers = min(4, mp.cpu_count())
            prefetch_factor = 2
        else:
            print(f"Using lazy dataset for memory efficiency")
            dataset = LazyDataset(
                subject_pairs, labels, self.feature_list, debug_mode=debug_mode
            )
            # Use more workers for lazy loading
            num_workers = min(mp.cpu_count(), 8)
            prefetch_factor = 4

        print(f"Training settings:")
        print(f" - batch size: {batch_size}")
        print(f" - workers: {num_workers}")
        print(f" - prefetch: {prefetch_factor}")
        print(f" - pre-encoded: {use_preencoded}")

        # Configure DataLoader with optimized settings
        try:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if device != "cpu" else False,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
                drop_last=True,
                timeout=30 if num_workers > 0 else 0,
            )
        except Exception as e:
            print(f"Error creating DataLoader: {e}")
            print("Falling back to single-threaded loading...")
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
            )

        # Training loop optimized for multi-GPU efficiency
        for epoch in range(epochs):
            dp_model.train()
            total_loss = 0.0
            num_batches = 0
            
            # CUDA memory management for Tesla K80s
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear cache before each epoch
                if epoch == 0:  # Show memory info on first epoch
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        cached = torch.cuda.memory_reserved(i) / 1024**3
                        print(
                            f"  GPU {i} memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
                        )
            
            try:
                for batch_idx, (arr1_batch, arr2_batch, label_batch) in enumerate(loader):
                    arr1_batch = arr1_batch.to(device)
                    arr2_batch = arr2_batch.to(device)
                    label_batch = label_batch.unsqueeze(1).to(device)

                    # Forward pass through DataParallel model
                    vec1, vec2 = dp_model(arr1_batch, arr2_batch)

                    # Compute similarity scores efficiently
                    dot = (vec1 * vec2).sum(dim=1, keepdim=True)
                    norm1 = vec1.norm(dim=1, keepdim=True)
                    norm2 = vec2.norm(dim=1, keepdim=True)
                    similarity = dot / (norm1 * norm2 + 1e-8)
                    similarity = (similarity + 1) / 2
                    similarity = torch.clamp(similarity, eps, 1 - eps)

                    # Compute loss
                    loss = -(
                        label_batch * torch.log(similarity) 
                        + (1 - label_batch) * torch.log(1 - similarity)
                    ).mean()

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    total_loss += loss.item() * arr1_batch.size(0)
                    num_batches += 1

                    if torch.isnan(loss):
                        print(
                            f"NaN detected in loss! similarity={similarity.cpu().detach().numpy()}, target={label_batch.cpu().detach().numpy()}"
                        )
                        break
                    
                    # Monitor GPU memory usage for large batches
                    if (
                        torch.cuda.is_available()
                        and batch_idx % 100 == 0
                        and debug_mode
                    ):
                        for i in range(torch.cuda.device_count()):
                            allocated = torch.cuda.memory_allocated(i) / 1024**3
                            print(
                                f"    Batch {batch_idx}, GPU {i}: {allocated:.2f}GB allocated"
                            )
                        
                    self._monitor_gpu_utilization(epoch, batch_idx, num_batches)

            except Exception as e:
                print(f"Error during training epoch {epoch+1}: {e}")
                if num_workers > 0:
                    print("Retrying with single-threaded DataLoader...")
                    # Recreate DataLoader without multiprocessing
                    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        drop_last=True,
                    )
                    continue
                else:
                    raise

            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

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
