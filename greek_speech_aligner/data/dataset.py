#
#   dataset.py
#
# PyTorch / Lightning data utilities for the preprocessed HParl dataset
# (output of ``greek_speech_aligner.data.preprocessing``).
#
# Author: Alexandros Tsingilis
#
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

SPEECH_COL = "pooled_speech_embeddings"
MASK_COL = "pooled_attn_masks"
TEXT_COL = "transcript_embeddings"


class SpeechTextDataset(Dataset):
    """
    Wrapper around a HuggingFace split with the columns:
      - ``pooled_speech_embeddings`` : ``[T, D_speech]``
      - ``pooled_attn_masks``        : ``[T]``
      - ``transcript_embeddings``    : ``[D_text]``
    """

    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "speech": torch.as_tensor(item[SPEECH_COL], dtype=torch.float32),
            "mask": torch.as_tensor(item[MASK_COL], dtype=torch.long),
            "text": torch.as_tensor(item[TEXT_COL], dtype=torch.float32),
        }


def collate_fn(batch):
    """Stack a batch, zero-padding speech/mask along time if lengths differ."""
    speech_list = [b["speech"] for b in batch]
    mask_list = [b["mask"] for b in batch]

    max_T = max(s.shape[0] for s in speech_list)
    D = speech_list[0].shape[1]
    padded_speech = torch.zeros(len(batch), max_T, D, dtype=torch.float32)
    padded_mask = torch.zeros(len(batch), max_T, dtype=torch.long)
    for i, (s, m) in enumerate(zip(speech_list, mask_list)):
        T = s.shape[0]
        padded_speech[i, :T] = s
        padded_mask[i, :T] = m

    return {
        "speech": padded_speech,
        "mask": padded_mask,
        "text": torch.stack([b["text"] for b in batch], dim=0),
    }


def make_dataloader(hf_split, batch_size=128, shuffle=False, num_workers=4, drop_last=False):
    return DataLoader(
        SpeechTextDataset(hf_split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


class SpeechTextDataModule(L.LightningDataModule):
    """DataModule for precomputed speech/text embeddings."""

    def __init__(self, train_ds, val_ds, batch_size=128, num_workers=4, drop_last=True):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

    def train_dataloader(self):
        # Dropping the last, smaller batch keeps the number of in-batch
        # negatives constant (B - 1) for every training step.
        return make_dataloader(self.train_ds, self.batch_size, shuffle=True,
                               num_workers=self.num_workers, drop_last=self.drop_last)

    def val_dataloader(self):
        return make_dataloader(self.val_ds, self.batch_size, shuffle=False,
                               num_workers=self.num_workers)
