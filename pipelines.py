#
#       indexing_pipelines.py
#
# Indexing pipelines for the multimodal Greek speech alignment project. 
# Contains functions to process and index the speech and text data, including feature extraction and storage.
#
# Author: Alexandros Tsingilis
# Date: 17 Feb 2026
#%%
import torch
import os
import json
import numpy as np
import librosa
import evaluate, preprocess, models
from tqdm import tqdm
from itertools import batched
import pdb



#
# No-ASR pipeline for segmenting audio, transcribing, and retrieving relevant segments by transcript embedding.
#
class AlignmentPipeline:
    """
    Pipeline for segmenting audio, extracting features, and aligning speech using a trained model.
    Handles loading audio, segmenting, model loading, and feature extraction for the multimodal Greek speech alignment project.
    """

    def __init__(self,
                 whisper_processor,
                 whisper_model,
                 aligner: models.AlignmentModel,
                 e5_model,
                 seg_length=10.0, # Seconds
                 overlap=5, # Seconds
                 sample_rate=16000,
                 kernel_size=8,
                 stride=4,
                 eps=1e-5,
                 device='cpu',
                 batch_size=8):
        """
        Initialize the IndexingPipeline.
        Args:
            file_path (str): Path to the audio file to process.
            seg_length (float): Length of each audio segment in seconds.
            overlap (float): Overlap between segments in seconds.
            sample_rate (int): Audio sample rate for loading.
            config_path (str): Path to the model config JSON file.
            ckpt_path (str): Path to the model checkpoint file.
            device (str): Device to use for model inference (e.g., 'cpu' or 'cuda').
        """
        self.seg_length = seg_length
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.device = device
        self.processor = whisper_processor
        self.whisper = whisper_model.to(device)
        self.aligner = aligner.eval().to(device)
        self.e5 = e5_model.to(device)
        # hyper parameters
        self.seg_length = seg_length
        self.overlap = overlap
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.device = device
        self.batch_size = batch_size
        # Storage for the aligned speech embeddings
        self.waveform = None
        self.speech_embs = None
        self.segments = None

    def _segment_audio(self, waveform):
        """
        Segment the audio waveform into overlapping segments.
        Args:
            wf (np.ndarray): Audio waveform array.
            sr (int): Sample rate of the audio.
            seg_length (float, optional): Segment length in seconds. Defaults to self.seg_length.
            overlap (float, optional): Overlap in seconds. Defaults to self.overlap.
        Returns:
            list: List of audio segments as numpy arrays.
        """
        print("Segmenting audio...", end='')
        seg_length = self.seg_length
        overlap = self.overlap
        sr = self.sample_rate
        # Convert segment length and overlap from seconds to samples
        seg_length_samples = int(seg_length * sr)
        overlap_samples = int(overlap * sr)
        # Calculate the step size for segmenting the audio
        step = seg_length_samples - overlap_samples
        segments = []
        # Segment the audio waveform into overlapping segments
        for start in range(0, len(waveform), step):
            end = start + seg_length_samples
            if end > len(waveform):
                end = len(waveform)
            segments.append(waveform[start:end])
            if end == len(waveform):
                break
        print(f"Done. Created {len(segments)} segments.")
        self.segments = segments
        return segments
    
    @torch.inference_mode()
    def _preprocess(self, segments, batch_size=8):
        """
        Preprocess audio segments for embedding.
        Args:
            segments (list of np.ndarray): List of audio segments to preprocess.
            query (str): Text query to embed.
            batch_size (int): Batch size for processing segments. Defaults to 8.
        Returns:
            tuple: (speech_hidden, attn_masks, transcript_embs) where each element is a torch.Tensor.
        """
        print("Preprocessing segments...", end='')
        speech_hidden = []  # List[Tensor], each of shape [batch, T, D]
        attn_masks = []     # List[Tensor], each of shape [batch, T]
        batches = batched(segments, batch_size)
        n_batches = (len(segments) + batch_size - 1) // batch_size
        # Process each batch of segments
        for batch in tqdm(batches, total=n_batches):
            # Preprocess the batch of audio segments and extract features using the Whisper model
            s, m = preprocess.embed_speech(self.processor,
                                        self.whisper,
                                        speech=batch,
                                        sr=self.sample_rate,
                                        device=self.device)
            # Downsample the features to match the aligner's expected input size
            s, m = preprocess.masked_mean_pool_time(s, m,
                                                    kernel_size=self.kernel_size,
                                                    stride=self.stride,
                                                    eps=self.eps)
            # Move the processed features and attention masks to CPU and store them in lists
            speech_hidden.append(s.cpu())
            attn_masks.append(m.cpu())
        # Concatenate the lists of processed features and attention masks into single tensors
        speech_hidden = torch.cat(speech_hidden, dim=0)  # Tensor, shape [num_segments, T, D]
        attn_masks = torch.cat(attn_masks, dim=0)        # Tensor, shape [num_segments, T]
        print("Done.")
        
        return speech_hidden, attn_masks, 

    @torch.inference_mode()
    def _align_embeddings(self, speech_hidden, attn_masks, batch_size=8):
        """
        Align the speech embeddings using the aligner model.
        Args:
            speech_hidden (torch.Tensor): Encoder hidden states of shape ``[B, T, D]``.
            attn_masks (torch.Tensor): Attention masks of shape ``[B, T]``.
            batch_size (int): Batch size for processing segments. Defaults to 8.
        Returns:
            torch.Tensor: Aligned speech embeddings of shape ``[B, D]``.
        """
        print("Aligning embeddings...", end='')
        aligned_embeddings = []
        n_batches = (speech_hidden.shape[0] + batch_size - 1) // batch_size
        batches = list(batched(range(speech_hidden.shape[0]), batch_size))
        # Process each batch of speech hidden states and attention masks through the aligner model
        for batch in tqdm(batches, total=n_batches):
            start, end = batch[0], batch[-1] + 1
            s = speech_hidden[start:end, :, :].to(self.device)  # Tensor, shape [batch_size, T, D]
            m = attn_masks[start:end, :].to(self.device)       # Tensor, shape [batch_size, T]
            #*db.set_trace()
            aligned_emb = self.aligner.encode(s, m)             # Tensor, shape [batch_size, D]
            aligned_embeddings.append(aligned_emb.cpu())
        # Concatenate the list of aligned embeddings into a single tensor
        aligned_embeddings = torch.cat(aligned_embeddings, dim=0)  # Tensor, shape [num_segments, D]
        print("Done.")
        return aligned_embeddings

    def load_file(self, file_path, batch_size=None):
        """
        Load an audio file and return the waveform and sample rate.
        Args:
            file_path (str): Path to the audio file to load.
            batch_size (int): Batch size for processing segments.
        Returns:
            tuple: (waveform, aligned_embedding) where waveform is a numpy array and aligned_embedding is a torch.Tensor.
        """
        if batch_size is None:
            batch_size = self.batch_size
        # Load and resample the audio file        
        wf, _ = librosa.load(file_path, sr=self.sample_rate)
        self.waveform = wf
        # split the audio into segments
        segments = self._segment_audio(wf)
        # preprocess the segments to extract features and attention masks
        speech_hidden, attn_masks, = self._preprocess(segments, batch_size=batch_size)
        # Produce final speech embeddings using the aligner model
        aligned_embedding = self._align_embeddings(speech_hidden, attn_masks, batch_size=batch_size)
        # Store the aligned embeddings
        self.speech_embs = aligned_embedding
    
    def _embed_query(self, query):
        """
        Embed a text query using the E5 model.
        Args:
            query (str): Text query to embed.
        Returns:
            torch.Tensor: Embedded query tensor of shape [1, D].
        """
        query_emb = preprocess.embed_transcripts(self.e5, 
                                                 {'sentence': [query]})
        return query_emb

    def _retrieve(self, speech_embs, query_emb, top_k=5):
        """
        Retrieve the top-k most relevant audio segments for a given query embedding.
        Args:
            speech_emb (torch.Tensor): Encoder hidden states of shape ``[B, T, D]`` (CPU, float32).
            attn_mask (torch.Tensor): Attention mask of shape ``[B, T]`` (CPU, int16).
            query_emb (torch.Tensor): Embedded query tensor.
            top_k (int): Number of top segments to retrieve.

        Returns:
            list: List of indices of the top-k most relevant segments.
        """
        # Compute cosine similarity between the query embedding and the aligned speech embeddings
        similarity = evaluate.compute_similarity_matrix(query_emb, 
                                                        speech_embs,
                                                        device=self.device,
                                                        batch_size=self.batch_size)  # Tensor, shape [num_queries, num_segments]

        # Retrieve the indices of the top-k most relevant segments based on similarity scores
        top_k_indices = torch.topk(similarity, k=top_k).indices.cpu().numpy().tolist()  # List[List[int]], shape [num_queries, top_k]
        return top_k_indices


    def search(self, query, top_k=5):
        """
        Search for the most relevant audio segments in a given file for a text query.
        Args:
            query (str): Text query to search for.
            top_k (int): Number of top segments to retrieve.

        Returns:
            list: List of indices of the top-k most relevant segments.
            list: List of time ranges (start_time, end_time) in seconds corresponding to the top-k segments.
        """
        if self.speech_embs is None:
            raise ValueError("No speech embeddings found. Please load an audio file first using load_file().")
        if self.segments is None:
            raise ValueError("No audio segments found. Please load an audio file first using load_file().")
        # Embed the query text using the E5 model
        query_emb = self._embed_query(query)  # Tensor, shape [1, D]
        top_k_indices = self._retrieve(self.speech_embs,
                                       query_emb, 
                                       top_k=top_k)[0] #! Chnage later for num(queries) > 1
        #*pdb.set_trace()
        audio_elements = [
            [idx, self.segments[idx]]
            for idx in top_k_indices
        ]
        return audio_elements


#
# ASR-based pipeline for segmenting audio, transcribing, and retrieving relevant segments by transcript embedding.
#
