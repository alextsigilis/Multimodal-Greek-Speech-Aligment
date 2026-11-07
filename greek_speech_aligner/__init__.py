"""
greek_speech_aligner
====================

Parameter-efficient cross-modal alignment for ASR-free Greek spoken-content
retrieval: a small trainable aligner maps frozen Whisper-tiny encoder states
into the embedding space of a frozen multilingual-e5-small text encoder.

Subpackages / modules
---------------------
- ``models``      aligners (MLP, CNN, LSTM, E5-MLP), pooling, ``AlignmentModel``
- ``losses``      bidirectional contrastive objective
- ``data``        preprocessing and PyTorch / Lightning data utilities
- ``config``      aligner config files
- ``checkpoint``  loading trained aligners
- ``training``    Lightning training loop
- ``evaluation``  Recall@K / MRR retrieval metrics
- ``pipelines``   ASR-free search over long recordings
- ``cli``         command-line entry points
"""

__version__ = "0.2.0"
