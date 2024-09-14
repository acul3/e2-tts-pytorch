import torch
from e2_tts_pytorch import E2TTS, DurationPredictor

from torch.optim import Adam
from datasets import load_dataset,load_from_disk

from e2_tts_pytorch.trainer import (
    HFDataset,
    E2Trainer
)

e2tts = E2TTS(
    tokenizer = 'custom_id',
    cond_drop_prob = 0.2,
    transformer = dict(
        dim = 384,
        depth = 12,
        heads = 6,
        max_seq_len = 2400,
        skip_connect_type = 'concat'
    ),
    mel_spec_kwargs = dict(
        filter_length = 1024,
        hop_length = 256,
        win_length = 1024,
        n_mel_channels = 100,
        sampling_rate = 24000,
    ),
    frac_lengths_mask = (0.7, 0.9)
)


train_dataset = HFDataset(load_from_disk("/scratch/bahasa/lintas-data"))

optimizer = Adam(e2tts.parameters(), lr=3e-4)

trainer = E2Trainer(
    e2tts,
    optimizer,
    num_warmup_steps=5000,
    grad_accumulation_steps = 1,
    name_checkpoint_path = 'e2tts',
    log_file = 'e2tts.txt'
)

epochs = 20
batch_size = 4

trainer.train(train_dataset, epochs, batch_size,num_workers=8, save_step=20000)
