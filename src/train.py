"""Training loop.

Forward pass, cross-entropy loss with padding excluded via ignore_index,
backward pass, optimizer step, periodic validation, and checkpointing.

Main Ressource: https://huggingface.co/learn/llm-course/chapter3/4
"""

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm.auto import tqdm

from configs.config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from src.data import create_dataloader
from src.model.vlm import VLM
from src.tokenizer import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device", device)

train_loader = create_dataloader(
    "data/train.pt",
    batch_size = BATCH_SIZE,
    shuffle=True
)
val_loader = create_dataloader(
    "data/val.pt",
    batch_size=BATCH_SIZE,
    shuffle=True
)

tokenizer = Tokenizer()
PAD_ID = tokenizer.stoi["<PAD>"]


model = VLM().to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = NUM_EPOCHS * len(train_loader)
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(NUM_EPOCHS):

    train_loss = 0

    for images, tokens in train_loader:

        images = images.to(device)
        tokens = tokens.to(device)

        input_tokens = tokens[:, :-1]
        targets = tokens

        # Forward pass
        logits = model(images, input_tokens)       # (B, T, VOCAB_SIZE)

        loss = F.cross_entropy(
            input = logits.reshape(-1, logits.size(-1)),
            target = targets.reshape(-1),
            ignore_index = PAD_ID
        )

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for images, tokens in val_loader:

            images = images.to(device)
            tokens = tokens.to(device)

            input_tokens = tokens[:, :-1]
            targets = tokens

            logits = model(images, input_tokens)

            loss = F.cross_entropy(
                input = logits.reshape(-1, logits.size(-1)),
                target = targets.reshape(-1),
                ignore_index = PAD_ID
            )
            val_loss += loss

    val_loss = val_loss / len(val_loader)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        f"- train loss: {train_loss:.4f}"
        f"- val_loss: {val_loss: .4f}"
    )

# Save the model
torch.save(
    model.state_dict(),
    "model.pt"
)

print("Save the model!")