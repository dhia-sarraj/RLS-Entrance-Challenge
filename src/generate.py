"""Greedy decoding.

Generates one letter at a time from the visual tokens until <eos> or
45 letters, for inference and for evaluation.
"""

import torch

from configs.config import BATCH_SIZE, MAX_TEXT_LEN
from src.data import create_dataloader
from src.model.vlm import VLM
from src.tokenizer import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def greedy_decode(model, images, tokenizer):
    """
        Args:
            model: trained VLM
            images: (B, 3, 64, 64)

        Returns:
            decoded_words: list[str]
    """
    model.eval()

    images = images.to(device)
    features = model.encoder(images)
    visual_tokens = model.adapter(features)     # (B, S, D_MODEL)

    # The generated text_tokens
    generated = torch.empty(BATCH_SIZE, 0, dtype=torch.long, device=device)

    for _ in range(MAX_TEXT_LEN):
        logits = model.decoder(visual_tokens, generated)

        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(next_token_logits, dim=-1)    # (BATCH_SIZE,)     # TODO: Try it with multinomial
        
        generated = torch.cat(
            [generated, next_token.unsqueeze(1)],
            dim=1,
        )

    decoded_words = [
        tokenizer.decode(tokens) for tokens in generated
    ]

    return decoded_words

if __name__ == "__main__":
    tokenizer = Tokenizer()

    test_loader = create_dataloader(
        "data/test.pt",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = VLM().to(device)

    checkpoint = torch.load(
        "src/model.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.eval()

    images, tokens = next(iter(test_loader))

    predictions = greedy_decode(
        model,
        images,
        tokenizer,
    )

    nb_correct_per_batch = 0

    for i, prediction in enumerate(predictions):
        target = tokenizer.decode(tokens[i])

        if target == prediction:
            nb_correct_per_batch +=1

        print(
            f"target: {target:60s}"
            f"prediction: {prediction}"
        )

    print(f"avg: {nb_correct_per_batch/32}")