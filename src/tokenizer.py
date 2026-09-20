"""Character-level tokenizer: 26 letters + <eos> (27 tokens).

Implements encode/decode between words and letter-index sequences, and
batches words of different lengths (padding + ignore_index for the loss).
"""

import torch


class Tokenizer:
    def __init__(self):
        self.chars = list("abcdefghijklmnopqrstuvwxyz")

        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.stoi["<EOS>"] = len(self.chars)
        self.stoi["<PAD>"] = len(self.chars)+1

        self.itos = {i:ch for ch,i in self.stoi.items()}

    def encode(self, word):
        return [self.stoi[ch] for ch in word] + [self.stoi["<EOS>"]]

    def decode(self, tokens):
        word = ""

        for token in tokens:
            token = int(token)

            if token == self.stoi["<EOS>"]:
                break
            if token == self.stoi["<PAD>"]:
                continue

            word += self.itos[token]

        return word

    def batch_encode(self, words):
        """
            Make all sequences in the batch the same length
        """
        encoded = [self.encode(word) for word in words]

        max_len = max(len(tokens) for tokens in encoded)

        padded = []

        for tokens in encoded:
            padding = [self.stoi["<PAD>"]] * (max_len - len(tokens))
            padded.append(tokens + padding)

        return torch.tensor(padded)

if __name__ == "__main__":
    tokenizer = Tokenizer()

    word = "rlsentrancechallenge"
    tokens = tokenizer.encode(word)
    print(tokens)
    print(tokenizer.decode(tokens))

    words = ["helloworldbingo", "rlsentrancechallengesss", "visionlanguagemodels"]
    print(tokenizer.batch_encode(words))
