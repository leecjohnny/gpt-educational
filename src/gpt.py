import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pydantic import BaseModel


class ModelConfig(BaseModel):
    vocab_size: int  # Size of vocabulary
    block_size: int  # Context size for the model
    embed_dim: int  # Embedding dimension
    hidden_dim: int  # Hidden dimension for transformer
    batch_size: int  # Batch size for training
    num_layers: int  # Number of transformer layers
    num_heads: int  # Number of attention heads
    head_size: int  # Size of each attention head
    dropout: float  # Dropout rate
    ffw_width_multiplier: int  # Multiplier for FFW layer width
    seed: int  # Random seed for reproducibility


class Head(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.q = nn.Linear(config.hidden_dim, config.head_size, bias=False)
        self.k = nn.Linear(config.hidden_dim, config.head_size, bias=False)
        self.v = nn.Linear(config.hidden_dim, config.head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, C = x.shape
        k = self.k(x)  ## (B, T, C)
        q = self.q(x)  ## (B, T, C)

        # compute attention scores
        scores = q @ k.transpose(-2, -1) * C**-0.5  ## (B, T, T)
        assert isinstance(self.tril, torch.Tensor)
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf"),  ## causal mask
        )
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)  # (B, T, T)

        # compute attention weights
        v = self.v(x)  ## (B, T, C)
        out = scores @ v  ## (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ffw = nn.Sequential(
            nn.Linear(
                config.hidden_dim, config.ffw_width_multiplier * config.hidden_dim
            ),
            nn.GELU(),
            nn.Linear(
                config.ffw_width_multiplier * config.hidden_dim, config.hidden_dim
            ),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffw(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.self_attention = MultiHeadAttention(config)
        self.ffw = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.seed)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(config.block_size, config.hidden_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)],
            nn.LayerNorm(config.hidden_dim),
        )
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.apply(self._init_weights)
        logging.info(f"Initializing model with {self.get_num_params()} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.shape
        ##project the input tokens to the hidden dimension
        embeds = self.embedding(x)
        positions = self.positional_embedding(torch.arange(T, device=x.device))
        x = embeds + positions  ## (B, T, C)
        x = self.blocks(x)  ## (B, T, C)
        logits = self.lm_head(x)  ## (B, T, V)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            batched_logits = rearrange(
                logits, "b t c -> (b t) c"
            )  ## c is the logits for each token in the vocabulary
            batched_targets = rearrange(targets, "b t -> (b t)")
            loss = F.cross_entropy(batched_logits, batched_targets)

        return logits, loss

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        num_samples: int = 1,
        random_seed: int | None = None,
    ) -> torch.Tensor:
        ## check model is in eval mode
        assert self.training is False, "Model must be in eval mode for generation"
        B, T = x.shape
        if num_samples > 1:
            x = repeat(x, "b t -> b s t", s=num_samples)  # (B, S, T)
            x = rearrange(x, "b s t -> (b s) t")  # (B*S, T)

        generator = torch.Generator(
            device=x.device,
        )

        if random_seed is not None:
            generator.manual_seed(random_seed)

        for _ in range(max_new_tokens):
            context = x[
                :,
                -self.config.block_size :,  ## shift the context window, truncate if necessary
            ]
            logits, _ = self(context)
            logits = logits[:, -1, :]  # only take the last timestamp's logits
            if temperature > 0.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(
                probs, num_samples=1, generator=generator
            )  # (B, 1)
            x = torch.cat((x, x_next), dim=1)  # (B, T+1)

        if num_samples > 1:
            x = rearrange(x, "(b s) t -> b s t", s=num_samples)  # (B, S, T)
        return x
