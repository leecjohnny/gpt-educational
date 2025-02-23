from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    embed_dim: int
    hidden_dim: int
    batch_size: int
    num_layers: int
    num_heads: int
    head_size: int
    dropout: float
    ffw_width_multiplier: int
    seed: int
    _generator: torch.Generator | None = None

    @property
    def generator(self) -> torch.Generator:
        if self._generator is None:
            self._generator = torch.Generator()
        # Always reset the generator state to ensure consistent random numbers
        self._generator.manual_seed(self.seed)
        return self._generator


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
        q = self.q(x)  ## (B, T, C)
        k = self.k(x)  ## (B, T, C)

        # compute attention scores
        scores = q @ rearrange(k, "b t h -> b h t") * C**-0.5  ## (B, T, T)
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
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.self_attention = MultiHeadAttention(config)
        self.ffw = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(config.block_size, config.hidden_dim)
        self.register_buffer("positions", torch.arange(config.block_size))
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)],
            nn.LayerNorm(config.hidden_dim),
        )
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.apply(self._init_weights)
        torch.use_deterministic_algorithms(True)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02, generator=self.config.generator
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02, generator=self.config.generator
            )

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.shape
        ##project the input tokens to the hidden dimension
        embeds = self.embedding(x)
        assert isinstance(self.positions, torch.Tensor)
        positions = self.positional_embedding(self.positions)
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
    ) -> torch.Tensor:
        ## check model is in eval mode
        assert self.training is False, "Model must be in eval mode for generation"
        B, T = x.shape
        if num_samples > 1:
            x = repeat(x, "b t -> b s t", s=num_samples)  # (B, S, T)
            x = rearrange(x, "b s t -> (b s) t")  # (B*S, T)

        for _ in range(max_new_tokens):
            context = x[:, -self.config.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :]  # only take the last timestamp's logits
            if temperature > 0.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(
                probs, num_samples=1, generator=self.config.generator
            )  # (B, 1)
            x = torch.cat((x, x_next), dim=1)  # (B, T+1)

        if num_samples > 1:
            x = rearrange(x, "(b s) t -> b s t", s=num_samples)  # (B, S, T)
        return x
