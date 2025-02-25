import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import fire
import torch
import yaml
from einops import rearrange
from pydantic import BaseModel

import bpe
import gpt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)


T = TypeVar("T")


class TokenizerConfig(BaseModel):
    """Configuration for tokenizer."""

    type: str  # Options: "bpe" or "tiktoken"
    max_vocab_size: int  # For BPE only
    model_name: str  # For tiktoken only


class TrainingConfig(BaseModel):
    """Configuration for training a GPT model."""

    # Model configuration
    gpt_model_config: gpt.ModelConfig

    # Training hyperparameters
    train_epochs: int
    val_epochs: int
    learning_rate: float
    eval_interval: int
    eval_iters: int
    train_val_split: float
    max_steps: int
    device: str

    # Tokenizer configuration
    tokenizer_config: TokenizerConfig

    # Data paths
    data_path: str
    tokenizer_path: Optional[str]
    model_save_path: Optional[str]

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create a TrainingConfig from a dictionary."""
        # Extract and create the ModelConfig
        model_config_dict = config_dict.pop("gpt_model_config", {})
        gpt_model_config = gpt.ModelConfig(**model_config_dict)

        # Extract and create the TokenizerConfig
        tokenizer_config_dict = config_dict.pop("tokenizer_config", {})
        tokenizer_config = TokenizerConfig(**tokenizer_config_dict)

        # Create the TrainingConfig with the remaining parameters
        return cls(
            gpt_model_config=gpt_model_config,
            tokenizer_config=tokenizer_config,
            **config_dict,
        )


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_yaml_config(
    config: Union[TrainingConfig, gpt.ModelConfig, TokenizerConfig], config_path: str
) -> None:
    """Save a configuration to a YAML file."""
    # Convert model to dictionary
    config_dict = config.model_dump()

    # Save to YAML file
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logging.info(f"Configuration saved to {config_path}")


def validate_config(
    config_dict: Dict[str, Any], config_class: Type[BaseModel]
) -> Tuple[bool, List[str]]:
    """Validate a configuration dictionary against a Pydantic model."""
    errors = []

    try:
        # Let Pydantic do the validation
        config_class(**config_dict)
        return True, []
    except Exception as e:
        # Parse error message
        error_message = str(e)
        errors = [error_message]
        return False, errors


def setup_tokenizer_and_data(
    config: TrainingConfig,
) -> Tuple[Union[bpe.TiktokenTokenizer, bpe.BPETokenizer], torch.Tensor, torch.Tensor]:
    """Set up tokenizer and prepare training/validation data."""
    # Create directories if needed
    if config.tokenizer_path:
        Path(config.tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
    if config.model_save_path:
        Path(config.model_save_path).parent.mkdir(parents=True, exist_ok=True)

    # Load or create tokenizer
    tokenizer_type = config.tokenizer_config.type

    if config.tokenizer_path and Path(config.tokenizer_path).exists():
        logging.info(f"Loading tokenizer from {config.tokenizer_path}")
        if tokenizer_type == "tiktoken":
            tokenizer = bpe.TiktokenTokenizer(config.tokenizer_config.model_name)
        else:
            tokenizer = bpe.BPETokenizer.load(config.tokenizer_path)
    else:
        if tokenizer_type == "tiktoken":
            logging.info(
                f"Creating new TiktokenTokenizer with model {config.tokenizer_config.model_name}"
            )
            tokenizer = bpe.TiktokenTokenizer(config.tokenizer_config.model_name)
        else:
            max_vocab_size = config.tokenizer_config.max_vocab_size
            logging.info(
                f"Creating new BPETokenizer with max_vocab_size={max_vocab_size}"
            )
            tokenizer = bpe.BPETokenizer(max_vocab_size=max_vocab_size)

    # Read data
    logging.info(f"Reading data from {config.data_path}")
    full_text = Path(config.data_path).read_text()

    # Train BPE tokenizer if needed
    if tokenizer_type != "tiktoken" and (
        not config.tokenizer_path or not Path(config.tokenizer_path).exists()
    ):
        logging.info("Training BPE tokenizer")
        assert isinstance(tokenizer, bpe.BPETokenizer)
        tokenizer.train(list(full_text.encode("utf-8")))

        # Save the trained tokenizer if path is provided
        if config.tokenizer_path:
            logging.info(f"Saving tokenizer to {config.tokenizer_path}")
            tokenizer.save(config.tokenizer_path)

    # Create dataset
    logging.info("Encoding dataset")
    full_dataset = torch.tensor(
        tokenizer.encode(full_text),
        dtype=torch.long,
        device=config.device,
    )

    # Split into train/val
    split_idx = int(config.train_val_split * len(full_dataset))
    train = full_dataset[:split_idx]
    val = full_dataset[split_idx:]

    logging.info(
        f"Dataset created with {len(train)} training tokens and {len(val)} validation tokens"
    )

    return tokenizer, train, val


def get_batches(
    data: torch.Tensor, model_config: gpt.ModelConfig, num_epochs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create training or validation batches."""
    # Calculate how many complete blocks we can make
    num_complete_blocks = (
        len(data) - model_config.block_size
    ) // model_config.block_size

    # Create indices for the start of each block
    base_indices = (
        torch.arange(num_complete_blocks, device=data.device) * model_config.block_size
    )

    # Calculate number of complete batches we can make
    num_complete_batches = num_complete_blocks // model_config.batch_size

    # Truncate to multiple of batch_size
    base_indices = base_indices[: num_complete_batches * model_config.batch_size]

    # Shuffle the indices using the device-aware randperm
    indices = base_indices[torch.randperm(len(base_indices), device=data.device)]

    # Get x and y tensors
    x = torch.stack([data[i : i + model_config.block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + model_config.block_size + 1] for i in indices])

    # Duplicate batches for epochs > 1
    if num_epochs > 1:
        x = torch.cat([x] * num_epochs, dim=0)
        y = torch.cat([y] * num_epochs, dim=0)

    # Rearrange to (s, batch_size, block_size)
    x = rearrange(
        x,
        "(batch_size s) block_size -> s batch_size block_size",
        batch_size=model_config.batch_size,
    )
    y = rearrange(
        y,
        "(batch_size s) block_size -> s batch_size block_size",
        batch_size=model_config.batch_size,
    )
    return x, y


@torch.no_grad()
def estimate_loss(
    model: gpt.DecoderTransformer,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    tokenizer: Union[bpe.TiktokenTokenizer, bpe.BPETokenizer],
    device: str,
    batches_num: int = 2,
) -> dict:
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(batches_num, device=device)
        for k in range(batches_num):
            X = x_train if split == "train" else x_val
            Y = y_train if split == "train" else y_val
            # Use fixed indices for consistent evaluation
            eval_indices = torch.arange(batches_num, device=device)
            X = rearrange(X[eval_indices], "b s t -> (b s) t")
            Y = rearrange(Y[eval_indices], "b s t -> (b s) t")
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    # Generate sample text
    sample = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 100)[
        0
    ].tolist()
    logging.info(f"Output tokens: {sample}")
    logging.info(f"\n=========\nOutput text: {tokenizer.decode(sample)}\n=========")

    model.train()
    return out


def train_model(config: TrainingConfig) -> gpt.DecoderTransformer:
    """Train a GPT model with the given configuration."""
    # Setup tokenizer and data
    tokenizer, train_data, val_data = setup_tokenizer_and_data(config)

    # Update model config with vocab size
    config.gpt_model_config.vocab_size = tokenizer.vocab_size
    logging.info(f"Using vocabulary size: {tokenizer.vocab_size}")

    # Create batches
    x_train, y_train = get_batches(
        train_data, config.gpt_model_config, config.train_epochs
    )
    x_val, y_val = get_batches(val_data, config.gpt_model_config, config.val_epochs)

    # Initialize model
    model = gpt.DecoderTransformer(config.gpt_model_config)
    model = model.to(config.device)
    logging.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_steps
    )

    # Training loop
    logging.info(f"Starting training with {x_train.shape=}")
    for idx in range(config.max_steps):
        if idx % config.eval_interval == 0:
            losses = estimate_loss(
                model,
                x_train,
                y_train,
                x_val,
                y_val,
                tokenizer,
                config.device,
                config.eval_iters,
            )
            logging.info(
                f"step {idx}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        step_start = time.time()

        # Training step
        logits, loss = model(x_train[idx], y_train[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if idx % 10 == 0:
            step_time = (time.time() - step_start) * 1000
            logging.info(f"step {idx}: {step_time:.2f}ms per step")

    # Save model if path is provided
    if config.model_save_path:
        model_path = Path(config.model_save_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}")

    return model


def run_training(
    # Configuration can be loaded from YAML or from CLI parameters
    config_path: str = None,  # Path to YAML config file
    # Required parameters if no config file is provided
    data_path: str = None,
    model_save_path: str = None,
    tokenizer_path: str = None,
    # Model configuration parameters
    device: str = None,
    tokenizer_type: str = None,
    max_vocab_size: int = None,
    model_name: str = None,
    block_size: int = None,
    embed_dim: int = None,
    hidden_dim: int = None,
    batch_size: int = None,
    num_layers: int = None,
    head_size: int = None,
    num_heads: int = None,
    dropout: float = None,
    ffw_width_multiplier: int = None,
    # Training parameters
    train_epochs: int = None,
    val_epochs: int = None,
    learning_rate: float = None,
    train_val_split: float = None,
    eval_interval: int = None,
    eval_iters: int = None,
    max_steps: int = None,
    seed: int = None,
    # Extra parameters
    **extra_config_params,
):
    """CLI interface to train a GPT model.

    Args:
        data_path: Path to the training data.
        model_save_path: Path to save the trained model.
        tokenizer_path: Path to save/load the tokenizer.
        config_path: Path to a YAML configuration file.

        # Model configuration parameters
        device: Device to use for training ('cpu', 'cuda', 'mps').
        tokenizer_type: Type of tokenizer to use ('bpe' or 'tiktoken').
        max_vocab_size: Maximum vocabulary size for BPE tokenizer.
        model_name: Model name for tiktoken tokenizer.
        block_size: Context size for the model.
        embed_dim: Embedding dimension.
        hidden_dim: Hidden dimension for transformer.
        batch_size: Batch size for training.
        num_layers: Number of transformer layers.
        head_size: Size of each attention head.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        ffw_width_multiplier: Multiplier for FFW layer width.

        # Training parameters
        train_epochs: Number of training epochs.
        val_epochs: Number of validation epochs.
        learning_rate: Learning rate for optimization.
        train_val_split: Train/validation split ratio.
        eval_interval: Evaluation interval in steps.
        eval_iters: Number of evaluation iterations.
        max_steps: Maximum number of training steps.
        seed: Random seed for reproducibility.
    """
    # Handle configuration from YAML file if provided
    if config_path:
        logging.info(f"Loading configuration from {config_path}")
        config_dict = load_yaml_config(config_path)

        # Validate the configuration
        is_valid, errors = validate_config(config_dict, TrainingConfig)
        if not is_valid:
            error_msg = "\n".join(errors)
            raise ValueError(f"Invalid configuration in {config_path}:\n{error_msg}")

        # Create configuration from YAML
        try:
            config = TrainingConfig.from_dict(config_dict)

            # Override with command-line arguments if provided
            if data_path:
                config.data_path = data_path
            if model_save_path:
                config.model_save_path = model_save_path
            if tokenizer_path:
                config.tokenizer_path = tokenizer_path

        except Exception as e:
            raise ValueError(f"Error creating configuration from {config_path}: {e}")
    else:
        # Check for required parameters
        if not data_path or not model_save_path or not tokenizer_path:
            raise ValueError(
                "When not using a config file, you must provide data_path, model_save_path, and tokenizer_path"
            )

        # Create configuration from parameters
        if seed is not None:
            torch.manual_seed(seed)
        else:
            # Use a default seed if none is provided
            seed = 42
            torch.manual_seed(seed)

        # Create model config
        model_config = gpt.ModelConfig(
            vocab_size=max_vocab_size,  # Will be updated based on tokenizer
            block_size=block_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            num_layers=num_layers,
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            ffw_width_multiplier=ffw_width_multiplier,
            seed=seed,
            **{
                k: v
                for k, v in extra_config_params.items()
                if hasattr(gpt.ModelConfig, k)
            },
        )

        # Create tokenizer config
        tokenizer_config = TokenizerConfig(
            type=tokenizer_type,
            max_vocab_size=max_vocab_size,
            model_name=model_name,
        )

        # Create training config
        config = TrainingConfig(
            gpt_model_config=model_config,
            train_epochs=train_epochs,
            val_epochs=val_epochs,
            learning_rate=learning_rate,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            train_val_split=train_val_split,
            max_steps=max_steps,
            device=device,
            tokenizer_config=tokenizer_config,
            data_path=data_path,
            tokenizer_path=tokenizer_path,
            model_save_path=model_save_path,
        )

    # Save the configuration if requested
    if extra_config_params.get("save_config_path"):
        save_yaml_config(config, extra_config_params["save_config_path"])

    # Train the model
    train_model(config)

    # Return a success message instead of the model object to avoid Fire issues
    logging.info(f"Training complete. Model saved to {config.model_save_path}")
    return {
        "status": "success",
        "model_path": config.model_save_path,
        "tokenizer_path": config.tokenizer_path,
    }


def main():
    """Main entry point for CLI."""
    fire.Fire(run_training)


if __name__ == "__main__":
    main()
