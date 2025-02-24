import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
import yaml

import bpe
import gpt
import training


@pytest.fixture
def sample_text():
    """Provide a small sample text for testing."""
    return "Hello world! This is a test for the training module."


@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_data_file(temp_dir, sample_text):
    """Create a temporary data file with sample text."""
    data_path = os.path.join(temp_dir, "sample.txt")
    with open(data_path, "w") as f:
        f.write(sample_text)
    return data_path


@pytest.fixture
def tokenizer_path(temp_dir):
    """Path for a temporary tokenizer file."""
    return os.path.join(temp_dir, "tokenizer.pkl")


@pytest.fixture
def model_path(temp_dir):
    """Path for a temporary model file."""
    return os.path.join(temp_dir, "model.pt")


@pytest.fixture
def mock_model():
    """Create a mock model that can be trained."""
    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    model.return_value = (torch.zeros(1, 10), torch.tensor(0.5))
    return model


def test_setup_tokenizer_and_data(sample_data_file, tokenizer_path):
    """Test setting up tokenizer and data for training."""
    # Create a minimal training config
    config = training.TrainingConfig(
        gpt_model_config=gpt.ModelConfig(
            vocab_size=256,
            block_size=4,
            embed_dim=8,
            hidden_dim=8,
            batch_size=2,
            num_layers=1,
            head_size=4,
            num_heads=1,
            dropout=0.0,
            ffw_width_multiplier=2,
            seed=42,
        ),
        train_epochs=1,
        val_epochs=1,
        learning_rate=0.1,
        eval_interval=10,
        eval_iters=2,
        train_val_split=0.8,
        max_steps=10,
        device="cpu",
        tokenizer_config=training.TokenizerConfig(
            type="bpe", max_vocab_size=256, model_name="gpt2"
        ),
        data_path=sample_data_file,
        tokenizer_path=tokenizer_path,
        model_save_path=None,
    )

    # Call the function
    tokenizer, train_data, val_data = training.setup_tokenizer_and_data(config)

    # Check the results
    assert isinstance(tokenizer, bpe.BPETokenizer)
    assert isinstance(train_data, torch.Tensor)
    assert isinstance(val_data, torch.Tensor)
    assert len(train_data) > 0
    assert Path(tokenizer_path).exists()


def test_get_batches():
    """Test batch generation."""
    # Create sample data
    data = torch.arange(20).long()

    # Create model config
    model_config = gpt.ModelConfig(
        vocab_size=256,
        block_size=4,
        embed_dim=8,
        hidden_dim=8,
        batch_size=2,
        num_layers=1,
        head_size=4,
        num_heads=1,
        dropout=0.0,
        ffw_width_multiplier=2,
        seed=42,
    )

    # Get batches
    x, y = training.get_batches(data, model_config, num_epochs=1)

    # Check shapes
    assert x.shape[0] > 0  # Should have at least one step
    assert x.shape[1] == model_config.batch_size  # Batch size should match
    assert x.shape[2] == model_config.block_size  # Block size should match
    assert x.shape == y.shape  # Inputs and targets should have the same shape

    # Check offset relationship (y should be x shifted by 1)
    # Take a batch and check the relationship
    assert torch.all(x[0, 0, 1:] == y[0, 0, :-1])


@pytest.fixture
def longer_sample_text():
    """Generate a longer sample text for training tests."""
    return """
    This is a longer sample text for training tests.
    It needs to be long enough to create multiple blocks for training.
    We need more text to ensure we can create batches for the model.
    The model has a small block size, but we still need enough data.
    Let's add some more lines to make sure we have enough data.
    This should be sufficient for our testing purposes.
    We want to test the training dynamics of our tiny model.
    The text doesn't need to make perfect sense, just be long enough.
    We're using a small block size and batch size for quick testing.
    """


@pytest.fixture
def longer_sample_data_file(temp_dir, longer_sample_text):
    """Create a temporary data file with longer sample text."""
    data_path = os.path.join(temp_dir, "longer_sample.txt")
    with open(data_path, "w") as f:
        f.write(longer_sample_text)
    return data_path


def test_train_model(longer_sample_data_file, tokenizer_path, model_path):
    """Test the actual training dynamics with a tiny model."""
    # Create a training config with a very small model for quick testing
    config = training.TrainingConfig(
        gpt_model_config=gpt.ModelConfig(
            vocab_size=256,  # Will be updated based on tokenizer
            block_size=4,  # Very small block size for testing
            embed_dim=8,
            hidden_dim=8,
            batch_size=2,  # Small batch size for testing
            num_layers=1,
            head_size=4,
            num_heads=2,
            dropout=0.0,
            ffw_width_multiplier=2,
            seed=42,
        ),
        train_epochs=1,
        val_epochs=1,
        learning_rate=0.01,
        eval_interval=2,
        eval_iters=1,
        train_val_split=0.8,
        max_steps=5,  # Very short training run for test
        device="cpu",
        tokenizer_config=training.TokenizerConfig(
            type="bpe",
            max_vocab_size=256,
            model_name="gpt2",
        ),
        data_path=longer_sample_data_file,
        tokenizer_path=tokenizer_path,
        model_save_path=model_path,
    )

    # Train the model
    model = training.train_model(config)

    # Verify the model was trained
    assert isinstance(model, gpt.DecoderTransformer)

    # Verify the model has parameters
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0

    # Verify the model file was saved
    assert Path(model_path).exists() and Path(model_path).stat().st_size > 0

    # Verify the tokenizer was saved
    assert Path(tokenizer_path).exists() and Path(tokenizer_path).stat().st_size > 0

    # Load the model and verify it can generate output
    loaded_state = torch.load(model_path)
    assert isinstance(loaded_state, dict)
    assert len(loaded_state) > 0


def test_estimate_loss(mock_model):
    """Test the loss estimation function."""
    # Create test data
    x_train = torch.ones((5, 2, 4), dtype=torch.long)
    y_train = torch.ones((5, 2, 4), dtype=torch.long)
    x_val = torch.ones((5, 2, 4), dtype=torch.long)
    y_val = torch.ones((5, 2, 4), dtype=torch.long)

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "Generated text"

    # Call the function
    with patch("training.rearrange", side_effect=lambda x, *args, **kwargs: x):
        losses = training.estimate_loss(
            mock_model, x_train, y_train, x_val, y_val, mock_tokenizer, "cpu", 2
        )

    # Check the results
    assert "train" in losses
    assert "val" in losses
    assert mock_model.generate.call_count == 1
    assert mock_tokenizer.decode.call_count == 1


def test_run_training(sample_data_file, tokenizer_path, model_path):
    """Test the CLI interface for training."""
    # Mock train_model to avoid actual training
    with patch("training.train_model") as mock_train:
        # Call the function
        result = training.run_training(
            data_path=sample_data_file,
            model_save_path=model_path,
            tokenizer_path=tokenizer_path,
            device="cpu",
            max_steps=2,  # Minimal training for test
        )

        # Verify the function was called with the correct config
        mock_train.assert_called_once()
        config = mock_train.call_args[0][0]
        assert config.data_path == sample_data_file
        assert config.model_save_path == model_path
        assert config.tokenizer_path == tokenizer_path

        # Verify the return object has the expected fields
        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["model_path"] == model_path
        assert result["tokenizer_path"] == tokenizer_path


def test_yaml_config():
    """Test loading and validating YAML configuration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a test YAML config file
        config_file = os.path.join(tmp_dir, "test_config.yaml")
        data_file = os.path.join(tmp_dir, "data.txt")
        tokenizer_file = os.path.join(tmp_dir, "tokenizer.pkl")
        model_file = os.path.join(tmp_dir, "model.pt")

        # Create a sample data file
        with open(data_file, "w") as f:
            f.write("This is a test sample for YAML configuration loading.")

        # Create a sample YAML config
        config = {
            "gpt_model_config": {
                "vocab_size": 256,
                "block_size": 4,
                "embed_dim": 8,
                "hidden_dim": 8,
                "batch_size": 1,
                "num_layers": 1,
                "head_size": 4,
                "num_heads": 2,
                "dropout": 0.0,
                "ffw_width_multiplier": 2,
                "seed": 42,
            },
            "train_epochs": 1,
            "val_epochs": 1,
            "learning_rate": 0.01,
            "eval_interval": 1,
            "eval_iters": 1,
            "train_val_split": 0.8,
            "max_steps": 1,
            "device": "cpu",
            "tokenizer_config": {
                "type": "bpe",
                "max_vocab_size": 256,
                "model_name": "gpt2",
            },
            "data_path": data_file,
            "tokenizer_path": tokenizer_file,
            "model_save_path": model_file,
        }

        # Save the config to YAML file
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Load and validate the config
        loaded_config = training.load_yaml_config(config_file)
        is_valid, errors = training.validate_config(
            loaded_config, training.TrainingConfig
        )

        # Test validation
        assert is_valid, f"Config validation failed with errors: {errors}"

        # Test config creation
        try:
            config_obj = training.TrainingConfig.from_dict(loaded_config)
            assert config_obj.data_path == data_file
            assert config_obj.tokenizer_path == tokenizer_file
            assert config_obj.model_save_path == model_file
            assert config_obj.gpt_model_config.vocab_size == 256
            assert config_obj.tokenizer_config.max_vocab_size == 256
        except Exception as e:
            pytest.fail(f"Failed to create config object: {e}")

        # Test config saving
        config_save_path = os.path.join(tmp_dir, "saved_config.yaml")
        training.save_yaml_config(config_obj, config_save_path)
        assert os.path.exists(config_save_path)

        # Test saved config loading
        loaded_saved_config = training.load_yaml_config(config_save_path)
        assert loaded_saved_config["data_path"] == data_file
        assert loaded_saved_config["tokenizer_path"] == tokenizer_file


def test_yaml_cli():
    """Test the YAML configuration CLI integration."""
    import subprocess
    import os

    # Create a test directory
    test_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "yaml_test"
    )
    os.makedirs(test_dir, exist_ok=True)

    # Create a longer sample data file
    data_file = os.path.join(test_dir, "yaml_test_data.txt")
    with open(data_file, "w") as f:
        f.write("""
        This is a longer sample text for testing the YAML configuration integration.
        It needs to have enough tokens to create proper validation batches.
        We're using a small block size and batch size for quick testing.
        Let's add some more text to ensure we have enough data for validation.
        The model configuration needs to be carefully chosen to work with this amount of data.
        """)

    # Define output paths
    model_file = os.path.join(test_dir, "yaml_test_model.pt")
    tokenizer_file = os.path.join(test_dir, "yaml_test_tokenizer.pkl")
    config_file = os.path.join(test_dir, "yaml_test_config.yaml")

    # Remove output files if they exist from previous runs
    for file in [model_file, tokenizer_file, config_file]:
        if os.path.exists(file):
            os.unlink(file)

    # Create a test YAML config
    config = {
        "gpt_model_config": {
            "vocab_size": 256,
            "block_size": 2,  # Very small block size for quick testing
            "embed_dim": 8,
            "hidden_dim": 8,
            "batch_size": 1,  # Small batch size
            "num_layers": 1,
            "head_size": 4,
            "num_heads": 2,
            "dropout": 0.0,
            "ffw_width_multiplier": 2,
            "seed": 42,
        },
        "train_epochs": 1,
        "val_epochs": 1,
        "learning_rate": 0.01,
        "eval_interval": 1,
        "eval_iters": 1,
        "train_val_split": 0.8,
        "max_steps": 1,
        "device": "cpu",
        "tokenizer_config": {
            "type": "bpe",
            "max_vocab_size": 256,
            "model_name": "gpt2",
        },
        "data_path": data_file,
        "tokenizer_path": tokenizer_file,
        "model_save_path": model_file,
    }

    # Save the config to YAML file
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Build the command with uv
    cmd = [
        "uv",
        "run",
        "python",
        os.path.join(os.path.dirname(__file__), "training.py"),
        "--config_path",
        config_file,
    ]

    # Run the command and capture output
    try:
        # We use check=True to raise an exception if the return code is non-zero
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )

        # Check if files were created
        assert os.path.exists(
            tokenizer_file
        ), f"Tokenizer file {tokenizer_file} was not created"
        assert os.path.exists(model_file), f"Model file {model_file} was not created"

        # Check output for success indication
        assert "status:" in result.stdout, "Status not found in output"
        assert "success" in result.stdout, "Success status not found in output"

    except subprocess.CalledProcessError as e:
        # If the command fails, include the output for debugging
        pytest.fail(
            f"YAML CLI command failed with code {e.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout: {e.stdout}\n"
            f"Stderr: {e.stderr}"
        )
    except AssertionError as e:
        pytest.fail(f"YAML CLI test assertion failed: {e}")
    finally:
        # Clean up output files (optional - you might want to keep them for debugging)
        # for file in [model_file, tokenizer_file, config_file]:
        #    if os.path.exists(file):
        #        os.unlink(file)
        pass


def test_fire_cli():
    """Test the Fire CLI integration using a subprocess call with uv."""
    import subprocess
    import os

    # Create a test directory within the data directory
    test_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "cli_test"
    )
    os.makedirs(test_dir, exist_ok=True)

    # Create a longer sample data file to ensure we have enough tokens for validation
    data_file = os.path.join(test_dir, "cli_test_data.txt")
    with open(data_file, "w") as f:
        f.write("""
        This is a longer sample text for testing the Fire CLI integration.
        It needs to have enough tokens to create proper validation batches.
        We're using a small block size and batch size for quick testing.
        Let's add some more text to ensure we have enough data for validation.
        The model configuration needs to be carefully chosen to work with this amount of data.
        We want to test that the CLI interface works correctly with the Fire library.
        """)

    # Define output paths
    model_file = os.path.join(test_dir, "cli_test_model.pt")
    tokenizer_file = os.path.join(test_dir, "cli_test_tokenizer.pkl")

    # Remove output files if they exist from previous runs
    for file in [model_file, tokenizer_file]:
        if os.path.exists(file):
            os.unlink(file)

    # Build the command with uv
    # Use very small parameters to ensure we have enough data for validation
    cmd = [
        "uv",
        "run",
        "python",
        os.path.join(os.path.dirname(__file__), "training.py"),
        "--data_path=" + data_file,
        "--model_save_path=" + model_file,
        "--tokenizer_path=" + tokenizer_file,
        "--max_steps=1",
        "--device=cpu",
        "--block_size=2",  # Very small block size
        "--embed_dim=8",
        "--hidden_dim=8",
        "--batch_size=1",  # Single batch size
        "--num_layers=1",
        "--train_val_split=0.8",  # More validation data
        "--head_size=4",
        "--num_heads=2",
        "--eval_iters=1",
        "--eval_interval=1",  # Ensure evaluation happens
    ]

    # Run the command and capture output
    try:
        # We use check=True to raise an exception if the return code is non-zero
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )

        # Check if files were created
        assert os.path.exists(
            tokenizer_file
        ), f"Tokenizer file {tokenizer_file} was not created"
        assert os.path.exists(model_file), f"Model file {model_file} was not created"

        # Check output for success indication
        assert "status:" in result.stdout, "Status not found in output"
        assert "success" in result.stdout, "Success status not found in output"

    except subprocess.CalledProcessError as e:
        # If the command fails, include the output for debugging
        pytest.fail(
            f"CLI command failed with code {e.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout: {e.stdout}\n"
            f"Stderr: {e.stderr}"
        )
    except AssertionError as e:
        pytest.fail(f"CLI test assertion failed: {e}")
    finally:
        # Clean up output files (optional - you might want to keep them for debugging)
        # for file in [model_file, tokenizer_file]:
        #    if os.path.exists(file):
        #        os.unlink(file)
        pass
