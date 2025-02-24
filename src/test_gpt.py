import pytest
import torch
from einops import rearrange

from gpt import DecoderTransformer, ModelConfig, MultiHeadAttention


@pytest.fixture
def model_inputs():
    """Generate deterministic inputs for testing"""
    return (
        torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 9],
                [3, 4, 5, 6, 7, 8, 9, 10],
                [4, 5, 6, 7, 8, 9, 10, 11],
            ]
        ),
        torch.tensor(
            [
                [2, 3, 4, 5, 6, 7, 8, 9],
                [3, 4, 5, 6, 7, 8, 9, 10],
                [4, 5, 6, 7, 8, 9, 10, 11],
                [5, 6, 7, 8, 9, 10, 11, 12],
            ]
        ),
    )


@pytest.fixture
def generation_config():
    """Default generation configuration"""
    return {"max_new_tokens": 8, "temperature": 0.8, "num_samples": 3}


class TestModelInitialization:
    def test_deterministic_initialization(self):
        """Test that model initialization is deterministic with fixed seed"""
        config = ModelConfig(
            vocab_size=100,
            block_size=8,
            embed_dim=32,
            hidden_dim=32,
            batch_size=4,
            num_layers=2,
            num_heads=4,
            head_size=8,
            dropout=0.1,
            ffw_width_multiplier=4,
            seed=42,
        )
        # Get initial parameters
        model1 = DecoderTransformer(config)
        initial_params = dict(model1.named_parameters())

        model2 = DecoderTransformer(config)
        new_params = dict(model2.named_parameters())

        # Check that parameters are identical
        for name in initial_params:
            assert torch.allclose(
                initial_params[name], new_params[name]
            ), f"Parameters differ for {name}"

    @pytest.mark.parametrize(
        "config_changes",
        [
            {"num_layers": 1},
            {
                "num_heads": 2,
                "head_size": 16,
            },  # Keep total attention dim same as hidden_dim
            {
                "hidden_dim": 64,
                "embed_dim": 64,
                "head_size": 16,
            },  # Keep dimensions compatible
        ],
    )
    def test_different_configurations(self, config_changes):
        """Test model initialization with different configurations"""
        config = ModelConfig(
            vocab_size=100,
            block_size=8,
            embed_dim=32,
            hidden_dim=32,
            batch_size=4,
            num_layers=2,
            num_heads=4,
            head_size=8,
            dropout=0.1,
            ffw_width_multiplier=4,
            seed=42,
        )
        for key, value in config_changes.items():
            setattr(config, key, value)
        model = DecoderTransformer(config)
        # Verify the model can do a forward pass
        x = torch.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.block_size),
        )
        logits, _ = model(x)
        assert logits.shape == (
            config.batch_size,
            config.block_size,
            config.vocab_size,
        )


class TestForwardPass:
    def test_deterministic_output(self, model_inputs):
        """Test forward pass produces expected shape and deterministic outputs"""
        config = ModelConfig(
            vocab_size=100,
            block_size=8,
            embed_dim=32,
            hidden_dim=32,
            batch_size=4,
            num_layers=2,
            num_heads=4,
            head_size=8,
            dropout=0.1,
            ffw_width_multiplier=4,
            seed=42,
        )
        model = DecoderTransformer(config)
        model.eval()
        x, targets = model_inputs

        # First forward pass
        logits1, loss1 = model(x, targets)

        # Second forward pass with same inputs
        logits2, loss2 = model(x, targets)

        # Check shapes
        assert logits1.shape == (
            config.batch_size,
            config.block_size,
            config.vocab_size,
        ), "Unexpected logits shape"

        # Check deterministic behavior
        assert torch.allclose(
            logits1, logits2
        ), f"Logits differ: {logits1} != {logits2}"
        assert torch.allclose(loss1, loss2), f"Losses differ: {loss1} != {loss2}"

    def test_different_batch_sizes(self):
        """Test forward pass with different batch sizes"""
        config = ModelConfig(
            vocab_size=100,
            block_size=8,
            embed_dim=32,
            hidden_dim=32,
            batch_size=4,
            num_layers=2,
            num_heads=4,
            head_size=8,
            dropout=0.1,
            ffw_width_multiplier=4,
            seed=42,
        )
        model = DecoderTransformer(config)
        model.eval()

        batch_sizes = [1, 2, 8]
        for batch_size in batch_sizes:
            x = torch.randint(
                0,
                config.vocab_size,
                (batch_size, config.block_size),
            )
            logits, _ = model(x)
            assert logits.shape == (
                batch_size,
                config.block_size,
                config.vocab_size,
            ), f"Failed for batch_size={batch_size}"


class TestTraining:
    def test_gradient_determinism(self, model_inputs):
        """Test that gradients are deterministic across multiple backward passes"""
        config = ModelConfig(
            vocab_size=100,
            block_size=8,
            embed_dim=32,
            hidden_dim=32,
            batch_size=4,
            num_layers=2,
            num_heads=4,
            head_size=8,
            dropout=0.0,  # Set to 0 for deterministic behavior
            ffw_width_multiplier=4,
            seed=42,
        )
        x, targets = model_inputs

        model = DecoderTransformer(config)
        model.train()

        # First forward/backward pass
        _, loss1 = model(x, targets)
        loss1.backward()
        grads1 = {name: param.grad.clone() for name, param in model.named_parameters()}

        # Reset gradients
        model.zero_grad()

        # Second forward/backward pass
        _, loss2 = model(x, targets)
        loss2.backward()
        grads2 = {name: param.grad.clone() for name, param in model.named_parameters()}

        # Compare gradients
        for name in grads1:
            assert torch.allclose(
                grads1[name], grads2[name]
            ), f"Gradients differ for {name}"

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_gradient_determinism_different_seeds(self, model_inputs, seed):
        """Test gradient determinism with different random seeds"""
        config = ModelConfig(
            vocab_size=100,
            block_size=8,
            embed_dim=32,
            hidden_dim=32,
            batch_size=4,
            num_layers=2,
            num_heads=4,
            head_size=8,
            dropout=0.0,  # Set to 0 for deterministic behavior
            ffw_width_multiplier=4,
            seed=seed,
        )
        x, targets = model_inputs

        # Create two models with same seed
        model1 = DecoderTransformer(config)
        model2 = DecoderTransformer(config)

        # Put models in train mode
        model1.train()
        model2.train()

        # Forward/backward on both models
        _, loss1 = model1(x, targets)
        loss1.backward()
        grads1 = {name: param.grad.clone() for name, param in model1.named_parameters()}

        _, loss2 = model2(x, targets)
        loss2.backward()
        grads2 = {name: param.grad.clone() for name, param in model2.named_parameters()}

        # Compare gradients with looser tolerances
        for name in grads1:
            assert torch.allclose(
                grads1[name], grads2[name]
            ), f"Gradients differ for {name} with seed {seed}"


class TestGeneration:
    def test_deterministic_generation(self, generation_config):
        """Test that generation is deterministic with fixed seed"""
        config = ModelConfig(
            vocab_size=100,
            block_size=8,
            embed_dim=32,
            hidden_dim=32,
            batch_size=4,
            num_layers=2,
            num_heads=4,
            head_size=8,
            dropout=0.1,
            ffw_width_multiplier=4,
            seed=42,
        )
        model = DecoderTransformer(config)
        model.eval()

        x = torch.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.block_size),
        )

        max_new_tokens = generation_config["max_new_tokens"]

        # First generation
        generated1 = model.generate(x, max_new_tokens=max_new_tokens)

        # Second generation with same input
        generated2 = model.generate(x, max_new_tokens=max_new_tokens)

        # Check shapes
        assert generated1.shape == (
            config.batch_size,
            config.block_size + max_new_tokens,
        ), "Unexpected generation shape"

        # Check deterministic behavior
        assert torch.equal(generated1, generated2), "Generation is not deterministic"

    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
    def test_temperature_sampling(self, model_inputs, generation_config, temperature):
        """Test generation with different temperature values"""
        config = ModelConfig(
            vocab_size=100,
            block_size=8,
            embed_dim=32,
            hidden_dim=32,
            batch_size=4,
            num_layers=2,
            num_heads=4,
            head_size=8,
            dropout=0.1,
            ffw_width_multiplier=4,
            seed=42,
        )
        model = DecoderTransformer(config)
        model.eval()

        x, _ = model_inputs
        generated = model.generate(
            x,
            max_new_tokens=generation_config["max_new_tokens"],
            temperature=temperature,
        )
        assert generated.shape[1] > x.shape[1], "No tokens were generated"

    def test_multiple_completions(self, model_inputs, generation_config):
        """Test generating multiple completions for the same input"""
        config = ModelConfig(
            vocab_size=100,
            block_size=8,
            embed_dim=32,
            hidden_dim=32,
            batch_size=4,
            num_layers=2,
            num_heads=4,
            head_size=8,
            dropout=0.1,
            ffw_width_multiplier=4,
            seed=42,
        )
        model = DecoderTransformer(config)
        model.eval()

        x, _ = model_inputs
        generated = model.generate(
            x,
            max_new_tokens=generation_config["max_new_tokens"],
            temperature=generation_config["temperature"],
            num_samples=generation_config["num_samples"],
        )

        # Check shape includes num_samples dimension
        assert generated.shape == (
            config.batch_size,
            generation_config["num_samples"],
            config.block_size + generation_config["max_new_tokens"],
        ), "Unexpected shape for multiple completions"


class TestMultiHeadAttentionMechanism:
    def test_causal_masking(self):
        """Test attention patterns in the model"""
        config = ModelConfig(
            vocab_size=100,
            block_size=8,
            embed_dim=32,
            hidden_dim=32,
            batch_size=4,
            num_layers=2,
            num_heads=4,
            head_size=8,
            dropout=0.1,
            ffw_width_multiplier=4,
            seed=42,
        )
        model = DecoderTransformer(config)

        ## check the model uses multi-head attention, if not skip the test
        if not isinstance(model.blocks[0].self_attention, MultiHeadAttention):
            pytest.skip("Model does not use multi-head attention")

        # Create sample input
        x = torch.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.block_size),
        )

        # Get attention scores from first head
        first_block = model.blocks[0]
        first_head = first_block.self_attention.heads[0]

        # Forward pass through embeddings
        embeds = model.embedding(x)
        pos_embeds = model.positional_embedding(torch.arange(x.shape[1]))
        hidden_states = embeds + pos_embeds

        attention_output = first_head(hidden_states)

        # Check output shape
        assert attention_output.shape == (
            config.batch_size,
            config.block_size,
            config.head_size,
        ), "Unexpected attention output shape"

        # Check attention scores
        with torch.no_grad():
            _, T, C = hidden_states.shape
            q = first_head.q(hidden_states)
            k = first_head.k(hidden_states)

            scores = (
                q @ rearrange(k, "b t h -> b h t") * first_head.q.out_features**-0.5
            )
            mask = first_head.tril[:T, :T] == 0
            scores = scores.masked_fill(mask, float("-inf"))

            # Verify causal masking
            for i in range(config.block_size):
                for j in range(config.block_size):
                    if j > i:  # Future positions should be masked
                        assert scores[0, i, j].item() == float(
                            "-inf"
                        ), f"Future position ({i}, {j}) not properly masked"
                    else:  # Past and present positions should have finite scores
                        assert scores[0, i, j].item() != float(
                            "-inf"
                        ), f"Past/present position ({i}, {j}) incorrectly masked"
