import pytest
import torch
from einops import rearrange

from gpt import DecoderTransformer, ModelConfig


@pytest.fixture
def model_config():
    return ModelConfig(
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


@pytest.fixture
def model_inputs(model_config):
    """Generate deterministic inputs for testing"""
    x = torch.randint(
        0,
        model_config.vocab_size,
        (model_config.batch_size, model_config.block_size),
        generator=model_config.generator,
    )
    targets = torch.randint(
        0,
        model_config.vocab_size,
        (model_config.batch_size, model_config.block_size),
        generator=model_config.generator,
    )
    return x, targets


@pytest.fixture
def model(model_config):
    """Create a model instance for testing"""
    return DecoderTransformer(model_config)


@pytest.fixture
def eval_model(model):
    """Create a model instance in eval mode for testing"""
    model.eval()
    return model


class TestModelInitialization:
    def test_deterministic_initialization(self, model_config):
        """Test that model initialization is deterministic with fixed seed"""
        # Get initial parameters
        model1 = DecoderTransformer(model_config)
        initial_params = dict(model1.named_parameters())

        model2 = DecoderTransformer(model_config)
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
    def test_different_configurations(self, model_config, config_changes):
        """Test model initialization with different configurations"""
        for key, value in config_changes.items():
            setattr(model_config, key, value)
        model = DecoderTransformer(model_config)
        # Verify the model can do a forward pass
        x = torch.randint(
            0,
            model_config.vocab_size,
            (model_config.batch_size, model_config.block_size),
            generator=model_config.generator,
        )
        logits, _ = model(x)
        assert logits.shape == (
            model_config.batch_size,
            model_config.block_size,
            model_config.vocab_size,
        )


class TestForwardPass:
    def test_deterministic_output(self, model_config, model_inputs, eval_model):
        """Test forward pass produces expected shape and deterministic outputs"""
        x, targets = model_inputs

        # First forward pass
        logits1, loss1 = eval_model(x, targets)

        # Second forward pass with same inputs
        logits2, loss2 = eval_model(x, targets)

        # Check shapes
        assert logits1.shape == (
            model_config.batch_size,
            model_config.block_size,
            model_config.vocab_size,
        ), "Unexpected logits shape"

        # Check deterministic behavior
        assert torch.allclose(
            logits1, logits2
        ), f"Logits differ: {logits1} != {logits2}"
        assert torch.allclose(loss1, loss2), f"Losses differ: {loss1} != {loss2}"

    def test_different_batch_sizes(self, model_config, eval_model):
        """Test forward pass with different batch sizes"""
        batch_sizes = [1, 2, 8]
        for batch_size in batch_sizes:
            x = torch.randint(
                0,
                model_config.vocab_size,
                (batch_size, model_config.block_size),
                generator=model_config.generator,
            )
            logits, _ = eval_model(x)
            assert logits.shape == (
                batch_size,
                model_config.block_size,
                model_config.vocab_size,
            ), f"Failed for batch_size={batch_size}"


class TestTraining:
    def test_gradient_determinism(self, model_config, model_inputs):
        """Test that gradients are deterministic across multiple backward passes"""
        x, targets = model_inputs

        # Reinitialize model with 0 dropout
        model_config.dropout = 0.0
        model = DecoderTransformer(model_config)
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
    def test_gradient_determinism_different_seeds(
        self, model_config, model_inputs, seed
    ):
        """Test gradient determinism with different random seeds"""
        x, targets = model_inputs

        # Set dropout to 0 in config for deterministic behavior
        model_config.dropout = 0.0

        # Create two models with same seed
        model_config.seed = seed
        model1 = DecoderTransformer(model_config)
        model2 = DecoderTransformer(model_config)

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
    def test_deterministic_generation(self, model_config, eval_model):
        """Test that generation is deterministic with fixed seed"""
        x = torch.randint(
            0,
            model_config.vocab_size,
            (model_config.batch_size, model_config.block_size),
            generator=model_config.generator,
        )

        max_new_tokens = 4

        # First generation
        generated1 = eval_model.generate(x, max_new_tokens=max_new_tokens)

        # Second generation with same input
        generated2 = eval_model.generate(x, max_new_tokens=max_new_tokens)

        # Check shapes
        assert generated1.shape == (
            model_config.batch_size,
            model_config.block_size + max_new_tokens,
        ), "Unexpected generation shape"

        # Check deterministic behavior
        assert torch.equal(generated1, generated2), "Generation is not deterministic"

    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
    def test_temperature_sampling(self, eval_model, model_inputs, temperature):
        """Test generation with different temperature values"""
        x, _ = model_inputs
        generated = eval_model.generate(x, max_new_tokens=4, temperature=temperature)
        assert generated.shape[1] > x.shape[1], "No tokens were generated"


class TestAttentionMechanism:
    def test_causal_masking(self, model_config):
        """Test attention patterns in the model"""
        model = DecoderTransformer(model_config)

        # Create sample input
        x = torch.randint(
            0,
            model_config.vocab_size,
            (model_config.batch_size, model_config.block_size),
            generator=model_config.generator,
        )

        # Get attention scores from first head
        first_block = model.blocks[0]
        first_head = first_block.self_attention.heads[0]

        # Forward pass through embeddings
        embeds = model.embedding(x)
        pos_embeds = model.positional_embedding(
            model.positions[: model_config.block_size]
        )
        hidden_states = embeds + pos_embeds

        attention_output = first_head(hidden_states)

        # Check output shape
        assert attention_output.shape == (
            model_config.batch_size,
            model_config.block_size,
            model_config.head_size,
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
            for i in range(model_config.block_size):
                for j in range(model_config.block_size):
                    if j > i:  # Future positions should be masked
                        assert scores[0, i, j].item() == float(
                            "-inf"
                        ), f"Future position ({i}, {j}) not properly masked"
                    else:  # Past and present positions should have finite scores
                        assert scores[0, i, j].item() != float(
                            "-inf"
                        ), f"Past/present position ({i}, {j}) incorrectly masked"

    def test_attention_output_range(self, model, model_inputs, model_config):
        """Test that attention outputs are in a reasonable range"""
        x, _ = model_inputs
        first_head = model.blocks[0].self_attention.heads[0]

        embeds = model.embedding(x)
        pos_embeds = model.positional_embedding(
            model.positions[: model_config.block_size]
        )
        hidden_states = embeds + pos_embeds

        with torch.no_grad():
            attention_output = first_head(hidden_states)
            assert not torch.isnan(attention_output).any(), "NaN in attention output"
            assert not torch.isinf(attention_output).any(), "Inf in attention output"
