import unittest

import torch

from train_block_residuals import Block, ModelConfig, count_parameters


def make_config(
    variant: str, residual_output_activation: str = "gelu"
) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        variant=variant,
        norm="pre",
        residual_output_activation=residual_output_activation,
    )


class ResidualOutputActivationTest(unittest.TestCase):
    def test_identity_variant_matches_standard_exactly(self) -> None:
        torch.manual_seed(7)
        standard = Block(make_config("standard")).eval()
        activated = Block(
            make_config("standard_act_both", "identity")
        ).eval()
        activated.load_state_dict(standard.state_dict())

        x = torch.randn(2, 8, 16)
        with torch.no_grad():
            expected = standard(x)
            actual = activated(x)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_gelu_variants_preserve_shape_and_parameter_count(self) -> None:
        torch.manual_seed(11)
        standard = Block(make_config("standard")).eval()
        x = torch.randn(2, 8, 16)

        for variant in (
            "standard_act_attn",
            "standard_act_ffn",
            "standard_act_both",
        ):
            with self.subTest(variant=variant):
                block = Block(make_config(variant, "gelu")).eval()
                self.assertEqual(count_parameters(block), count_parameters(standard))
                with torch.no_grad():
                    output = block(x)
                self.assertEqual(output.shape, x.shape)
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
