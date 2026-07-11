"""Out-of-tree vLLM registration: dense-MLP Qwen3-Next.

vLLM's Qwen3NextForCausalLM assumes MoE (written for Qwen3-Next-80B-A3B);
its MixtureOfExperts mixin raises when every layer is dense. The
MiniCPM-V-4.6 text decoder is exactly a dense Qwen3-Next (verified
bitwise-isomorphic under transformers), so this subclass only relaxes the
MoE bookkeeping and registers under a distinct architecture name.

Launch: python evals/dspark/vllm_dense_qwen3next.py serve <model> ...
(registers, then defers to vllm's CLI main).
"""

from __future__ import annotations


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    class DenseQwen3NextForCausalLM(Qwen3NextForCausalLM):
        def set_moe_parameters(self):
            try:
                super().set_moe_parameters()
            except RuntimeError as err:
                if "No Qwen3Next layer found" not in str(err):
                    raise
                self.expert_weights = []
                self.moe_layers = []
                self.num_moe_layers = 0

    ModelRegistry.register_model(
        "DenseQwen3NextForCausalLM", DenseQwen3NextForCausalLM
    )


if __name__ == "__main__":
    import sys

    register()
    from vllm.entrypoints.cli.main import main

    sys.argv[0] = "vllm"
    main()
