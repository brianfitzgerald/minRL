import torch
from minrl.algorithms import sync_weights_to_vllm
from minrl.lora import apply_lora_to_model, LoRAConfig, LoRALinear


def test_sync_weights_to_vllm_with_lora(hf_model, vllm_model):
    """Test that sync_weights_to_vllm correctly merges and restores LoRA weights."""

    # Create a simple LoRA config
    lora_config = LoRAConfig(
        rank=2,
        alpha=8.0,
    )

    # Apply LoRA to the model
    apply_lora_to_model(hf_model, lora_config)

    # Count LoRA modules
    lora_modules = {
        name: module
        for name, module in hf_model.named_modules()
        if isinstance(module, LoRALinear)
    }
    assert len(lora_modules) > 0, "No LoRA modules found after applying LoRA"

    # Store original LoRA weights for comparison
    original_lora_weights = {}
    for name, module in lora_modules.items():
        original_lora_weights[name] = {
            "lora_A": module.lora_A.data.clone(),
            "lora_B": module.lora_B.data.clone(),
            "base_weight": module.base_layer.weight.data.clone(),
        }

    # Call sync_weights_to_vllm with lora=True
    # This should merge LoRA weights, sync to vLLM, and restore LoRA weights
    sync_weights_to_vllm(hf_model, vllm_model, lora=True)

    # Verify that LoRA weights were restored after syncing
    for name, module in lora_modules.items():
        assert torch.allclose(
            module.lora_A.data, original_lora_weights[name]["lora_A"], atol=1e-6
        ), f"LoRA A weights for {name} should be restored to original values"

        assert torch.allclose(
            module.lora_B.data, original_lora_weights[name]["lora_B"], atol=1e-6
        ), f"LoRA B weights for {name} should be restored to original values"

        # Base layer weights should also be restored (with merged delta subtracted)
        assert torch.allclose(
            module.base_layer.weight.data,
            original_lora_weights[name]["base_weight"],
            atol=1e-5,
        ), f"Base layer weights for {name} should be restored to original values"


def test_sync_weights_to_vllm_without_lora(hf_model, vllm_model):
    """Test that sync_weights_to_vllm works correctly without LoRA."""

    # Store original weights
    original_state_dict = hf_model.state_dict()

    # Call sync_weights_to_vllm with lora=False
    sync_weights_to_vllm(hf_model, vllm_model, lora=False)

    # Verify that the model weights are unchanged
    final_state_dict = hf_model.state_dict()
    for key in original_state_dict.keys():
        assert torch.equal(original_state_dict[key], final_state_dict[key]), (
            f"Value for {key} should remain unchanged"
        )
