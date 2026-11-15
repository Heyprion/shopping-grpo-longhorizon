"""Preserve per-sequence layout for veRL's fused PPO head with padded Qwen input."""


def dense_token_outputs_to_nested(output, input_ids, calculate_entropy, torch):
    lengths = input_ids.offsets().diff()
    batch_size = len(lengths)
    starts = torch.zeros_like(lengths)

    def nested(values):
        values = values.reshape(batch_size, -1)
        padded = torch.nested.narrow(values, 1, starts, lengths, layout=torch.jagged)
        return torch.nested.nested_tensor_from_jagged(
            torch.cat([row for row in padded.unbind()]), input_ids.offsets())

    result = {"log_probs": nested(output.log_probs)}
    if calculate_entropy:
        result["entropy"] = nested(output.entropy)
    return result


def install_fused_output_layout():
    import torch
    import verl.utils.tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    if getattr(FSDPEngineWithLMHead.prepare_model_outputs, "_shopping_fused_layout", False):
        return
    original = FSDPEngineWithLMHead.prepare_model_outputs

    def prepare(self, output, output_args, micro_batch, logits_processor_func):
        fused = tu.get_non_tensor_data(micro_batch, "use_fused_kernels", default=False)
        remove_padding = tu.get_non_tensor_data(micro_batch, "use_remove_padding", default=True)
        if fused and not remove_padding:
            pad_mode = tu.get_non_tensor_data(micro_batch, "pad_mode", default=DatasetPadMode.NO_PADDING)
            if pad_mode != DatasetPadMode.NO_PADDING:
                raise ValueError("Fused Shopping PPO requires the NO_PADDING TensorDict layout")
            entropy = tu.get_non_tensor_data(micro_batch, "calculate_entropy", default=False)
            return dense_token_outputs_to_nested(output, micro_batch["input_ids"], entropy, torch)
        return original(self, output, output_args, micro_batch, logits_processor_func)

    prepare._shopping_fused_layout = True
    FSDPEngineWithLMHead.prepare_model_outputs = prepare
