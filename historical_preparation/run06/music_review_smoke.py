"""Actual small LoRA update/save/reload check, run only when called in Colab."""
from pathlib import Path
import gc
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import music_review_pipeline as pipeline
import music_review_training as training


def run_smoke_test(base, tokenizer, train_rows, validation_rows, device, amp_dtype, output_dir):
    """Use six training rows and three validation probes; never accepts a test set.

    Returns the original frozen base with all smoke adapters removed (no merge).
    This verifies execution, not model usefulness or final performance.
    """
    folder = Path(output_dir)
    folder.mkdir(exist_ok=False)
    selected = []
    for label in pipeline.LABELS:
        selected.extend([r for r in train_rows if r['label'] == label][:2])
    assert len(selected) == 6 and len(validation_rows) >= 3
    probes = validation_rows[:3]
    wrapped = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=.05, target_modules=['q_proj', 'v_proj'], bias='none'))
    wrapped.config.use_cache = False
    wrapped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    wrapped.enable_input_require_grads()
    parameters = {name: p for name, p in wrapped.named_parameters() if p.requires_grad}
    assert parameters and all('lora_' in name for name in parameters), 'Only adapter weights may train.'
    before = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=2e-4)
    scaler = torch.amp.GradScaler('cuda', init_scale=256.0, enabled=amp_dtype == torch.float16)
    losses = []
    wrapped.train()
    for start in range(0, len(selected), 2):
        examples = [training.supervised_example(tokenizer, r['text'], r['label']) for r in selected[start:start+2]]
        batch = {k: v.to(device) for k, v in training.collate_examples(examples, tokenizer.pad_token_id).items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            loss = wrapped(**batch).loss
        assert bool(torch.isfinite(loss)), 'Nonfinite smoke loss.'
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0)
        assert bool(torch.isfinite(norm)), 'Nonfinite smoke gradients.'
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    assert any(not torch.equal(before[name], p.detach().cpu()) for name, p in parameters.items()), 'No adapter weight changed.'
    wrapped.eval()
    expected_predictions, expected_scores = training.predict_closed_set(wrapped, tokenizer, probes, device, amp_dtype)
    adapter_dir = folder / 'adapter'
    wrapped.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    assert (adapter_dir / 'adapter_model.safetensors').is_file()
    # unload, rather than merge_and_unload, restores the unchanged base.
    restored_base = wrapped.unload()
    del optimizer, parameters, before, wrapped, loss, batch
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(restored_base, adapter_dir, is_trainable=False)
    actual_predictions, actual_scores = training.predict_closed_set(reloaded, tokenizer, probes, device, amp_dtype)
    assert actual_predictions == expected_predictions, 'Saved/reloaded predictions differ.'
    expected_values = torch.tensor([[s[label] for label in pipeline.LABELS] for s in expected_scores])
    actual_values = torch.tensor([[s[label] for label in pipeline.LABELS] for s in actual_scores])
    assert torch.allclose(expected_values, actual_values, atol=2e-4, rtol=2e-4), 'Reloaded adapter scores differ.'
    restored_base = reloaded.unload()
    restored_base.gradient_checkpointing_disable()
    restored_base.disable_input_require_grads()
    assert not any('lora_' in name for name, _ in restored_base.named_parameters()), 'Smoke adapter was not removed.'
    record = {'status': 'PASS', 'kind': 'execution_smoke_not_performance_result',
        'training_rows': [r['passage_id'] for r in selected],
        'validation_probe_rows': [r['passage_id'] for r in probes], 'test_rows_used': 0,
        'optimizer_steps': len(losses), 'finite_losses': losses, 'adapter_weights_changed': True,
        'save_reload_predictions_and_scores_match': True, 'smoke_adapters_removed_without_merge': True,
        'label_provenance': 'AI-assisted across all splits; NOT independent human evaluation'}
    pipeline.save_json(folder / 'smoke_test.json', record)
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    return restored_base, record
