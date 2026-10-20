"""Exactly two user-run LoRA experiments; accepts training and validation only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['WANDB_DISABLED'] = 'true'
import csv
import gc
import hashlib
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, set_seed, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
import validation_contract as contract

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
EXPERIMENTS = {'A_unweighted_ABC': False, 'B_train_weighted_ABC': True}


def fingerprint(parameters):
    """Hash all named weights in small CPU chunks, never clone a full base model."""
    h = hashlib.sha256()
    for name, parameter in sorted(parameters.items()):
        h.update(name.encode()); h.update(str(tuple(parameter.shape)).encode()); h.update(str(parameter.dtype).encode())
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), 1_048_576):
            h.update(flat[start:start+1_048_576].to('cpu').contiguous().numpy().tobytes())
    return h.hexdigest()


def mapping(tokenizer):
    contract.require(tokenizer.eos_token_id is not None and tokenizer.pad_token_id is not None,
                     'Tokenizer must have explicit EOS and padding tokens.')
    ids = [tokenizer.encode(code, add_special_tokens=False) for code in contract.CODES]
    contract.require(all(len(x) == 1 for x in ids) and len({x[0] for x in ids}) == 3,
                     'A/B/C must be three distinct single tokens.')
    values = [x[0] for x in ids]
    contract.require([tokenizer.decode([i], skip_special_tokens=False) for i in values] == list(contract.CODES),
                     'A/B/C token round-trip failed.')
    return values


def make_examples(tokenizer, rows):
    result = []
    for row in rows:
        ids = contract.encode_prompt(tokenizer, row['text'])
        contract.require(1 < len(ids) <= 512, 'Invalid prompt length.')
        # No answer/EOS is appended. The target is a separate category index.
        result.append({'input_ids': ids, 'target': contract.LABELS.index(row['label'])})
    return result


def collate(examples, pad_token_id):
    longest = max(len(r['input_ids']) for r in examples)
    return {'input_ids': torch.tensor([r['input_ids'] + [pad_token_id] * (longest-len(r['input_ids'])) for r in examples]),
            'attention_mask': torch.tensor([[1] * len(r['input_ids']) + [0] * (longest-len(r['input_ids'])) for r in examples]),
            'targets': torch.tensor([r['target'] for r in examples], dtype=torch.long)}


def class_logits(model, batch, codes, amp_dtype):
    mask = batch['attention_mask']
    positions = mask.sum(dim=1) - 1
    contract.require(bool((positions >= 0).all()), 'Empty prompt.')
    expected_mask = torch.arange(mask.shape[1], device=mask.device)[None, :] <= positions[:, None]
    contract.require(bool(torch.equal(mask.bool(), expected_mask)), 'Expected contiguous right-padded prompts.')
    with torch.autocast(device_type='cuda', dtype=amp_dtype):
        output = model(input_ids=batch['input_ids'], attention_mask=mask, use_cache=False)
        selected = output.logits[torch.arange(len(positions), device=positions.device)[:, None],
                                 positions[:, None], torch.tensor(codes, device=positions.device)[None, :]]
    contract.require(bool(torch.isfinite(selected).all()), 'Nonfinite category logits; reject predictions.')
    return selected.float()


@torch.inference_mode()
def causal_checks(base, tokenizer, train_rows, device, amp_dtype, codes):
    """Check final valid prompt positions, batching, and future-answer invariance."""
    probes = [next(r for r in train_rows if r['label'] == label) for label in contract.LABELS]
    examples = make_examples(tokenizer, probes)
    batch = {k: v.to(device) for k, v in collate(examples, tokenizer.pad_token_id).items()}
    base.eval()
    combined = class_logits(base, batch, codes, amp_dtype)
    largest_difference = 0.0
    for i, example in enumerate(examples):
        ids = example['input_ids']
        single = {k: v.to(device) for k, v in collate([example], tokenizer.pad_token_id).items()}
        unpadded = class_logits(base, single, codes, amp_dtype)[0]
        future = torch.tensor([ids + [codes[example['target']], tokenizer.eos_token_id]], device=device)
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            forced = base(input_ids=future, attention_mask=torch.ones_like(future), use_cache=False).logits[0, len(ids)-1, codes].float()
        contract.require(torch.allclose(combined[i], unpadded, atol=.02, rtol=.002), 'Padded versus unpadded scoring differs.')
        contract.require(torch.allclose(unpadded, forced, atol=.02, rtol=.002), 'Future-answer tokens changed the causal category position.')
        largest_difference = max(largest_difference, float((unpadded-forced).abs().max()), float((combined[i]-unpadded).abs().max()))
        contract.require(ids == contract.encode_prompt(tokenizer, probes[i]['text']), 'Prompt construction is inconsistent.')
    return {'status': 'PASS', 'training_probe_ids': [r['passage_id'] for r in probes],
            'A_B_C_token_ids': dict(zip(contract.CODES, codes)), 'target_position': 'final non-padding prompt token',
            'training_inputs_include_answers': False, 'largest_logit_difference': largest_difference,
            'comparison_tolerance': {'atol': .02, 'rtol': .002}}


@torch.inference_mode()
def validate(model, tokenizer, rows, examples, device, amp_dtype, codes, output_file):
    model.eval(); predicted = []; output_rows = []; ce_sum = 0.0; exact_top_ties = 0
    for start in range(0, len(examples), 2):
        batch = {k: v.to(device) for k, v in collate(examples[start:start+2], tokenizer.pad_token_id).items()}
        logits = class_logits(model, batch, codes, amp_dtype)
        top_two = logits.topk(2, dim=-1).values
        exact_top_ties += int((top_two[:, 0] == top_two[:, 1]).sum())
        ce_sum += float(F.cross_entropy(logits, batch['targets'], reduction='sum'))
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        guesses = logits.argmax(dim=-1).cpu().tolist()
        for j, (guess, scores) in enumerate(zip(guesses, probabilities)):
            label = contract.LABELS[guess]; predicted.append(label)
            output_rows.append(dict(rows[start+j], predicted_label=label,
                restricted_score_A=scores[0], restricted_score_B=scores[1], restricted_score_C=scores[2]))
    if output_file is not None:
        with Path(output_file).open('x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(output_rows[0])); w.writeheader(); w.writerows(output_rows)
    metrics = contract.classification_metrics([r['label'] for r in rows], predicted)
    metrics['exact_top_ties'] = exact_top_ties
    return metrics, ce_sum / len(rows)


def run_experiment(name, tokenizer, train_rows, validation_rows, manifest, root, versions):
    contract.require(name in EXPERIMENTS, 'Only the two predeclared experiments are permitted.')
    output = Path(root) / name; output.mkdir(exist_ok=False)
    seed = manifest['seed']; set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device('cuda'); amp_dtype = torch.float16  # Known working T4 path.
    codes = mapping(tokenizer)
    train_examples = make_examples(tokenizer, train_rows); validation_examples = make_examples(tokenizer, validation_rows)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        torch_dtype=torch.float32, trust_remote_code=False, use_safetensors=True, attn_implementation='eager').to(device)
    base.config.pad_token_id = tokenizer.pad_token_id; base.config.use_cache = False
    contract.require(not getattr(base, 'peft_config', None), 'Fresh base unexpectedly has adapter metadata.')
    contract.require(not any('lora_' in p for p, _ in base.named_parameters()), 'Fresh base has LoRA parameters.')
    checks = causal_checks(base, tokenizer, train_rows, device, amp_dtype, codes)
    contract.save_json(output / 'causal_checks.json', checks)
    frozen = dict(base.named_parameters())
    for p in frozen.values(): p.requires_grad_(False)
    base_before = fingerprint(frozen)
    set_seed(seed)  # Same initial adapters for A and B, independent of loading/check probes.
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        lora_dropout=.05, target_modules=['q_proj', 'v_proj'], bias='none'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    contract.require(bool(trainable) and all('lora_' in n for n in trainable), 'Only new LoRA parameters may train.')
    contract.require(all(not p.requires_grad for p in frozen.values()), 'A base parameter became trainable.')
    adapter_before = fingerprint(trainable)
    weights = contract.class_weights(train_rows) if EXPERIMENTS[name] else [1.0, 1.0, 1.0]
    weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
    settings = {'experiment': name, 'model': MODEL_ID, 'revision': MODEL_REVISION, 'seed': seed,
        'epochs': 3, 'batch_size': 2, 'gradient_accumulation': 4, 'learning_rate': .0002,
        'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': .05, 'lora_targets': ['q_proj','v_proj'],
        'objective': 'A/B/C-only cross entropy at final valid prompt position; no EOS loss',
        'class_weights_A_B_C': weights, 'weight_source': 'training labels only' if EXPERIMENTS[name] else 'unweighted control',
        'accumulation_normalization': 'sum of target class weights over the entire accumulated group',
        'base_sha256_before': base_before, 'adapter_initial_sha256': adapter_before,
        'versions': versions, 'gpu': torch.cuda.get_device_name(0), 'manifest': manifest,
        'original_validation_macro_f1': contract.ORIGINAL_VALIDATION_MACRO_F1,
        'independent_human_evaluation': False, 'status': 'configured_not_completed'}
    contract.save_json(output / 'settings.json', settings)
    baseline_metrics, baseline_ce = validate(model, tokenizer, validation_rows, validation_examples, device, amp_dtype, codes,
                                            output / 'validation_predictions_before_training.csv')
    contract.save_json(output / 'before_training_validation.json', {'metrics': baseline_metrics, 'unweighted_ce': baseline_ce})
    generator = torch.Generator().manual_seed(seed)
    from functools import partial
    loader = DataLoader(train_examples, batch_size=2, shuffle=True, generator=generator,
                        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id), num_workers=0)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=.0002, weight_decay=.01)
    planned_steps = 3 * math.ceil(len(loader)/4)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(.1*planned_steps)), planned_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    history = []; step_records = []; successful = skipped = attempted = 0
    set_seed(seed)
    for epoch in range(1, 4):
        model.train(); weighted_sum = unweighted_sum = weight_sum = 0.0; seen = 0
        epoch_success = epoch_skip = 0
        iterator = iter(loader)
        for group_index in range(math.ceil(len(loader)/4)):
            group = []
            for _ in range(4):
                try: group.append(next(iterator))
                except StopIteration: break
            denominator = sum(float(weight_tensor[b['targets'].to(device)].sum()) for b in group)
            contract.require(denominator > 0, 'Invalid accumulation denominator.')
            optimizer.zero_grad(set_to_none=True)
            for cpu_batch in group:
                batch = {k: v.to(device) for k, v in cpu_batch.items()}
                logits = class_logits(model, batch, codes, amp_dtype)
                ce = F.cross_entropy(logits, batch['targets'], reduction='none')
                weights_here = weight_tensor[batch['targets']]
                numerator = (ce * weights_here).sum()
                loss = numerator / denominator
                contract.require(bool(torch.isfinite(loss)), 'Nonfinite loss; stop rather than silently continue.')
                scaler.scale(loss).backward()
                weighted_sum += float(numerator.detach()); unweighted_sum += float(ce.detach().sum())
                weight_sum += float(weights_here.sum()); seen += len(ce)
            scaler.unscale_(optimizer)
            # Inspect the norm without mutating gradients; float64 avoids overflow in the diagnostic sum.
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(p.grad.detach().double())
                for p in trainable.values() if p.grad is not None]))
            finite_gradients = bool(torch.isfinite(norm))
            if finite_gradients: torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0)
            old_scale = scaler.get_scale(); lr_used = optimizer.param_groups[0]['lr']
            scaler.step(optimizer); scaler.update(); attempted += 1
            updated = scaler.get_scale() >= old_scale
            contract.require(finite_gradients or not updated, 'A nonfinite-gradient update was not skipped; reject run.')
            if updated:
                scheduler.step(); successful += 1; epoch_success += 1
            else:
                skipped += 1; epoch_skip += 1
            step_records.append({'epoch':epoch, 'accumulated_group':group_index+1, 'attempt':attempted,
                'optimizer_step_succeeded':updated, 'learning_rate_used':lr_used,
                'scale_before':old_scale, 'scale_after':scaler.get_scale(),
                'finite_gradient_norm':finite_gradients, 'unclipped_gradient_norm':float(norm) if finite_gradients else None,
                'group_examples':sum(len(b['targets']) for b in group), 'group_weight_denominator':denominator})
            optimizer.zero_grad(set_to_none=True)
        contract.require(seen == len(train_rows), 'An epoch did not visit exactly the frozen training rows.')
        metrics, val_ce = validate(model, tokenizer, validation_rows, validation_examples, device, amp_dtype, codes,
                                   output / f'validation_predictions_epoch_{epoch}.csv')
        checkpoint = output / f'adapter_epoch_{epoch}'; checkpoint.mkdir(exist_ok=False)
        model.save_pretrained(checkpoint, safe_serialization=True); tokenizer.save_pretrained(checkpoint)
        record = {'epoch':epoch, 'training_weighted_ce':weighted_sum/weight_sum,
            'training_unweighted_ce':unweighted_sum/seen, 'validation_unweighted_ce':val_ce,
            'validation':metrics, 'successful_optimizer_steps':epoch_success, 'skipped_optimizer_steps':epoch_skip,
            'checkpoint':checkpoint.name}
        contract.save_json(output / f'epoch_{epoch}.json', record); history.append(record)
        print(name, 'epoch', epoch, 'validation macro-F1', round(metrics['macro_f1'],6),
              'predictions',metrics['prediction_counts'], 'updates/skips',epoch_success,epoch_skip)
    contract.require(successful > 0, 'No successful optimizer update.')
    adapter_after = fingerprint(trainable)
    contract.require(adapter_after != adapter_before, 'Adapter parameters never changed.')
    contract.require(all(not p.requires_grad and p.grad is None for p in frozen.values()), 'Frozen-base gradient invariant failed.')
    base_after = fingerprint(frozen)
    contract.require(base_after == base_before, 'Frozen base weights changed; reject this run.')
    best = contract.select_checkpoint(history)
    summary = {'experiment':name, 'status':'completed_validation_only', 'selected':best,
        'original_validation_macro_f1':contract.ORIGINAL_VALIDATION_MACRO_F1,
        'delta_vs_original_validation':best['validation']['macro_f1']-contract.ORIGINAL_VALIDATION_MACRO_F1,
        'attempted_optimizer_steps':attempted, 'successful_optimizer_steps':successful, 'skipped_optimizer_steps':skipped,
        'successful_nonzero_learning_rate_steps':sum(r['optimizer_step_succeeded'] and r['learning_rate_used'] > 0 for r in step_records),
        'planned_optimizer_steps':planned_steps, 'base_sha256_before':base_before, 'base_sha256_after':base_after,
        'adapter_initial_sha256':adapter_before, 'adapter_final_sha256':adapter_after,
        'frozen_base_verified':True, 'independent_human_evaluation':False,
        'scope':'training and validation only; no confirmatory final comparison'}
    contract.save_json(output / 'optimizer_steps.json', step_records)
    contract.save_json(output / 'training_history.json', history)
    contract.save_json(output / 'selected_checkpoint.json', best)
    contract.save_json(output / 'experiment_summary.json', summary)
    del optimizer, scheduler, scaler, trainable, frozen, model, base, logits, ce, loss, numerator, batch
    gc.collect(); torch.cuda.empty_cache()
    return summary
