"""Checkpoint diagnostics run only after C's final optimizer update.

This intentionally does not hook into the training loop, its random generator,
dropout, shuffled loader or optimizer schedule. No further training follows it.
"""
import csv
import gc
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import c_contract as contract
import c_protocol as protocol
import c_training as training


def tokenization_audit(tokenizer, parts, output):
    records = []
    for split in ('train', 'validation'):
        for row in parts[split]:
            raw = tokenizer.encode(row['text'], add_special_tokens=False)
            prompt = contract.encode_prompt(tokenizer, row['text'])
            records.append({'split': split, 'passage_id': row['passage_id'],
                'original_passage_tokens': len(raw), 'passage_tokens_removed': max(0, len(raw)-256),
                'full_prompt_tokens': len(prompt)})
    with (output/'tokenization_audit.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    return {split: {'n': len(parts[split]), 'truncated_passages': sum(
        r['passage_tokens_removed'] > 0 for r in records if r['split'] == split)} for split in parts}


def run_post_training_diagnostics(experiment_output, tokenizer, parts, summary):
    experiment_output = Path(experiment_output)
    contract.require(summary['experiment'] == protocol.C_NAME, 'Only C diagnostics are permitted.')
    contract.require(summary['status'] == 'completed_validation_only', 'Finish all three C epochs first.')
    contract.require(summary['frozen_base_verified'] is True, 'C frozen-base check missing.')
    contract.require(set(parts) == {'train', 'validation'}, 'Diagnostics accept train/validation only.')
    output = experiment_output/'post_training_diagnostics'; output.mkdir(exist_ok=False)
    tokenization = tokenization_audit(tokenizer, parts, output)
    reports = []
    history = json.loads((experiment_output/'training_history.json').read_text())
    contract.require([r['epoch'] for r in history] == [1, 2, 3], 'Expected exactly three completed epochs.')
    for record in history:
        epoch = record['epoch']
        contract.require(record['checkpoint'] == 'adapter_epoch_' + str(epoch), 'Unexpected checkpoint path.')
        checkpoint = experiment_output/record['checkpoint']
        epoch_output = output/('epoch_' + str(epoch)); epoch_output.mkdir(exist_ok=False)
        loaded_tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
        loaded_tokenizer.padding_side = 'right'
        contract.require(loaded_tokenizer.pad_token_id == tokenizer.pad_token_id, 'Reloaded padding token changed.')
        codes = training.mapping(loaded_tokenizer)
        examples = {split: training.make_examples(loaded_tokenizer, parts[split]) for split in parts}
        for split in parts:
            contract.require(examples[split] == training.make_examples(tokenizer, parts[split]),
                             'Checkpoint tokenizer changed prompt/label encoding.')
        base = AutoModelForCausalLM.from_pretrained(training.MODEL_ID, revision=training.MODEL_REVISION,
            torch_dtype=torch.float32, trust_remote_code=False, use_safetensors=True,
            attn_implementation='eager').to('cuda')
        base.config.pad_token_id = loaded_tokenizer.pad_token_id; base.config.use_cache = False
        base_hash = training.fingerprint(dict(base.named_parameters()))
        contract.require(base_hash == summary['base_sha256_before'], 'Reloaded base fingerprint differs.')
        model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=False, autocast_adapter_dtype=False)
        model.config.use_cache = False; model.eval()
        contract.require(all(not p.requires_grad for p in model.parameters()), 'Diagnostic parameters unexpectedly trainable.')
        adapter_parameters = {n: p for n, p in model.named_parameters() if 'lora_' in n}
        contract.require(bool(adapter_parameters), 'Saved adapter weights are missing.')
        adapter_hash = training.fingerprint(adapter_parameters)
        expected_hash = json.loads((experiment_output/f'adapter_fingerprint_epoch_{epoch}.json').read_text())['sha256']
        contract.require(adapter_hash == expected_hash, 'Reloaded adapter fingerprint differs.')
        report = {'epoch': epoch, 'checkpoint': record['checkpoint'], 'base_sha256': base_hash,
                  'adapter_sha256': adapter_hash, 'scoring_stage': 'after_all_training_updates'}
        for split in ('train', 'validation'):
            prediction_file = epoch_output/(split + '_predictions.csv')
            metrics, unweighted_ce = training.validate(model, loaded_tokenizer, parts[split], examples[split],
                torch.device('cuda'), torch.float16, codes, prediction_file)
            error_count = protocol.save_errors(prediction_file, epoch_output/(split + '_errors.csv'))
            report[split] = dict(metrics, unweighted_ce=unweighted_ce, error_count=error_count)
        report['saved_reloaded_validation_agreement'] = protocol.compare_prediction_files(
            experiment_output/f'validation_predictions_epoch_{epoch}.csv', epoch_output/'validation_predictions.csv')
        # Each reloaded validation metric must reproduce its original checkpoint report.
        for key in ['accuracy', 'macro_f1', 'per_class', 'confusion_matrix', 'prediction_counts']:
            contract.require(report['validation'][key] == record['validation'][key], 'Reloaded validation metric differs: ' + key)
        contract.save_json(epoch_output/'metrics.json', report); reports.append(report)
        print('Reloaded epoch', epoch, '| train accuracy', round(report['train']['accuracy'], 6),
              '| validation accuracy', round(report['validation']['accuracy'], 6),
              '| validation macro-F1', round(report['validation']['macro_f1'], 6))
        del adapter_parameters, model, base, loaded_tokenizer, examples
        gc.collect(); torch.cuda.empty_cache()
    selected = next(r for r in reports if r['epoch'] == summary['selected']['epoch'])
    result = {'status': 'PASS', 'all_three_checkpoints_reloaded': True, 'epochs': reports,
              'selected_epoch': selected['epoch'], 'selected': selected,
              'saved_reloaded_selected_adapter_agreement': selected['saved_reloaded_validation_agreement'],
              'tokenization': tokenization, 'diagnostics_occur_after_training_only': True,
              'selection_uses_validation_only': True, 'independent_human_evaluation': False}
    contract.save_json(output/'diagnostic_summary.json', result)
    return result
