"""Reload selected artifacts and reproduce development predictions before test access."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import math
import expansion_contract as contract


def verify_prediction_agreement(expected_rows, actual_rows, development_rows, actual_metrics, actual_ce, expected_metrics, expected_ce):
    contract.require(len(expected_rows) == len(actual_rows) == len(development_rows), 'Preflight prediction count differs.')
    for old, new, source in zip(expected_rows, actual_rows, development_rows):
        contract.require(old['passage_id'] == new['passage_id'] == source['passage_id'] and old['label'] == new['label'] == source['label'], 'Preflight prediction order/labels differ.')
        contract.require(old['predicted_label'] == new['predicted_label'], 'Reloaded model predictions differ on validation.')
        for code in contract.CODES:
            key = 'restricted_score_' + code
            contract.require(math.isclose(float(old[key]), float(new[key]), abs_tol=.005, rel_tol=.002), 'Reloaded model scores differ on validation.')
    contract.require(actual_metrics == expected_metrics, 'Reloaded validation metrics differ.')
    contract.require(math.isfinite(actual_ce) and math.isclose(actual_ce, expected_ce, abs_tol=.005, rel_tol=.002), 'Reloaded validation CE differs.')


def preflight_selected_model(train_validation_package, authorized_freeze_sha256, trained_run):
    """Future authorized GPU operation. Never receives/opens a test package."""
    run = Path(trained_run)
    selection = contract.verify_selection(run, authorized_freeze_sha256)
    if (run / 'test_attempt_started.json').exists():
        raise FileExistsError('A final test attempt already exists; no preflight/retry after test.')
    parts, manifest = contract.load_package(train_validation_package, authorized_freeze_sha256)
    settings = json.loads((run / 'settings.json').read_text())
    contract.require(manifest['rows_digest'] == settings['manifest']['rows_digest'], 'Preflight development rows differ from training.')
    output = run / ('preflight_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    model = base = None
    try:
        # Heavy libraries are deliberately lazy: all preparation tests run without
        # importing them or loading a model. This path runs only after authorization.
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
        from peft import PeftModel
        from expansion_training import mapping, make_examples, validate, fingerprint
        contract.require(torch.cuda.is_available(), 'Authorized CUDA runtime required for preflight.')
        set_seed(contract.SEED)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        selected = run / selection['selected']['checkpoint']
        tokenizer = AutoTokenizer.from_pretrained(selected, local_files_only=True, trust_remote_code=False)
        signature, encodings = contract.verify_tokenizer_preflight(tokenizer, parts, settings)
        codes = mapping(tokenizer)
        causal = json.loads((run / 'causal_checks.json').read_text())
        contract.require(dict(zip(contract.CODES, codes)) == causal['A_B_C_token_ids'], 'Saved tokenizer category token IDs changed.')
        device = torch.device('cuda'); amp_dtype = torch.float16
        base = AutoModelForCausalLM.from_pretrained(contract.MODEL_ID, revision=contract.MODEL_REVISION,
            torch_dtype=torch.float32, trust_remote_code=False, use_safetensors=True, attn_implementation='eager').to(device)
        base.config.pad_token_id = tokenizer.pad_token_id; base.config.use_cache = False
        contract.require(not getattr(base, 'peft_config', None) and not any('lora_' in n for n, _ in base.named_parameters()), 'Preflight requires a fresh pinned base.')
        frozen = dict(base.named_parameters())
        for parameter in frozen.values(): parameter.requires_grad_(False)
        base_hash = fingerprint(frozen)
        contract.require(base_hash == settings['base_sha256_before'], 'Reloaded frozen base differs from training.')
        model = PeftModel.from_pretrained(base, selected, is_trainable=False)
        contract.require(not any(p.requires_grad for p in model.parameters()), 'Preflight adapter must be inference-only.')
        adapter = {n: p for n, p in model.named_parameters() if 'lora_' in n}
        contract.require(bool(adapter), 'Selected adapter has no LoRA parameters.')
        adapter_hash = fingerprint(adapter)
        contract.verify_model_fingerprints(base_hash, adapter_hash, settings, selection['selected'])
        rows = parts['validation']; examples = make_examples(tokenizer, rows)
        # The same already-reloaded model survives into final scoring. Disabled
        # adapters must first reproduce the saved unadapted validation predictions.
        with model.disable_adapter():
            metrics, ce = validate(model, tokenizer, rows, examples, device, amp_dtype, codes,
                                   output / 'unadapted_validation_predictions.csv')
        # Adapter enable/disable implementations may toggle requires_grad flags.
        # Restore inference-only flags without changing any weights or activation.
        for parameter in model.parameters(): parameter.requires_grad_(False)
        reference = json.loads((run / 'before_training_validation.json').read_text())
        verify_prediction_agreement(contract.read_csv(run / 'validation_predictions_before_training.csv'),
            contract.read_csv(output / 'unadapted_validation_predictions.csv'), rows, metrics, ce, reference['metrics'], reference['unweighted_ce'])
        metrics, ce = validate(model, tokenizer, rows, examples, device, amp_dtype, codes,
                               output / 'adapted_validation_predictions.csv')
        epoch = selection['selected']['epoch']
        verify_prediction_agreement(contract.read_csv(run / ('validation_predictions_epoch_' + str(epoch) + '.csv')),
            contract.read_csv(output / 'adapted_validation_predictions.csv'), rows, metrics, ce,
            selection['selected']['validation'], selection['selected']['validation_unweighted_ce'])
        contract.verify_model_fingerprints(fingerprint(frozen), fingerprint(adapter), settings, selection['selected'])
        contract.verify_selection(run, authorized_freeze_sha256)
        proof = {'schema': 'music-expanded-preflight-v1', 'status': 'passed',
                 'freeze_sha256': authorized_freeze_sha256, 'selection_receipt_sha256': contract.sha(run / 'selection_receipt.json'),
                 'tokenizer_signature': signature, 'encoding_sha256': encodings,
                 'base_sha256': base_hash, 'adapter_sha256': adapter_hash,
                 'unadapted_validation_agreement': True, 'adapted_validation_agreement': True,
                 'test_data_opened': False, 'rows_encoded': {'train': 400, 'validation': 200},
                 'validation_rows_compared_per_model': len(rows),
                 'prediction_agreement': 'exact labels/metrics; score and CE tolerance abs .005, rel .002',
                 'prediction_files': {p.name: contract.sha(p) for p in output.glob('*.csv')}}
        receipt_path = output / 'preflight.json'; contract.save_json(receipt_path, proof)
        return {'model': model, 'tokenizer': tokenizer, 'frozen': frozen, 'adapter': adapter,
                'base_sha256': base_hash, 'adapter_sha256': adapter_hash, 'settings': settings,
                'selection': selection, 'manifest': manifest, 'development_rows': parts,
                'receipt_path': receipt_path, 'device': device, 'amp_dtype': amp_dtype, 'codes': codes}
    except Exception as error:
        contract.save_json(output / 'preflight_failed.json', {'status': 'failed_before_test_access',
            'error_type': type(error).__name__, 'error': str(error), 'test_attempt_consumed': False})
        model = base = None
        gc.collect()
        raise


def verify_prepared_context(prepared, trained_run, authorized_freeze_sha256):
    """No inference: recheck the live preflight objects before consuming the gate."""
    from expansion_training import fingerprint
    selection = contract.verify_preflight_receipt(trained_run, authorized_freeze_sha256, prepared['receipt_path'])
    contract.verify_tokenizer_preflight(prepared['tokenizer'], prepared['development_rows'], prepared['settings'])
    contract.verify_model_fingerprints(fingerprint(prepared['frozen']), fingerprint(prepared['adapter']), prepared['settings'], selection['selected'])
    contract.require(not any(p.requires_grad for p in prepared['model'].parameters()), 'Prepared final model became trainable.')
    return selection
