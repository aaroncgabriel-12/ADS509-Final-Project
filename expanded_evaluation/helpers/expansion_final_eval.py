"""One post-selection test comparison. Never trains or chooses a checkpoint."""
from pathlib import Path
import gc
import json
import expansion_contract as contract
import expansion_preflight as preflight


def prepare_test_comparison(train_validation_package, authorized_freeze_sha256, trained_run, *, authorized=False):
    """Must finish before the notebook even requests the test package upload."""
    contract.require(authorized is True, 'Authorize the exact freeze before selected-model preflight.')
    if (Path(trained_run) / 'test_attempt_started.json').exists():
        raise FileExistsError('A final test attempt already exists; stop.')
    return preflight.preflight_selected_model(train_validation_package, authorized_freeze_sha256, trained_run)


def run_test_comparison(test_package, authorized_freeze_sha256, trained_run, prepared, *, authorized=False):
    contract.require(authorized is True, 'Aaron must authorize the exact frozen data and this one final test comparison.')
    run = Path(trained_run)
    if (run / 'test_attempt_started.json').exists():
        raise FileExistsError('A final test attempt already exists; stop.')
    # Neither test data nor the one-time gate is touched unless the saved tokenizer,
    # selected adapter and development-prediction checks have all succeeded.
    preflight.verify_prepared_context(prepared, run, authorized_freeze_sha256)
    selection = contract.claim_test_attempt(run, authorized_freeze_sha256, prepared['receipt_path'])
    output = run / 'one_time_test_comparison'; output.mkdir(exist_ok=False)
    try:
        parts, manifest = contract.load_package(test_package, authorized_freeze_sha256, test=True)
        rows = parts['test']
        import torch
        from expansion_training import make_examples, validate, fingerprint
        settings = prepared['settings']
        training_manifest = settings['manifest']
        test_ids = {r['passage_id'] for r in rows}
        contract.require(not test_ids & set(training_manifest['membership']), 'Test IDs overlap development data.')
        for key in ('group_id', 'review_id'):
            contract.require(not {r[key] for r in rows} & {m[key] for m in training_manifest['membership'].values()}, 'Test leakage: ' + key)
        model, tokenizer = prepared['model'], prepared['tokenizer']
        frozen, adapter = prepared['frozen'], prepared['adapter']
        device, amp_dtype, codes = prepared['device'], prepared['amp_dtype'], prepared['codes']
        examples = make_examples(tokenizer, rows)
        with model.disable_adapter():
            unadapted, unadapted_ce = validate(model, tokenizer, rows, examples, device, amp_dtype, codes,
                                             output / 'unadapted_test_predictions.csv')
        for parameter in model.parameters(): parameter.requires_grad_(False)
        selected = run / selection['selected']['checkpoint']
        adapted, adapted_ce = validate(model, tokenizer, rows, examples, device, amp_dtype, codes,
                                     output / 'adapted_test_predictions.csv')
        contract.verify_model_fingerprints(fingerprint(frozen), fingerprint(adapter), settings, selection['selected'])
        result = {'status': 'completed_one_time_test_comparison', 'freeze_sha256': authorized_freeze_sha256,
                  'test_csv_sha256': manifest['csv_sha256']['test'], 'selected_checkpoint': selected.name,
                  'selection_receipt_sha256': contract.sha(run / 'selection_receipt.json'),
                  'preflight_sha256': contract.sha(prepared['receipt_path']),
                  'unadapted_path': 'Same fresh pinned base with adapters disabled; validated against pretraining validation predictions before test access',
                  'selection_rule': selection['selection_rule'], 'test_used_for_selection': False,
                  'unadapted': {'metrics': unadapted, 'unweighted_ce': unadapted_ce},
                  'adapted': {'metrics': adapted, 'unweighted_ce': adapted_ce},
                  'adapted_minus_unadapted_macro_f1': adapted['macro_f1'] - unadapted['macro_f1'],
                  'label_provenance': 'AI-assisted labels with human review; not independent human gold',
                  'independent_human_evaluation': False,
                  'test_groups': len({r['group_id'] for r in rows}),
                  'limits': 'Passages are clustered by review/release; no tuning or checkpoint reselection after this comparison. Restricted A/B/C softmax scores are not calibrated confidence.'}
        contract.save_json(output / 'comparison.json', result)
        contract.save_json(run / 'test_attempt_result.json', {'status': result['status'], 'comparison_sha256': contract.sha(output / 'comparison.json')})
        prepared.clear()
        del model, frozen, adapter
        gc.collect(); torch.cuda.empty_cache()
        return result
    except Exception as error:
        contract.save_json(run / 'test_attempt_result.json', {'status': 'failed_or_incomplete_test_attempt',
                           'error_type': type(error).__name__, 'error': str(error), 'automatic_retry_allowed': False})
        raise
