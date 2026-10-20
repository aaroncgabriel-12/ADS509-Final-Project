"""CPU-only checks of masking/padding/scoring; no pretrained model or training."""
import unittest
from types import SimpleNamespace
import torch
import music_review_pipeline as p
import music_review_training as t


class Tokenizer:
    eos_token_id = 9
    def encode(self, text, **kwargs):
        if text in ("A", "B", "C"): return [{"A": 2, "B": 3, "C": 4}[text]]
        return [5] * min(len(text.split()), kwargs.get("max_length", 1000))
    def decode(self, tokens, **kwargs): return "passage " * len(tokens)
    def apply_chat_template(self, messages, **kwargs): return [1, 5, 6, 7]


class FixedModel:
    def eval(self): return self
    def __call__(self, input_ids, attention_mask):
        logits = torch.zeros(1, input_ids.shape[1], 10)
        logits[0, -1, 3] = 4  # B wins the deliberately invented next-token logits.
        return SimpleNamespace(logits=logits)


class TrainingTests(unittest.TestCase):
    def test_prompt_mask_and_answer_eos(self):
        row = t.supervised_example(Tokenizer(), "invented passage", p.LABELS[0])
        self.assertEqual(row["input_ids"], [1, 5, 6, 7, 2, 9])
        self.assertEqual(row["labels"], [-100, -100, -100, -100, 2, 9])
        with self.assertRaises(ValueError): t.supervised_example(Tokenizer(), "text", "fabricated_label")

    def test_padding_never_contributes_to_loss(self):
        rows = [t.supervised_example(Tokenizer(), "text", p.LABELS[0]),
                {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}]
        batch = t.collate_examples(rows, pad_token_id=0)
        self.assertEqual(batch["labels"][1].tolist(), [-100, 2, -100, -100, -100, -100])
        self.assertEqual(batch["attention_mask"][1].tolist(), [1, 1, 0, 0, 0, 0])

    def test_closed_set_prediction_alignment(self):
        predictions, scores = t.predict_closed_set(FixedModel(), Tokenizer(), [{"text": "fixture"}],
                                                     torch.device("cpu"), torch.float32)
        self.assertEqual(predictions, [p.LABELS[1]])
        self.assertAlmostEqual(sum(scores[0].values()), 1.0, places=6)


if __name__ == "__main__": unittest.main()
