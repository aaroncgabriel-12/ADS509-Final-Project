"""Offline checks with invented fixtures. No API, downloads, or model training."""
import json
import tempfile
import unittest
from pathlib import Path
import music_review_pipeline as p


class Response:
    def __init__(self, rows): self.rows = rows
    def raise_for_status(self): pass
    def json(self): return {"reviews": self.rows, "count": 5}


class FakeSession:
    def __init__(self): self.calls = []
    def get(self, url, params, timeout):
        self.calls.append(dict(params))
        def row(i, **kw):
            return dict(id=str(i), entity_id="e" + str(i), entity_type="release_group",
                        language="en", text="A fictional review " + str(i), **kw)
        return Response({0: [row(1), row(1)], 2: [row(2), row(3)], 4: [row(4)]}[params["offset"]])


def fixture_rows():
    rows = []
    for group in range(40):
        for j, label in enumerate(p.LABELS):
            rows.append(dict(passage_id="p%d_%d" % (group, j), review_id="r%d" % group,
                entity_id="e%d" % group, group_id="entity:e%d" % group,
                text="Invented offline fixture group %d label %d" % (group, j), word_count=8,
                license_id="fixture", license_name="fixture", license_url="", author="fictional",
                source_url="", review_url="", collected_at="fixture", label=label, status="done", notes=""))
    return rows


class PipelineTests(unittest.TestCase):
    def test_nested_and_flat_licenses(self):
        r = p.normalized_review({"id": "x", "text": "test", "license": {"id": "L", "full_name": "License", "info_url": "URL"}, "user": {"display_name": "Author", "email": "not-retained"}}, "now")
        self.assertEqual((r["license_id"], r["author"]), ("L", "Author"))
        self.assertNotIn("email", json.dumps(r))
        self.assertEqual(p.normalized_review({"license_id": "F", "full_name": "Flat", "info_url": "U"}, "now")["license_name"], "Flat")

    def test_cached_collection_and_parameter_spelling(self):
        with tempfile.TemporaryDirectory() as folder:
            s = FakeSession()
            rows = p.collect_reviews(folder, max_reviews=5, max_pages=3, page_size=2, session=s, sleep_fn=lambda _: None)
            self.assertEqual(len(rows), 4)
            self.assertEqual([x["offset"] for x in s.calls], [0, 2, 4])
            self.assertTrue(all(x["entity_type"] == "release_group" and "entity-type" not in x for x in s.calls))
            s2 = FakeSession()
            self.assertEqual(rows, p.collect_reviews(folder, max_reviews=5, max_pages=3, page_size=2, session=s2, sleep_fn=lambda _: None))
            self.assertEqual(s2.calls, [])

    def test_html_and_exact_deduplication(self):
        self.assertEqual(p.clean_text("<p>Not bad &amp; clear.</p><script>ignore()</script>"), "Not bad & clear.")
        reviews = [dict(review_id="r1", entity_id="e1", text_raw="Clear bass and drums. Excellent record!"),
                   dict(review_id="r2", entity_id="e2", text_raw="Clear bass and drums.")]
        rows, audit = p.make_passages(reviews)
        self.assertEqual(len(rows), 1)  # Two-word opinion falls below the declared minimum.
        self.assertEqual(audit["duplicate_passage"], 1)

    def test_annotation_checkpoint_and_integrity(self):
        ref = p.annotation_sample(fixture_rows(), limit=4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.csv"
            path.write_bytes(p.csv_bytes(ref, p.ANNOTATION_FIELDS))
            with self.assertRaisesRegex(ValueError, "checkpoint"): p.load_annotations(path, ref)
            done = [dict(r, status="done", label=p.LABELS[0]) for r in ref]
            path.write_bytes(p.csv_bytes(done, p.ANNOTATION_FIELDS))
            accepted, audit = p.load_annotations(path, ref)
            self.assertEqual(len(accepted), 4)
            done[0]["text"] = "changed"
            path.write_bytes(p.csv_bytes(done, p.ANNOTATION_FIELDS))
            with self.assertRaisesRegex(ValueError, "Changed source"): p.load_annotations(path, ref)

    def test_grouped_freeze_reuse_and_changed_label_rejection(self):
        rows = fixture_rows()
        with tempfile.TemporaryDirectory() as folder:
            parts = p.freeze_splits(folder, rows)
            p.check_splits(parts)
            self.assertEqual(sum(map(len, parts.values())), len(rows))
            self.assertEqual(parts, p.freeze_splits(folder, rows))
            rows[0]["label"] = p.LABELS[1]
            with self.assertRaisesRegex(ValueError, "changed"): p.freeze_splits(folder, rows)

    def test_small_or_leaking_data_rejected(self):
        with self.assertRaises(ValueError): p.grouped_splits(fixture_rows()[:3])
        row = fixture_rows()[0]
        with self.assertRaisesRegex(ValueError, "Leakage"): p.check_splits({"train": [row], "test": [row]})

    def test_metrics_and_rule(self):
        rows = fixture_rows()[:3]
        gold = [r["label"] for r in rows]
        self.assertEqual(p.evaluate_predictions(rows, gold)["macro_f1"], 1.0)
        self.assertEqual(p.keyword_prediction("The vocals are too quiet"), p.LABELS[0])
        self.assertEqual(p.keyword_prediction("I love this record"), p.LABELS[1])
        self.assertEqual(p.keyword_prediction("Released in 2002"), p.LABELS[2])

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.txt"
            p.write_once(path, "original")
            p.write_once(path, "original")
            with self.assertRaises(FileExistsError): p.write_once(path, "replacement")
            self.assertEqual(path.read_text(), "original")


if __name__ == "__main__": unittest.main()
