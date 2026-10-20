"""Music Review Insight Finder: collection, annotation, splits, and evaluation.

Authored project code. Nothing runs on import. Python 3.9 compatible.
Network access occurs only when collect_reviews() is called without a cached page.
"""
import csv
import hashlib
import html
import io
import json
import random
import re
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

LABELS = ("specific_musical_observation", "general_opinion", "background_other")
LABEL_CODES = dict(zip(LABELS, ("A", "B", "C")))
API_URL = "https://critiquebrainz.org/ws/1/review/"
REVIEW_FIELDS = ["review_id", "entity_id", "entity_type", "language", "text_raw",
                 "rating", "license_id", "license_name", "license_url", "author",
                 "source", "source_url", "review_url", "published_on", "collected_at"]
PASSAGE_FIELDS = ["passage_id", "review_id", "entity_id", "group_id", "text",
                  "word_count", "license_id", "license_name", "license_url", "author",
                  "source_url", "review_url", "collected_at"]
ANNOTATION_FIELDS = PASSAGE_FIELDS + ["label", "status", "notes"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    """Stable SHA-256 fingerprint for structured objects or UTF-8 text."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_once(path, content):
    """Create new artifacts; allow identical reruns, refuse changed overwrites."""
    path = Path(path)
    data = content.encode("utf-8") if isinstance(content, str) else content
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return path
        raise FileExistsError("Refusing to replace existing work: %s. Use a new session folder." % path)
    with path.open("xb") as handle:
        handle.write(data)
    return path


def save_json(path, value):
    return write_once(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def save_csv(path, rows, fields):
    return write_once(path, csv_bytes(rows, fields))


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_review(record, fetched_at):
    """Support both nested documented licenses and flattened current responses.

    Preserve public attribution, not complete user profiles or API credentials.
    """
    license_info = record.get("license") or {}
    if not isinstance(license_info, dict):
        license_info = {"id": str(license_info)}
    user = record.get("user") or {}
    author = user.get("display_name", user.get("name", "")) if isinstance(user, dict) else str(user)
    rid = str(record.get("id") or "")
    return dict(zip(REVIEW_FIELDS, [
        rid, str(record.get("entity_id") or ""), str(record.get("entity_type") or ""),
        str(record.get("language") or ""), str(record.get("text") or ""),
        record.get("rating") if record.get("rating") is not None else "",
        record.get("license_id") or license_info.get("id", ""),
        license_info.get("full_name") or record.get("full_name", ""),
        license_info.get("info_url") or record.get("info_url", ""), author,
        record.get("source") or "", record.get("source_url") or "",
        "https://critiquebrainz.org/review/" + rid,
        record.get("published_on") or record.get("created") or "", fetched_at]))


def collect_reviews(run_dir, max_reviews=500, max_pages=30, page_size=50,
                    pause_seconds=1.0, session=None, sleep_fn=time.sleep):
    """Bounded, cached GET pagination. Existing pages are reused on a rerun.

    Defaults are a project proposal, not an instructor-mandated sample size.
    Sorting oldest-first reduces offset churn; the sample remains nonrandom.
    """
    if not 1 <= page_size <= 50 or max_reviews < 1 or max_pages < 1:
        raise ValueError("Use page_size 1..50 and positive collection limits.")
    run_dir = Path(run_dir)
    config = {"max_reviews": max_reviews, "max_pages": max_pages,
              "page_size": page_size, "api_url": API_URL, "language": "en",
              "entity_type": "release_group", "review_type": "review",
              "sort": "published_on", "sort_order": "asc"}
    save_json(run_dir / "collection_config.json", config)
    if session is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"], respect_retry_after_header=True)
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers.update({"User-Agent": "MusicReviewInsightFinder/1.0 (ADS509 coursework)"})
    reviews, seen = [], set()
    rejected = Counter()
    for page_number in range(max_pages):
        offset = page_number * page_size
        page_path = run_dir / "raw_pages" / ("offset_%06d.json" % offset)
        if page_path.exists():
            page = json.loads(page_path.read_text(encoding="utf-8"))
        else:
            params = {k: config[k] for k in ("language", "entity_type", "review_type", "sort", "sort_order")}
            params.update(limit=page_size, offset=offset)
            response = session.get(API_URL, params=params, timeout=(10, 45))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("reviews"), list):
                raise ValueError("API response has no review list; cached work is preserved.")
            timestamp = utc_now()
            valid_records = [r for r in payload["reviews"] if isinstance(r, dict)]
            page = {"offset": offset, "params": params, "fetched_at": timestamp,
                    "reported_count": payload.get("count"), "received": len(payload["reviews"]),
                    "records": [normalized_review(r, timestamp) for r in valid_records
                                if not r.get("is_draft") and not r.get("is_hidden")]}
            save_json(page_path, page)
            sleep_fn(pause_seconds)
        for review in page["records"]:
            rid = review["review_id"]
            if not rid or rid in seen:
                rejected["missing_or_duplicate_id"] += 1
                continue
            seen.add(rid)
            if review["language"] != "en" or review["entity_type"] != "release_group":
                rejected["unexpected_language_or_entity"] += 1
                continue
            if not review["text_raw"].strip():
                rejected["empty_text"] += 1
                continue
            reviews.append(review)
            if len(reviews) >= max_reviews:
                break
        if len(reviews) >= max_reviews or page["received"] < page_size:
            break
    if not reviews:
        raise ValueError("No usable reviews collected. Inspect cached pages before changing scope.")
    save_csv(run_dir / "reviews.csv", reviews, REVIEW_FIELDS)
    save_json(run_dir / "collection_audit.json", {"kept": len(reviews), "rejected": dict(rejected),
        "sampling": "Oldest-first bounded API sample; not representative of all music listeners.",
        "stopped_at_page_limit": page_number + 1 == max_pages and len(reviews) < max_reviews})
    return reviews


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.suppressed = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.suppressed += 1
        if tag in ("p", "br", "div", "li"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.suppressed = max(0, self.suppressed - 1)
        if tag in ("p", "div", "li"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.suppressed:
            self.parts.append(data)


def clean_text(text):
    """Remove HTML markup, preserve case/negation/punctuation and raw originals."""
    parser = _TextExtractor()
    parser.feed(str(text))
    return re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts))).strip()


def make_passages(reviews, max_words=100, min_words=3):
    """Approximate sentence boundaries; long sentences become word chunks.

    This is deliberately simple, and can split abbreviations incorrectly.
    Exact case-insensitive duplicate passages are removed before sampling/splits.
    """
    if max_words < min_words or min_words < 1:
        raise ValueError("Invalid passage length limits.")
    passages, seen_text, audit = [], set(), Counter()
    for review in reviews:
        cleaned = clean_text(review["text_raw"])
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
            words = sentence.split()
            for start in range(0, len(words), max_words):
                chunk = " ".join(words[start:start + max_words])
                if len(chunk.split()) < min_words:
                    audit["under_minimum_words"] += 1
                    continue
                key = digest(chunk.casefold())
                if key in seen_text:
                    audit["duplicate_passage"] += 1
                    continue
                seen_text.add(key)
                passage = {k: review.get(k, "") for k in PASSAGE_FIELDS}
                passage.update(passage_id=digest(review["review_id"] + ":" + key)[:24],
                    text=chunk, word_count=len(chunk.split()),
                    group_id="entity:" + review["entity_id"] if review["entity_id"]
                             else "review:" + review["review_id"])
                passages.append(passage)
    return passages, dict(audit)


def annotation_sample(passages, limit=300, per_review=3, seed=509):
    """Seeded sampling with a review cap; no model-generated gold labels."""
    if limit < 1 or per_review < 1:
        raise ValueError("Annotation limits must be positive.")
    pool = list(passages)
    random.Random(seed).shuffle(pool)
    chosen, counts = [], Counter()
    for row in pool:
        if counts[row["review_id"]] >= per_review:
            continue
        chosen.append(dict(row, label="", status="todo", notes=""))
        counts[row["review_id"]] += 1
        if len(chosen) >= limit:
            break
    return chosen


def load_annotations(path, reference):
    """Validate an edited CSV against frozen text/IDs; require deliberate status."""
    rows = load_csv(path)
    expected = {str(r["passage_id"]): r for r in reference}
    seen, accepted, todo, skipped = set(), [], 0, 0
    for row in rows:
        rid = row.get("passage_id", "")
        if rid not in expected or rid in seen:
            raise ValueError("Unknown or repeated passage_id: " + rid)
        seen.add(rid)
        for field in PASSAGE_FIELDS:
            if str(row.get(field, "")) != str(expected[rid].get(field, "")):
                raise ValueError("Changed source field %s in %s; edit only label/status/notes." % (field, rid))
        status, label = row.get("status", "").strip(), row.get("label", "").strip()
        if status == "todo":
            todo += 1
        elif status == "skip":
            if not row.get("notes", "").strip():
                raise ValueError("A skipped passage needs a reason in notes: " + rid)
            skipped += 1
        elif status == "done" and label in LABELS:
            accepted.append(dict(row, label=label))
        else:
            raise ValueError("Use status todo/done/skip and an allowed label for done rows: " + rid)
    if seen != set(expected):
        raise ValueError("Annotation rows were added/deleted; keep rows and mark exclusions skip.")
    if todo:
        raise ValueError("Human-label checkpoint: %d passages still todo. Fill labels/status first." % todo)
    if not accepted:
        raise ValueError("No labeled passages remain after exclusions.")
    return accepted, {"accepted": len(accepted), "skipped": skipped}


def check_splits(parts):
    """No entity group, review, passage ID, or exact normalized text crosses splits."""
    names = list(parts)
    for i, a in enumerate(names):
        if not parts[a]:
            raise ValueError("Empty split: " + a)
        for b in names[i + 1:]:
            for field in ("group_id", "review_id", "passage_id"):
                if {r[field] for r in parts[a]} & {r[field] for r in parts[b]}:
                    raise ValueError("Leakage between %s/%s in %s" % (a, b, field))
            if {r["text"].casefold() for r in parts[a]} & {r["text"].casefold() for r in parts[b]}:
                raise ValueError("Duplicate text crosses splits.")


def grouped_splits(rows, seed=509, min_class_examples=10, attempts=300):
    """Choose approximate 60/20/20 entity-grouped splits using labels, never scores.

    Coverage thresholds are workflow safeguards, not course requirements.
    No labels are changed, generated, duplicated, or silently dropped.
    """
    from sklearn.model_selection import GroupShuffleSplit
    import numpy as np
    counts = Counter(r["label"] for r in rows)
    if any(counts[label] < min_class_examples for label in LABELS):
        raise ValueError("Need at least %d real labels per class for this exploratory workflow; counts=%s. "
                         "Label additional examples; do not invent labels." % (min_class_examples, dict(counts)))
    groups = np.array([r["group_id"] for r in rows])
    if len(set(groups)) < 10:
        raise ValueError("Need more independent review/entity groups (workflow default: 10).")
    indices = np.arange(len(rows))
    best, best_score = None, float("inf")
    for k in range(attempts):
        rest, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed + k).split(indices, groups=groups))
        train_rel, val_rel = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 10000 + k).split(rest, groups=groups[rest]))
        parts = {"train": [rows[i] for i in rest[train_rel]],
                 "validation": [rows[i] for i in rest[val_rel]], "test": [rows[i] for i in test]}
        if any(any(Counter(r["label"] for r in part)[label] < 2 for label in LABELS) for part in parts.values()):
            continue
        score = sum(abs(len(parts[name]) / len(rows) - target) for name, target in [("train", .6), ("validation", .2), ("test", .2)])
        for part in parts.values():
            pc = Counter(r["label"] for r in part)
            score += sum(abs(pc[label] / len(part) - counts[label] / len(rows)) for label in LABELS)
        if score < best_score:
            best, best_score = parts, score
    if best is None:
        raise ValueError("No grouped split covers all classes. Collect/label more independent reviews, "
                         "especially rare classes; do not switch to leaking random passage splits.")
    check_splits(best)
    return best


def freeze_splits(run_dir, rows, seed=509):
    folder = Path(run_dir) / "splits"
    fingerprint = digest(sorted(rows, key=lambda r: r["passage_id"]))
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["annotation_sha256"] != fingerprint or manifest["seed"] != seed:
            raise ValueError("Labels/settings changed after split freeze. Use a new session folder.")
        parts = {name: load_csv(folder / (name + ".csv")) for name in ("train", "validation", "test")}
        for name in parts:
            if digest(parts[name]) != manifest["split_sha256"][name]:
                raise ValueError("Frozen split was edited: " + name)
        check_splits(parts)
        return parts
    parts = grouped_splits(rows, seed=seed)
    for name, part in parts.items():
        save_csv(folder / (name + ".csv"), part, ANNOTATION_FIELDS)
    # Hash the CSV-loaded representation, so integer word counts round-trip safely.
    saved = {name: load_csv(folder / (name + ".csv")) for name in parts}
    save_json(manifest_path, {"seed": seed, "annotation_sha256": fingerprint,
        "split_sha256": {name: digest(part) for name, part in saved.items()},
        "rows": {name: len(part) for name, part in saved.items()},
        "grouping": "entity_id, falling back to review_id", "target_proportions": [0.6, 0.2, 0.2],
        "labels": list(LABELS), "created_at": utc_now()})
    return saved


MUSIC_TERMS = set("vocal vocals voice harmony harmonies melody melodic guitar bass drum drums "
    "rhythm rhythmic brass instrumentation arrangement production mixing mix mastering "
    "chorus verse piano synth synthesizer percussion tempo saxophone violin flute trumpet".split())
OPINION_TERMS = set("love loved like liked hate hated great good amazing favorite favourite "
    "boring bad awful excellent terrible beautiful best worst awesome incredible".split())


def keyword_prediction(text):
    words = set(re.findall(r"[a-z]+", text.casefold()))
    if words & MUSIC_TERMS:
        return LABELS[0]
    if words & OPINION_TERMS:
        return LABELS[1]
    return LABELS[2]


def majority_label(train):
    counts = Counter(r["label"] for r in train)
    return max(LABELS, key=lambda label: counts[label])


def evaluate_predictions(rows, predictions):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    if len(rows) != len(predictions) or not rows:
        raise ValueError("Need exactly one prediction per nonempty evaluation row.")
    if any(p not in LABELS for p in predictions):
        raise ValueError("Unexpected prediction label.")
    gold = [r["label"] for r in rows]
    return {"n": len(rows), "accuracy": float(accuracy_score(gold, predictions)),
        "macro_f1": float(f1_score(gold, predictions, labels=list(LABELS), average="macro", zero_division=0)),
        "classification_report": classification_report(gold, predictions, labels=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(gold, predictions, labels=list(LABELS)).tolist(), "label_order": list(LABELS)}


def make_training_bundle(run_dir, source_dir):
    """Create a local archive; never upload it or publish anything."""
    run_dir, source_dir = Path(run_dir), Path(source_dir)
    members = {"splits/" + name: run_dir / "splits" / name
               for name in ("train.csv", "validation.csv", "test.csv", "manifest.json")}
    members["music_review_pipeline.py"] = source_dir / "music_review_pipeline.py"
    members["music_review_training.py"] = source_dir / "music_review_training.py"
    members["LABELING_GUIDE.md"] = source_dir / "LABELING_GUIDE.md"
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in members.items()}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in members.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("bundle_sha256.json", date_time=(2026, 10, 15, 0, 0, 0))
        z.writestr(info, json.dumps(hashes, indent=2))
    return write_once(run_dir / "training_bundle.zip", archive.getvalue())
