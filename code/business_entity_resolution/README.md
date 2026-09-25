# Business Entity Resolution — ML Challenge 2026

Matches every Source-1 business record to all Source-2/Source-3 records of the same real-world
business (macro F0.5 per S1, singletons included). Pure CPU pipeline (LightGBM + engineered
features), with an optional small cross-encoder. Uses only the provided data.

```
prep ─► blocking ─► stage-0 ranker ─► features + context ─► stage A (LightGBM, 3-fold OOF)
     ─► collective features + orphan model ─► [cross-encoder] ─► stage B (LightGBM)
     ─► calibration ─► many-to-one assignment ─► expected-F0.5 decision ─► output/*.tsv
```

## 1. Setup (≈ 3–5 min)

```bash
cd code/business_entity_resolution
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt     # exact pins; full environment freeze: requirements.lock.txt
# optional cross-encoder: uv pip install -r requirements-neural.txt
python -m pytest src/tests -q          # < 1 min, pure functions, no data needed
```

The data is expected in `../../dataset/{train,test}` (the challenge layout); caches go to
`../../work/`, outputs to `../../output/`. Change them in `configs/default.yaml` (`paths`).

## 2. Run

**First, the dev slice (~5 % of the data, 15–30 min end-to-end)** — catches problems before the
long run and gives real per-stage timings to extrapolate from:

```bash
python -m src.run_pipeline dev-slice                 # builds ../../work/dev_data (≈ 2 min)
./run_all.sh configs/dev.yaml
cat ../../work/dev/reports/summary.md
```

**Full run:**

```bash
./run_all.sh                                          # keeps the Mac awake via caffeinate
python3 ../../utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test --check-ids
```

Every stage writes a `_DONE` marker and is skipped when re-run, so an interrupted run resumes.
`work/reports/summary.md` collects timings, blocking recall, OOF macro F0.5 (overall / per
country / singletons), feature importance and test prediction stats — paste it back for tuning.

### Measured time on an Apple M5 MacBook Air (10 cores, 16 GB RAM)

Full data, plugged in, lid open (a closed lid sleeps the Mac and pauses the run). Every stage
logs its wall time and peak RAM to `work/reports/timings.jsonl`.

| Step | Command | Time |
|---|---|---|
| Prep train + test | `prep --split train/test` | ≈ 25 min |
| Blocking train (10.3 M queries × 2.2 M S1, + stage-0 ranker) | `block --split train` | ≈ 13.9 h |
| Blocking test (10.0 M queries × 1.7 M S1) | `block --split test` | ≈ 4.8 h |
| Features + context train + test (124 M + 120 M pairs) | `features --split train/test` | ≈ 40 min |
| Stage A: 3 folds + OOF + test predictions | `train-a`, `predict-a --split test` | ≈ 55 min |
| Collective + orphan model (train + test) | `collective --split train/test` | ≈ 25 min |
| Stage B + decision tuning + test predictions | `train-b`, `predict-b`, `decide` | ≈ 40 min |
| Test decision + TSVs + validator | `write` | ≈ 5 min |
| **Full run** | `./run_all.sh` | **≈ 21 h** (blocking ≈ 18.7 h) |
| Re-run after a model/feature change (blocking cached) | `clean --from features` then `./run_all.sh` | ≈ 3 h |
| Re-run after a decision-rule change | `decide`, `write` | ≈ 10 min |
| Optional cross-encoder on MPS | `-o neural.enabled=true` | + 3–5 h |

An interrupted blocking run continues where it stopped with
`python -m src.run_pipeline block --split train --resume` (identical result).
Peak RAM ≈ 9–12 GB; if memory gets tight lower `blocking.query_chunk`, `model.sample.max_rows`
or `runtime.n_workers`. Disk: ~60–90 GB in `work/`.

### Useful commands

```bash
python -m src.run_pipeline <command> [--split train|test] [-o key=value ...]
python -m src.run_pipeline clean --from features     # drop features and everything after
python -m src.run_pipeline loco                       # leave-one-country-out check (France proxy)
python -m src.run_pipeline decide --tag A             # decision quality of stage A alone
python -m src.run_pipeline report                     # rebuild summary.md
```

## 3. Method (short)

* **Normalisation** (`text_norm`, `parse_address`, `translit`): NFKC, zero-width removal, accent
  folding, leetspeak repair, alias splitting ("X doing business as Y", "formerly", "|"),
  domain/handle collapse, legal-form canonicalisation (Pvt/Ltd/LLC/SARL/SAS…), rule-based
  transliteration of 9 Indic scripts plus a native→Latin lexicon mined from training matches,
  address parsing with labelled house numbers (H.No / Door No / Plot No), ranges, bis/ter,
  zero-padding, units/PO boxes, street, city, state (data-derived gazetteer + mined synonyms).
* **Blocking** (`blocking`, `prerank`), per country: char-3-gram name TF-IDF, address TF-IDF,
  joint name+address TF-IDF (sparse top-k), exact name / house+street keys, reverse S1→S2/S3
  queries; the union is cut to the top-M by a stage-0 LightGBM ranker. That final set is exactly
  what the models score and what `candidate_pairs.tsv` contains.
* **Features** (`features`, `context_features`, `collective`): ~150 per pair — fuzzy/IDF name
  similarity, a house-number suite designed around decoys (shifted numbers) vs typos (digit
  edits), address/street/city/state/unit, name-family and shared-address ambiguity, competition
  among candidates, label-free cluster consensus (do the other records of this S1 agree on its
  house number?), and stage-A probability aggregates + a record-level "matches anything" model.
* **Models**: LightGBM stage A and stage B, GroupKFold by S1, inverse-probability-weighted
  negative sampling; isotonic calibration on OOF.
* **Decision** (`decide`): each S2/S3 record keeps only its best S1 (many-to-one), an S1-level
  model estimates P(singleton), and the rule (threshold+margin or per-S1 expected-F0.5 subset)
  is tuned on OOF macro F0.5 over all 2.2 M training S1.
* **Unseen country**: no country feature (country only partitions blocking), NaN instead of
  mismatch for unmapped gazetteer values, lexicons mined per split from exact-key pseudo-pairs,
  leave-one-country-out validation.

Fair play: no external data, APIs, geocoding or pretrained entity resources; only small generic
seed lists (street types, legal forms, US/Indian state abbreviations) in `src/seeds.py`.
Models: LightGBM (MIT); optional cross-encoder `paraphrase-multilingual-MiniLM-L12-v2`
(Apache-2.0, 118 M parameters).

## 4. Layout

```
configs/default.yaml   all settings (full data)      configs/dev.yaml   dev-slice overrides
run_all.sh             end-to-end driver             requirements*.txt  dependencies
src/run_pipeline.py    CLI                           src/prep.py        stage 1
src/blocking.py        stage 2 (+ src/prerank.py)    src/features.py    stage 3
src/context_features.py  stage 3b                    src/train_gbdt.py  stages A / B
src/collective.py      stage-B inputs                src/decide.py      calibration + rule
src/write_outputs.py   TSVs + validator              src/evaluate.py    exact scorer
src/lexicon_mining.py  mined lexicons                src/loco.py        country transfer check
src/neural/            optional cross-encoder        src/tests/         unit tests
```
