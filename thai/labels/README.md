# Thai labels (C3–C7)

Files here are produced by `python thai/build_label_pool.py` (after the retrieval notebook has
written `thai/results/topk_n10000_s42.csv`):

- `label_sheet.csv`: the blind sheet to label. It does not say which system found each sentence.
- `label_sheet_annotator2.csv`: a 60-pair subset for the second annotator (C5).
- `pool_key.csv`: the hidden key (system, rank, score). **Annotators should not open it.**

## How to label

1. Open `label_sheet.csv` in Google Sheets (File → Import) or Excel.
2. For each row, read `query_th` and `text` and fill `label` with **0, 1 or 2**, following
   `thai/annotation_guidelines.md`:
   0 = not relevant, 1 = partly satisfies the query, 2 = clearly satisfies it.
   `note` is optional free text.
3. **Do not reorder, sort, delete or add rows, and do not edit `pair_id` or `text`.** Filtering
   to view one query at a time is fine. Rows are already grouped by query.
4. Save or download as **CSV UTF-8** with the same filename, so the Thai text survives.
5. Score with `python thai/score_labels.py`. To score a pilot subset before everything is
   labelled, use `python thai/score_labels.py --partial` (it only scores queries whose rows
   are all labelled).

The second annotator fills `label_sheet_annotator2.csv` the same way, without seeing the first
annotator's labels. `score_labels.py` reports agreement (exact %, Cohen's kappa) when that file
has labels.

`build_label_pool.py` will not overwrite a sheet that already has labels unless you pass `--force`.
