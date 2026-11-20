# DiagnoTech – Development Log

## 2026-11-19 – Inspection findings (before any design)

* Two datasets supplied. The XLSX (`DiagnoTech_Respiratory_Clean__1.xlsx`)
  is the richest copy: 19,584 rows, 7 columns, Arabic headers and a fully
  Arabic category column (9 categories), source name + source URL.
* The CSV (`data_chatbot_0_860u.csv`) is the same dataset with English
  headers: 19,607 rows (a superset by record_id, +23), no source columns,
  plus an unnamed flag column whose value is `needs_review` on 19,585 rows.
* All question/answer content is English in both files. No Arabic content
  exists to "recover"; Arabic answers must be generated cross-lingually.
* Composition: 19,139 Medical MCQ exam rows (97.7%) vs 445 Mayo Clinic
  patient-education rows (81 topics). 5,096 exact duplicate Q+A copies in
  the XLSX (5,131 across the merged set). One empty answer.
* 3,408 MCQ answers reference a missing option letter ("Ans. is 'c'") —
  the options are not in the data at all.

## Key decisions (with reasons)

1. **Patient index = Mayo Clinic rows only.** MCQ items are exam
   questions with missing options; presenting them as patient guidance
   would be unsafe and often meaningless. They remain in the processed
   dataset, flagged `patient_usable=false`.
2. **Embeddings: `intfloat/multilingual-e5-small`** (384-dim) — strong
   Arabic and true cross-lingual retrieval (Arabic query → English doc).
   TF-IDF backend kept as a no-download fallback; FAISS `IndexFlatIP`
   (exact cosine at this scale), NumPy store as fallback.
3. **LLM: Qwen/Qwen2.5-1.5B-Instruct locally** via transformers on CPU —
   multilingual with usable Arabic, small enough for a CPU-only machine.
   An OpenAI-compatible provider covers llama.cpp servers or a gateway.
4. **Merging rule**: XLSX is primary (source + URL), CSV contributes the
   `needs_review` flag and its 23 unique records (source unknown →
   not patient-usable).
5. **Environment fix**: the sandbox Python is externally managed
   (PEP 668), so the project uses its own `.venv`. pip's temp dir had to
   be moved off the 512 MB `/tmp` tmpfs (`TMPDIR`) after an
   out-of-space failure during the torch install.

## 2026-11-20 – Answer-quality fixes (user feedback: too many "I don't know")

User testing showed the app answering "المعلومة غير متوفرة" too often,
and sometimes grounding on the wrong topic. Measured diagnosis on the
real index (production threshold 0.45, e5 embedder):

* Retrieval *ranking* was already strong — spelling-variant evaluation
  (hamza/ة/ى variants, dialect phrasing, a typo) scored 19/19 with the
  correct disease. The failures were elsewhere:
* Questions with **no detected disease** still retrieved high-scoring
  *irrelevant* evidence, because cross-lingual e5 cosine for unrelated
  medical text sits at 0.80–0.86 — far above the 0.45 threshold.
  "ما هو الالتهاب الرئوي؟" (Pneumonia — genuinely absent from the KB)
  returned Asthma attack/COPD chunks at 0.815; "اعراض نزله البرد ايه؟"
  (Common cold — also absent) returned sinusitis chunks at 0.864; a
  symptom-only question returned sinusitis/flu chunks at 0.839.
* The strict system prompt made the small Qwen2.5-1.5B model emit the
  unavailable sentence even for on-topic evidence, and the chat service
  then discarded the answer entirely.

Fixes implemented:

1. `retrieval.py` — the Arabic question is embedded after
   `normalize_arabic` (hamza/ة/ى unified), English expansion unchanged,
   so spelling variants produce stable query vectors.
2. `retrieval.py` + `config.py` — **unanchored gate**: when no disease
   is detected in the question, the effective threshold becomes
   `max(threshold, settings.unanchored_threshold)` with the new
   `unanchored_threshold` setting (default 0.88,
   `DIAGNOTECH_UNANCHORED_THRESHOLD`). Anchored questions keep the
   normal 0.45 path with the disease boost.
3. `diseases.py` — conservative fuzzy fallback in `detect_diseases`,
   active only when exact matching finds nothing: single-word lexicon
   terms matched against punctuation-cleaned query tokens with
   Levenshtein distance ≤1 (words ≥4 chars) or ≤2 (words ≥7 chars);
   local implementation, no new dependencies. Short words (e.g. سل)
   are never fuzzy-matched.
4. `prompts.py` — rule 3 softened: the model must answer from the
   evidence whenever it concerns the same disease/topic, summarizing
   what the evidence says about the asked aspect; the exact unavailable
   sentence is reserved for evidence about a completely different
   topic. New rule 7 states that Arabic question vs English evidence,
   or wording differences, are never a reason to refuse. All other
   safety rules unchanged.
5. `chat_service.py` — retry-on-bail: if the model still emits the
   unavailable phrase while evidence exists, generation is retried once
   with the same evidence plus a firm instruction
   (`prompts.RETRY_INSTRUCTION`); only a second bail (or a generation
   failure) falls back to the previous behavior.

Verification (this machine, real index + e5):

* Test suite: **45 passed** (37 existing + 8 new: fuzzy detection ×3,
  query normalization, unanchored gate suppression, anchored control,
  retry-on-bail recovery, retry-on-bail still-unavailable).
* `scripts/evaluate_retrieval.py`: Recall@5 = 1.0, MRR = 0.9444 —
  unchanged.
* `scripts/evaluate_variants.py`: 19/19 answered, correct-disease rate
  1.0 — unchanged.
* Hard cases re-checked: pneumonia / common-cold / symptom-only
  questions now return **no local evidence** (they correctly fall
  through to external research / the unavailable message instead of
  grounding on the wrong disease), while "هل الربو معدي؟",
  "ايه علاج الربو؟" and "هل السل معدي؟" still return their disease's
  evidence (top scores 0.98–1.01).

## Status

* [x] Dataset inspection (both files, independently verified)
* [x] Modular source: config, arabic, domain, ingestion, chunking,
      embeddings, vector_store, retrieval, llm, prompts, research,
      chat_service, database, api, main
* [x] Data pipeline executed: 19,607 merged → 14,475 clean records
      (445 patient-usable), 1,399 patient chunks
* [x] Arabic RTL frontend, service/backup/eval scripts, systemd unit
* [x] Test suite: 45 tests passing (unit + integration + API)
* [x] Answer-quality round (2026-11-20): query normalization before
      embedding, unanchored-evidence gate (0.88), fuzzy disease
      detection for typos, softened refusal rule, retry-on-bail —
      hard cases verified on the real index (see entry above)
* [x] Live server smoke test via scripts/service.py: health, greeting
      chat, emergency escalation chat, RTL frontend serving, clean
      start/stop — all verified against the real uvicorn server
* [x] Backup script verified (database + stats copied)
* [x] FAISS index built: 1,399 chunks from the 445 patient records
      (sentence-transformers e5-small, FAISS backend)
* [x] Retrieval evaluation on 18 Arabic questions (redesigned to target
      diseases the KB actually covers — it has no Pneumonia/Common cold
      records): baseline Recall@5 = 0.25, MRR = 0.17. Root cause: the
      embedding model matched the question *aspect* but not the Arabic
      disease name. Fix: Arabic->English disease + aspect anchoring
      (diseases.py) — query expansion, a +0.15 metadata boost for chunks
      of the detected disease, ال-stripping for phrase matching, and
      acute/chronic sinusitis disambiguation. After the fix:
      **Recall@5 = 1.0, MRR = 0.94** (16 of 18 at rank 1)
* [ ] Local LLM download + end-to-end grounded chat verification —
      Qwen2.5-1.5B-Instruct download in progress

- 2026-11-20 (Windows service fix): scripts/service.py used the POSIX os.kill(pid, 0) liveness check, which raises OSError [WinError 11] on Windows and crashed stop/status on the user's machine; replaced with a cross-platform _pid_alive() (OpenProcess on Windows). Also fixed the venv interpreter lookup to check .venv\Scripts\python.exe before .venv/bin/python. The code-update files themselves had already extracted cleanly; only the service control script was affected.

- 2026-11-20 (submission prep): Prepared the project for submission via a
  GitHub link. (1) Frontend refresh (frontend/index.html, style.css,
  app.js; no API changes): modernized RTL layout and branding, message
  bubbles with per-state badges, sources rendered as cards (disease,
  source, category, match score), conversation sidebar with model info,
  suggestion chips, and a thinking indicator that shows elapsed seconds
  while an answer is generated (CPU generation is slow and the old
  static "searching" text left users unsure whether the app had hung).
  The composer is also disabled while an answer is pending, which
  prevents the duplicate sends seen in manual testing. Verified by
  serving the app locally: /, /static/style.css and /static/app.js all
  return the new files (HTTP 200) and /api/health reports the index
  ready. (2) README.md rewritten as submission documentation: how the
  pipeline works, architecture with a diagram, the answer states, an
  honest dataset section (445 Mayo patient records / 81 topics indexed
  by a documented safety rule; the ~14k MCQ records lack their answer
  options; no pneumonia, common cold or acute bronchitis records), the
  measured results, Windows-friendly setup, and configuration including
  the optional OpenAI-compatible provider. (3) A .gitignore (added
  earlier today) keeps models/, .venv/, .env, logs, the conversation
  database and backups out of git. Verification this round: 45 pytest
  tests pass; evaluate_retrieval.py Recall@5 = 1.0, MRR = 0.9444;
  evaluate_variants.py 19/19. Live end-to-end chat with the real Qwen
  model is still not verified on this machine (weights download
  incomplete here); it was verified by the user on his own machine,
  where the remaining problem was Windows paging-file space, fixed by
  moving the project to his D: drive.
