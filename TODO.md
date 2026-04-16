# Inertialink — Roadmap

Goal: user writes with the IMU pen → real-time soft copy generated, no OCR, no camera.

---

## Phase 1 — Character-Level Training Data

- [ ] Extend `augment_seed_data.py` logic to rotate/warp generic seed data effectively
  - [ ] a–z (26 lowercase)
  - [ ] A–Z (26 uppercase)
  - [ ] 0–9 (10 digits)
- [ ] Generate isolated character samples (~200 per character)
- [ ] Generate random short word samples to teach character transitions in continuous writing
- [ ] Rebuild dataset cache (`data/dataset_cache.pt`) from new training data

## Phase 2 — Model Retraining

- [ ] Retrain BiLSTM CTC on the full character dataset
- [ ] Remove `snapToVocab()` from `decoder_main.cpp` — it blocks any word not in the 12-word list
- [ ] Verify beam search decodes arbitrary words correctly (not just trained words)

## Phase 3 — Space & Sentence Support

- [ ] Add `' '` (space) as a trainable character — it is already in the ALPHABET at index 1
- [ ] Generate training data that includes spaces between words (short phrases)
- [ ] Decide on sentence boundary strategy:
  - Option A: idle timeout per word, concatenate predictions into a sentence buffer
  - Option B: predict space inline — model emits space when pen pauses between words

## Phase 4 — Real Hardware Data Collection

- [ ] Collect real IMU data with the ESP32 pen using `data_collector`
  - [ ] ~50–100 samples per character minimum
  - [ ] Consistent writing speed and grip
- [ ] Fine-tune the synthetic-pretrained model on real data
- [ ] Validate on held-out real samples (not used in training)

## Phase 5 — Production Polish

- [ ] Sentence buffer in `decoder_main.cpp`: accumulate word predictions, print on full stop / newline / long pause
- [ ] Export final model to ONNX
- [ ] Update `eval_model.py` to evaluate on full character set
- [ ] Test end-to-end: write a sentence → read the printed soft copy

---

**Architecture note:** No changes to BiLSTM + CTC or the IO/visualizer pipeline are needed.
The model, decoder, and hardware backend already support arbitrary character output.
The only work is the training data and sentence-level assembly.
