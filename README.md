# POS Tagging on UD English EWT

Part-of-speech taggers for the Universal Dependencies English EWT treebank:

- **Phase 1:** a Hidden Markov Model (HMM) tagger with Viterbi decoding
- **Phase 2:** a BiLSTM tagger with an optional character-level CNN

## Features

- HMM-based POS tagging
- Viterbi decoding in log space
- Unknown-word shape buckets
- Token accuracy, macro F1, and unseen-word evaluation
- BiLSTM tagger (word embeddings + char-CNN) with early stopping
- Automated tests with pytest (GitHub Actions CI)

## Results (test set)

| Model | Token acc. | Macro F1 | Unknown-word acc. |
| :--- | :---: | :---: | :---: |
| HMM v2 (shape buckets) | 92.24% | 85.53% | 75.72% |
| BiLSTM (word only) | 92.32% | 86.56% | 55.15% |
| BiLSTM (word + char-CNN) | **95.33%** | **91.57%** | **81.40%** |

BiLSTM numbers are the mean over 5 seeds; see `results/dl_ablations.json`.
HMM ablations are written up in [docs/ablations.md](docs/ablations.md).

## Installation

```bash
python -m pip install -e .
python -m pip install -r requirements.txt   # extras for experiments / BiLSTM (torch, matplotlib)
```

## Usage

```python
from hmm_tagger import HMMTagger
from hmm_tagger.data import load_ewt

data = load_ewt("data")
tagger = HMMTagger().fit(data["train"])

print(tagger.predict("The movie was really good".split()))
```

## Tests

```bash
python -m pytest -v
```

Skip the slow end-to-end regression tests with:

```bash
python -m pytest -m "not slow"
```

## Experiments

```bash
python experiments/POS_tagger/ablations.py   # HMM baselines and ablations
python experiments/BiLSTM/bilstm_ablations.py   # BiLSTM comparisons
```

## Project structure

```
src/hmm_tagger/      HMM model, Viterbi decoder, data loading, metrics
src/bilstm_tagger/   BiLSTM model and training loop
tests/               pytest suite
experiments/         ablation and comparison scripts
results/             saved metrics (JSON)
figures/             generated plots
docs/                write-ups
notebooks/           Colab / Jupyter notebooks
reports/             Phase 1 and Phase 2 reports
```
