# MPIXEL

This repository contains code for **MPIXEL** introduced in _Scaling Multilingual Pretraining of Pixel Language Models_. MPIXEL is a family of pixel language models pretrained on up to sixteen visually and linguistically diverse languages spanning ten writing systems, at two model sizes (86M and 307M parameters). Starting from the four-language setup of [PIXEL-M4](https://github.com/ilkerkesen/pixel-m4), we scale multilingual pretraining along three axes while keeping per-language compute fixed: more languages within an already seen script, more scripts, and larger model capacity. For more details about MPIXEL, please have a look at [our paper](https://arxiv.org/abs/2610.29681).

## Models

We release two main models, `mpixel-base` and `mpixel-large`, together with four 86M-parameter control models used in our experiments.

| Model | Parameters | Languages | Pretraining languages |
|---|---|---|---|
| **[mpixel-base](https://huggingface.co/Team-PIXEL/mpixel-base)** | 86M | 16 | ARB, BEN, CMN, DAN, ELL, ENG, FAS, FIN, HEB, HIN, KOR, SPA, THA, TUR, UKR, VIE |
| **[mpixel-large](https://huggingface.co/Team-PIXEL/mpixel-large)** | 307M | 16 | ARB, BEN, CMN, DAN, ELL, ENG, FAS, FIN, HEB, HIN, KOR, SPA, THA, TUR, UKR, VIE |
| [pixel-base-adaptive](https://huggingface.co/Team-PIXEL/pixel-base-adaptive) | 86M | 1 | ENG |
| [mpixel-4s](https://huggingface.co/Team-PIXEL/mpixel-4s) | 86M | 4 | CMN, ENG, HIN, UKR |
| [mpixel-8l](https://huggingface.co/Team-PIXEL/mpixel-8l) | 86M | 8 | CMN, DAN, ENG, FIN, HIN, SPA, TUR, UKR |
| [mpixel-8s](https://huggingface.co/Team-PIXEL/mpixel-8s) | 86M | 8 | ARB, CMN, ELL, ENG, HIN, KOR, THA, UKR |

## Model Details

MPIXEL models are based on the [PIXEL](https://github.com/xplip/pixel) architecture. The base models have 86M parameters, and `mpixel-large` scales the encoder up to 24 layers, hidden size 1024 and 16 attention heads (307M parameters).

### Adaptive rendering

A single rendering strategy does not fit all writing systems, so we implement an **adaptive** rendering strategy that also supports right-to-left scripts:

- **By default**, text is rendered as grapheme-aware bigrams, so diacritics and combining marks always stay with their base characters.
- **Chinese, Japanese and Korean text** is rendered one character per patch, because most bigrams of wide characters would otherwise span multiple patches.
- **Arabic and Hebrew scripts** are rendered word by word in logical order, so ligatures are not severed and visual-order rendering does not distort word structure.
## Setup

This codebase is a new implementation built on [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) and [Transformers](https://github.com/huggingface/transformers). MPIXEL renders text with the Noto Sans fonts from a fixed commit of [google/fonts](https://github.com/google/fonts), plus Noto Serif Tibetan for Tibetan. You can find the font list in `mpixel/renderer/fonts.py`.

You can set up this codebase as follows to get started with using MPIXEL:

<details>
  <summary><i>Show Instructions</i></summary>
&nbsp;

1. Clone repo
```
git clone https://github.com/ilkerkesen/mpixel.git
cd mpixel
```

2. Create a clone of the original environment using exported environment file,
```
conda env create -f environment.yml
conda activate mpixel
pip install -r requirements.txt
pip install -e .
```

Note that this environment only supports systems with CUDA. If you want to use MPIXEL on a system without CUDA, you need to install non-CUDA PyTorch packages with the same version numbers.

3. Verify Installation on Finnish Dependency Parsing
```bash
# Create a folder in which we keep the data
mkdir -p data
# Get and extract the UD data for parsing and POS tagging
wget -c https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-6149/ud-treebanks-v2.18.tgz
tar xvf ud-treebanks-v2.18.tgz -C data

# treebank, directories and hyperparameters
TREEBANK="UD_Finnish-FTB"
DATA_DIR="./data/ud-treebanks-v2.18/${TREEBANK}"
SEED=0
LR=1e-5

python mpixel/experiments/udp.py fit \
    --model.model_id_or_path=Team-PIXEL/mpixel-base \
    --model.learning_rate=${LR} \
    --model.warmup_steps=100 \
    --model.dropout=0.1 \
    --data.data_dir=${DATA_DIR} \
    --seed_everything=${SEED} \
    --trainer.logger.init_args.save_dir=./debug/mpixel/udp/${TREEBANK}/${LR}--${SEED}
```

If everything is configured correctly, you should expect to see results similar to the following:

| Model | Treebank | Test LAS |
|---|---|---|
| `mpixel-base` | `UD_Finnish-FTB` | 90.6 ± 0.1 |

This score is the average over the three runs with the best validation LAS.

Note that we strongly recommend a complete finetuning for verifying the installation to have a reproducible environment in better conditions.
The results may vary from one GPU architecture to another. The above results are specifically produced on NVIDIA H100 machines.
</details>

## Usage

After the setup, you can load a pretrained model and get its outputs as follows:

```python
from transformers import AutoModel
import mpixel.models.auto_register  # registers MPIXEL with the Auto classes
from mpixel.processing import MPIXELProcessor

processor = MPIXELProcessor.from_pretrained("Team-PIXEL/mpixel-base")
model = AutoModel.from_pretrained("Team-PIXEL/mpixel-base")

inputs = processor(["Hello world!", "Merhaba dünya!"])
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

We provide Gradio demos for text rendering and masked autoencoding in [scripts/demo](scripts/demo).

## Pretraining MPIXEL

We provide instructions for pretraining MPIXEL in [PRETRAINING.md](.github/PRETRAINING.md).

You can find our pretrained MPIXEL models at [https://huggingface.co/Team-PIXEL](https://huggingface.co/Team-PIXEL).

## Finetuning MPIXEL

We provide instructions for finetuning MPIXEL in [FINETUNING.md](.github/FINETUNING.md).

We also provide Hugging Face Trainer examples for NER and SIB-200 in [scripts/hf-examples](scripts/hf-examples).

## Reproducibility

Finetuning results vary noticeably across random seeds. To account for this seed noise, we repeated the same experiment eight times using different seeds (list of seeds: 0,1,2,3,4,5,6,7).
We also observed varying results for different versions of PyTorch, CUDA, HuggingFace Transformers and Pango/Cairo packages.
So, to reproduce results reported in the paper, we strongly recommend setting up the environment using the `environment.yml` and `requirements.txt` files.

## Citation & Contact

```bibtex
@misc{kesen2026scalingmultilingualpretrainingpixel,
      title={Scaling Multilingual Pretraining of Pixel Language Models}, 
      author={Ilker Kesen and Jenny Kunz and Kasra Malihi and Desmond Elliott},
      year={2026},
      eprint={2610.29681},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.29681}, 
}
```

Feel free to open an issue here or send an email to ask questions about this work or report problems with the code! We emphasize that this is experimental research code.

**Contact:**
Ilker Kesen (name.surname--at--hu-berlin.de)

## License

The code in this repository is released under the [Apache-2.0 License](LICENSE).
