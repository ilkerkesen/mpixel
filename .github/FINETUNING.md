## Finetuning MPIXEL

Here we provide instructions for finetuning MPIXEL. You can use these to reproduce our results or train your own MPIXEL-based model on a different dataset. If your dataset is currently not supported and you don't know how to get started (because the renderer is missing a feature, a classification head is missing, etc.) feel free to open an issue about it.

Requirements:
- An environment set up as described in our main [README.md](../README.md)

### Downloading data
<details>
  <summary><i>Show Instructions</i></summary>
&nbsp;

#### Data

Use the provided scripts to download the data.

```bash
# Create a folder in which we keep the data
mkdir -p data/downstream

# UD data for parsing
wget -qO- https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/handle/11234/1-6149/ud-treebanks-v2.18.tgz | tar xvz -C data/downstream

# SIB-200 data (all languages)
python mpixel/datamodules/sib200.py
mv data/sib-200 data/downstream/sib-200

# Indic NER data (Naamapadam)
python scripts/downstream/download_naamapadam.py --output-dir ./data/downstream/ner

# KLUE/NER data
python scripts/downstream/download_klue_ner.py --output-dir ./data/downstream/ner

# Ukrainian NER data (NER-UK 2.0)
python scripts/downstream/download_ner_uk.py --output-dir ./data/downstream/ner
```

For Universal NER, clone the repos and run the preprocessing script:

```bash
mkdir -p data/downstream/uner-raw

git clone https://github.com/UniversalNER/UNER_English-EWT ./data/downstream/uner-raw/UNER_English-EWT
git clone https://github.com/UniversalNER/UNER_Serbian-SET ./data/downstream/uner-raw/UNER_Serbian-SET
git clone https://github.com/UniversalNER/UNER_Chinese-GSD ./data/downstream/uner-raw/UNER_Chinese-GSD
git clone https://github.com/UniversalNER/UNER_Danish-DDT ./data/downstream/uner-raw/UNER_Danish-DDT
git clone https://github.com/UniversalNER/UNER_Hebrew-HTB ./data/downstream/uner-raw/UNER_Hebrew-HTB
git clone https://github.com/UniversalNER/UNER_Greek_GDT ./data/downstream/uner-raw/UNER_Greek_GDT

python scripts/downstream/preprocess_universal_ner.py --input-dir ./data/downstream/uner-raw --output-dir ./data/downstream/ner
```

The script writes one folder for each treebank: `universal-ner-english-ewt`, `universal-ner-serbian-set`, `universal-ner-chinese-gsd`, `universal-ner-danish-ddt`, `universal-ner-hebrew-htb` and `universal-ner-greek-gdt`.

</details>

### Training

We finetune MPIXEL with PyTorch Lightning. Each task has one script in [`mpixel/experiments`](../mpixel/experiments), and you start it with the `fit` command. Some important points:
- Instead of a tokenizer, we use a renderer. The default is adaptive rendering. Use `--data.renderer_type` to select another strategy. See [RENDERING.md](RENDERING.md) for details.
- Text is rendered into a single strip of patches. `--data.max_num_patches` sets the width of this strip.
- `--model.model_id_or_path` accepts a Hugging Face Hub ID, a local Hugging Face model folder, or a Lightning `.ckpt` file.
- Each script sets its own defaults for precision (`bf16-mixed`), early stopping and checkpointing. Add `--print_config` to see all settings.
- After training, each script evaluates the best checkpoint and writes the scores to `all_results.json` in the log folder.

We also provide Hugging Face Trainer examples for NER and SIB-200 in [scripts/hf-examples](../scripts/hf-examples).

Here are some examples for how to finetune MPIXEL:

#### SIB-200
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
MODEL="Team-PIXEL/mpixel-base"
LANGUAGE="hin_Deva"

# fixed hyperparameters for SIB-200 experiments.
MAX_NUM_PATCHES=256
BSZ=32
NUM_STEPS=15000

# sweeped hyperparameters.
SEED=0
LR=1e-5

# set this based on your compute.
NUM_WORKERS=8

OUTPUT_DIR="./logs/$(basename ${MODEL})/sib-200/${LANGUAGE}/${LR}--${SEED}"
DATA_DIR="./data/downstream"
python mpixel/experiments/text_classification.py fit \
    --model.model_id_or_path=${MODEL} \
    --model.learning_rate=${LR} \
    --model.dropout=0.1 \
    --data.language=${LANGUAGE} \
    --data.root_data_dir=${DATA_DIR} \
    --data.max_num_patches=${MAX_NUM_PATCHES} \
    --data.batch_size=${BSZ} \
    --data.num_workers=${NUM_WORKERS} \
    --trainer.max_steps=${NUM_STEPS} \
    --seed_everything=${SEED} \
    --trainer.logger.init_args.save_dir=${OUTPUT_DIR}
```
</details>


#### Dependency Parsing
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
# model and treebank
MODEL="Team-PIXEL/mpixel-base"
TREEBANK="UD_Hindi-HDTB"

# fixed hyperparameters for dependency parsing experiments.
MAX_NUM_PATCHES=512
BSZ=64
NUM_STEPS=15000

# sweeped hyperparameters.
SEED=0
LR=1e-5

# set this based on your compute.
NUM_WORKERS=8

OUTPUT_DIR="./logs/$(basename ${MODEL})/udp/${TREEBANK}/${LR}--${SEED}"
DATA_DIR="./data/downstream/ud-treebanks-v2.18/${TREEBANK}"
python mpixel/experiments/udp.py fit \
    --model.model_id_or_path=${MODEL} \
    --model.learning_rate=${LR} \
    --model.warmup_steps=100 \
    --model.dropout=0.1 \
    --data.data_dir=${DATA_DIR} \
    --data.max_num_patches=${MAX_NUM_PATCHES} \
    --data.batch_size=${BSZ} \
    --data.num_workers=${NUM_WORKERS} \
    --trainer.max_steps=${NUM_STEPS} \
    --seed_everything=${SEED} \
    --trainer.logger.init_args.save_dir=${OUTPUT_DIR}
```

</details>

#### NER
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
MODEL="Team-PIXEL/mpixel-base"
DATASET="naamapadam-hi"

# fixed hyperparameters for NER experiments.
MAX_NUM_PATCHES=256
BSZ=64
NUM_STEPS=15000

# sweeped hyperparameters.
SEED=0
LR=1e-5

# set this based on your compute.
NUM_WORKERS=8

OUTPUT_DIR="./logs/$(basename ${MODEL})/ner/${DATASET}/${LR}--${SEED}"
DATA_DIR="./data/downstream/ner"
python mpixel/experiments/ner.py fit \
    --model.model_id_or_path=${MODEL} \
    --model.learning_rate=${LR} \
    --model.warmup_steps=100 \
    --model.dropout=0.1 \
    --data.dataset=${DATASET} \
    --data.root_data_dir=${DATA_DIR} \
    --data.max_num_patches=${MAX_NUM_PATCHES} \
    --data.batch_size=${BSZ} \
    --data.num_workers=${NUM_WORKERS} \
    --trainer.max_steps=${NUM_STEPS} \
    --seed_everything=${SEED} \
    --trainer.logger.init_args.save_dir=${OUTPUT_DIR}
```

Note that the above script targets Naamapadam, the Indic NER benchmark.
To use another benchmark, set `DATASET` to its folder in `./data/downstream/ner`, for example `klue-ner`, `ner-uk` or `universal-ner-english-ewt`.
`klue-ner` has no test split, so the script reports only its validation scores.
</details>
