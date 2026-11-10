## Pretraining MPIXEL

Here we provide instructions for pretraining MPIXEL. You can use these to reproduce our results or train your own MPIXEL model on other languages.

Requirements:
- An environment set up as described in our main [README.md](../README.md)
- If you would like to take advantage of [Flash Attention 2](https://github.com/Dao-AILab/flash-attention), please install it. Otherwise, set `attn_implementation` to `sdpa` in `configs/pixel-base.json` or `configs/pixel-large.json`.

### Downloading data
<details>
  <summary><i>Show Instructions</i></summary>
&nbsp;

We use [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) for English and [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) for all other languages. Each language gets the same budget of about 7.5B patches.

[`download_fineweb2.py`](../scripts/data/download_fineweb2.py) streams the documents of one language from the Hugging Face Hub, splits them into sequences of at most 512 patches, and writes the sequences to compressed JSONL shards. We do not prerender the data. The data module renders the text during training.

```bash
DATA_DIR="./data/pretraining"

for LANGUAGE in arb_Arab ben_Beng cmn_Hani dan_Latn ell_Grek eng_Latn fas_Arab fin_Latn \
                heb_Hebr hin_Deva kor_Hang spa_Latn tha_Thai tur_Latn ukr_Cyrl vie_Latn; do
    python scripts/data/download_fineweb2.py ${LANGUAGE} \
        --num-train-patches=7500000000 \
        --patches-per-file=2500000 \
        --output-dir=${DATA_DIR}/${LANGUAGE}
done
```

Some important points:
- Each language needs its own folder in `DATA_DIR`, and the name of the folder must be the language code.
- The command above writes about 3000 shards for each language. Use `--patches-per-file` to change this.
- The total number of shards should be a multiple of the number of GPUs. Otherwise, each GPU reads all shards and skips most examples, which is slow.
- The total number of shards should also be at least the number of GPUs multiplied by `data.num_workers`.
- You can change the number of shards later with [`reshard.py`](../scripts/data/reshard.py). This script reads only `.jsonl` and `.jsonl.gz` files, so add `--compression=gzip` to the download command if you plan to use it.

</details>

### Training

We pretrain MPIXEL with PyTorch Lightning, using [`mpixel/experiments/pretraining.py`](../mpixel/experiments/pretraining.py). The configs in [`configs/experiments`](../configs/experiments) contain the settings of our two main models. Both configs train for 1M steps on 8 nodes with 4 GPUs each. Some important points:
- `data.batch_size` is the batch size per GPU. The effective batch size is `data.batch_size` x `trainer.devices` x `trainer.num_nodes`, which is 1024 in both configs.
- On a SLURM cluster, start the command with `srun`, and set `--nodes` and `--ntasks-per-node` to the values of `trainer.num_nodes` and `trainer.devices`.
- `--data.languages` selects the pretraining languages. Each language code must be a folder in `--data.data_dir`.
- Instead of a tokenizer, we use a renderer. The default is adaptive rendering. Use `--data.renderer_name` to select another strategy. See [RENDERING.md](RENDERING.md) for details.
- Lightning saves a checkpoint every 10,000 steps to `<save_dir>/<name>/<version>/checkpoints`. The folder `<save_dir>/<name>/<version>` also contains the `config.yaml` of the run.
- Add `--print_config` to see all settings.

#### mpixel-base
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
# This run takes about 1350 GPU hours on NVIDIA H100 GPUs.
python mpixel/experiments/pretraining.py fit \
    --config=configs/experiments/mpixel-base.yaml \
    --data.data_dir=./data/pretraining \
    --trainer.devices=4 \
    --trainer.num_nodes=8 \
    --trainer.logger.init_args.save_dir=./logs/pretraining
```
</details>

#### mpixel-large
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
# This run takes about 2500 GPU hours on NVIDIA H100 GPUs.
python mpixel/experiments/pretraining.py fit \
    --config=configs/experiments/mpixel-large.yaml \
    --data.data_dir=./data/pretraining \
    --trainer.devices=4 \
    --trainer.num_nodes=8 \
    --trainer.logger.init_args.save_dir=./logs/pretraining
```
</details>

#### Control models
<details>
  <summary><i>Show Instructions</i></summary>
&nbsp;

The control models use the settings of `mpixel-base.yaml`, with the changes below. To keep the compute per language fixed, the effective batch size grows with the number of languages. The learning rate follows the square root of the batch size.

| Model | `--data.languages` | Effective batch size | `--model.learning_rate` | `--trainer.num_nodes` (4 GPUs each) |
|---|---|---|---|---|
| `mpixel-4s` | `cmn_Hani,eng_Latn,hin_Deva,ukr_Cyrl` | 256 | `1.5e-4` | 2 |
| `mpixel-8l` | `cmn_Hani,dan_Latn,eng_Latn,fin_Latn,hin_Deva,spa_Latn,tur_Latn,ukr_Cyrl` | 512 | `2.12e-4` | 4 |
| `mpixel-8s` | `arb_Arab,cmn_Hani,ell_Grek,eng_Latn,hin_Deva,kor_Hang,tha_Thai,ukr_Cyrl` | 512 | `2.12e-4` | 4 |
| `pixel-base-adaptive` | `eng_Latn` | 256 | `1.5e-4` | 2 |

For example, to train `mpixel-4s`:

```bash
python mpixel/experiments/pretraining.py fit \
    --config=configs/experiments/mpixel-base.yaml \
    --data.data_dir=./data/pretraining \
    --data.languages=cmn_Hani,eng_Latn,hin_Deva,ukr_Cyrl \
    --model.learning_rate=1.5e-4 \
    --trainer.devices=4 \
    --trainer.num_nodes=2 \
    --trainer.logger.init_args.name=mpixel-4s \
    --trainer.logger.init_args.save_dir=./logs/pretraining
```

`pixel-base-adaptive` spends the 30B-patch budget of `mpixel-4s` on English only, so it needs its own data folder:

```bash
python scripts/data/download_fineweb2.py eng_Latn \
    --num-train-patches=30000000000 \
    --patches-per-file=2500000 \
    --output-dir=./data/pretraining-english/eng_Latn
```

Then train the model:

```bash
python mpixel/experiments/pretraining.py fit \
    --config=configs/experiments/mpixel-base.yaml \
    --data.data_dir=./data/pretraining-english \
    --data.languages=eng_Latn \
    --model.learning_rate=1.5e-4 \
    --trainer.devices=4 \
    --trainer.num_nodes=2 \
    --trainer.logger.init_args.name=pixel-base-adaptive \
    --trainer.logger.init_args.save_dir=./logs/pretraining
```
</details>

### Selecting a checkpoint

We select the checkpoint with zero-shot bitext mining on [FLORES-200](https://huggingface.co/datasets/facebook/flores), over the pretraining languages of each model. To prevent leakage, [`download_flores.py`](../scripts/validation/download_flores.py) removes the examples that overlap with SIB-200.

FLORES-200 uses different codes for two languages: `zho_Hans` for `cmn_Hani`, and `pes_Arab` for `fas_Arab`.

```bash
LOG_DIR="./logs/pretraining/mpixel/v0"

# Download FLORES-200 to ./data/flores
python scripts/validation/download_flores.py --output-dir=./data

# Evaluate all checkpoints
python scripts/validation/run_flores.py \
    --checkpoint ${LOG_DIR}/checkpoints \
    --languages arb_Arab ben_Beng zho_Hans dan_Latn ell_Grek eng_Latn pes_Arab fin_Latn \
                heb_Hebr hin_Deva kor_Hang spa_Latn tha_Thai tur_Latn ukr_Cyrl vie_Latn \
    --data-dir=./data \
    --renderer=adaptive \
    --output-path=${LOG_DIR}/flores.json

# Print the five best (step, layer) pairs by average recall@5
python scripts/validation/rank_checkpoints.py --input=${LOG_DIR}/flores.json --top=5
```

From these five entries, we select the checkpoint with the largest step, which is the most trained one.

### Converting to Hugging Face format

[`convert_to_huggingface.py`](../scripts/utils/convert_to_huggingface.py) converts a Lightning checkpoint into a model folder that also contains the processor config.

```bash
LOG_DIR="./logs/pretraining/mpixel/v0"

python scripts/utils/convert_to_huggingface.py \
    --checkpoint=${LOG_DIR}/checkpoints/<best-checkpoint>.ckpt \
    --config=${LOG_DIR}/config.yaml \
    --output-dir=./models/my-mpixel
```

You can load this folder with `from_pretrained`, or finetune it as described in [FINETUNING.md](FINETUNING.md).
