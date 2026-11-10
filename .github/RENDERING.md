## Rendering guide
MPIXEL renders text into an image strip before it reaches the model. This guide covers the [`MPIXELProcessor`](../mpixel/processing.py), which prepares model inputs, and the [rendering processors](../mpixel/renderer/processor.py) that it builds on. Our default strategy, adaptive rendering, extends the bigrams rendering strategy proposed in [Text Rendering Strategies for Pixel Language Models](https://aclanthology.org/2023.emnlp-main.628/).


### Loading a processor
A [`MPIXELProcessor`](../mpixel/processing.py) object can be created directly via its `__init__` method or using its `from_pretrained` method. The easiest way is to use `from_pretrained`, which supports loading from the Hugging Face Hub or from disk. It reads a `preprocessor_config.json` file, which can look as follows:

```json
{
  "patch_size": 16,
  "max_num_patches": 512,
  "font_size": 12
}
```

- `patch_size`: the height and width of one image patch, in pixels.
- `max_num_patches`: the width of the output strip, in patches.
- `font_size`: the font size used for rendering.

All fields are optional. The values above are the defaults.

```python
from mpixel.processing import MPIXELProcessor

# Load from Hugging Face Hub
processor = MPIXELProcessor.from_pretrained("Team-PIXEL/mpixel-base")

# Load from a folder on disk containing a preprocessor_config.json file
processor = MPIXELProcessor.from_pretrained("/path/to/my/model")
```

You can write this file with `processor.save_pretrained("/path/to/my/model")`.

#### Fonts
You do not need a font file or a directory of fallback fonts. `MPIXELProcessor` always uses `FONTS_NOTO_SANS_EXTENDED` from [`mpixel/renderer/fonts.py`](../mpixel/renderer/fonts.py). This list contains the Noto Sans fonts from a fixed commit of [google/fonts](https://github.com/google/fonts), plus Noto Serif Tibetan. The fonts are downloaded automatically, and rendering does not depend on the fonts installed on your system.


### Inputs and outputs
#### Processor inputs
The processor takes a string or a list of strings:

```python
from mpixel.processing import MPIXELProcessor

processor = MPIXELProcessor.from_pretrained("Team-PIXEL/mpixel-base")

# Render a single string
inputs = processor("My cat loves oatmeal.")

# Render a batch of strings
inputs = processor(["My cat loves oatmeal.", "Kedim yulafı çok seviyor."])
```

The processor does not take string pairs or lists of words. For these tasks, the data modules render each sentence or token separately. See [`glue.py`](../mpixel/datamodules/glue.py) for sentence pairs and [`ner.py`](../mpixel/datamodules/ner.py) for token-level rendering.

#### Processor outputs
The processor outputs a dictionary that you can pass directly to the model:

```python
pixel_values: torch.Tensor    # (batch_size, 3, patch_size, max_num_patches * patch_size)
attention_mask: torch.Tensor  # (batch_size, max_num_patches)
```

`pixel_values` contains RGB values between 0 and 1. A black separator patch follows the text, and white patches fill the remaining width. `attention_mask` is 1 for the text patches and the separator patch, and 0 for the white patches. If the text does not fit in `max_num_patches`, the processor cuts the text and keeps the separator patch.


### Rendering processors
The [rendering processors](../mpixel/renderer/processor.py) produce a strip with the width of the text, without padding or tensor conversion. First, they split the text on whitespace into segments. Then, they split each segment into units, and render each unit at the start of a new patch.

| Processor | Arabic, Hebrew, N'Ko, Syriac, Adlam, Mongolian, Mandaic, Hanifi Rohingya | Han, Hiragana, Katakana, Hangul | Other scripts |
|---|---|---|---|
| `AdaptiveRenderingProcessor` | Whole words | Single graphemes | Grapheme bigrams |
| `WordsRenderingProcessor` | Whole words | Single graphemes | Whole words |
| `BigramsRenderingProcessor` | Grapheme bigrams | Grapheme bigrams | Grapheme bigrams |

`MPIXELProcessor` uses `AdaptiveRenderingProcessor`. Units always stay in logical text order, also for right-to-left scripts.

#### Renderer inputs
A rendering processor takes one string and these optional arguments:

- `block_size`: the patch size in pixels. The default is 16.
- `font_size`: the font size. The default is 12.
- `pad`: the horizontal padding of each unit in pixels. An `int` sets the left padding only. A `(left, right)` tuple sets both sides. The default is 1.

#### Renderer outputs
A rendering processor outputs a dictionary with these fields:

```python
pixel_values: np.ndarray        # (block_size, width, 3), uint8 RGB
segment_start_mask: np.ndarray  # (width // block_size,), int64
```

The last patch of `pixel_values` is the black separator patch. `segment_start_mask` has one value for each patch. The value is 1 on the first patch of each segment and 0 on all other patches, including the separator patch.

#### Selecting a processor for training
Set `renderer_name` for pretraining and `renderer_type` for finetuning. Both accept `adaptive` (the default), `bigrams` and `words`.

#### Demo
To see the output of the rendering processors, run the Gradio demo:

```bash
pip install gradio
python scripts/demo/renderer.py
```
