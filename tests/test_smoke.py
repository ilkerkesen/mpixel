"""Smoke tests: build a tiny MPIXEL model, run it, and build each experiment."""

import importlib
import pkgutil

import pytest
import torch

import mpixel.experiments
from mpixel.models.mpixel.configuration_mpixel import MPIXELConfig
from mpixel.models.mpixel.modeling_mpixel import MPIXELForPreTraining
from mpixel.experiments.pretraining import PixelPretrainingExperiment
from mpixel.experiments.text_classification import TextClassificationExperiment
from mpixel.experiments.glue import GLUEExperiment
from mpixel.experiments.xnli import XNLIExperiment
from mpixel.experiments.ner import NERExperiment
from mpixel.experiments.udp import UDPExperiment
from mpixel.models.tasks.classification import PixelForSequenceClassification
from mpixel.models.tasks.token_classification import PixelForTokenClassification
from mpixel.models.tasks.udp import PixelForBiaffineParsing

PATCH_SIZE = 16
MAX_NUM_PATCHES = 8
BATCH_SIZE = 2


def _tiny_config(**kwargs):
    # mask_ratio must be > 0: the default 0.0 masks no patches and gives a NaN loss.
    defaults = dict(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        decoder_hidden_size=32,
        decoder_num_hidden_layers=1,
        decoder_num_attention_heads=2,
        decoder_intermediate_size=64,
        patch_size=PATCH_SIZE,
        max_num_patches=MAX_NUM_PATCHES,
        mask_ratio=0.25,
        attn_implementation="sdpa",
    )
    defaults.update(kwargs)
    return MPIXELConfig(**defaults)


def _build_model(**kwargs):
    torch.manual_seed(0)
    return MPIXELForPreTraining._from_config(_tiny_config(**kwargs)).eval()


@pytest.fixture
def inputs():
    """A batch of two images; the second one has two padding patches."""
    torch.manual_seed(1)
    pixel_values = torch.rand(BATCH_SIZE, 3, PATCH_SIZE, PATCH_SIZE * MAX_NUM_PATCHES)
    attention_mask = torch.ones(BATCH_SIZE, MAX_NUM_PATCHES)
    attention_mask[1, -2:] = 0
    patch_mask = torch.zeros(BATCH_SIZE, MAX_NUM_PATCHES)
    patch_mask[:, 2:4] = 1
    return pixel_values, attention_mask, patch_mask


@pytest.fixture
def checkpoint_path(tmp_path):
    """A Lightning-style pretraining checkpoint of the tiny model."""
    model = _build_model()
    path = tmp_path / "tiny.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
            "model_config": model.config.to_dict(),
            "hyper_parameters": {},
        },
        path,
    )
    return str(path)


class TestMPIXELForPreTraining:
    """Forward pass of the pretraining model."""

    def test_forward_with_patch_mask(self, inputs):
        """Controlled masking: finite loss and one reconstruction per patch."""
        pixel_values, attention_mask, patch_mask = inputs
        model = _build_model()
        with torch.no_grad():
            out = model(pixel_values=pixel_values, attention_mask=attention_mask, patch_mask=patch_mask)
        assert torch.isfinite(out.loss)
        assert out.logits.shape == (BATCH_SIZE, MAX_NUM_PATCHES, PATCH_SIZE**2 * 3)
        num_masked = MAX_NUM_PATCHES - int(MAX_NUM_PATCHES * (1 - model.config.mask_ratio))
        assert out.mask.sum(dim=1).tolist() == [num_masked] * BATCH_SIZE

    def test_forward_without_patch_mask(self, inputs):
        """Without a patch mask, the model falls back to random masking."""
        pixel_values, attention_mask, _ = inputs
        model = _build_model()
        with torch.no_grad():
            out = model(pixel_values=pixel_values, attention_mask=attention_mask)
        assert torch.isfinite(out.loss)
        assert out.logits.shape == (BATCH_SIZE, MAX_NUM_PATCHES, PATCH_SIZE**2 * 3)

    def test_eager_matches_sdpa(self, inputs):
        """Eager and SDPA attention give the same outputs, also with padding."""
        pixel_values, attention_mask, patch_mask = inputs
        sdpa = _build_model(attn_implementation="sdpa")
        eager = _build_model(attn_implementation="eager")
        eager.load_state_dict(sdpa.state_dict())
        outputs = []
        for model in (sdpa, eager):
            with torch.no_grad():
                encoder = model.vit(pixel_values, attention_mask=attention_mask, output_hidden_states=True)
                torch.manual_seed(2)  # controlled masking adds a small random noise
                out = model(pixel_values=pixel_values, attention_mask=attention_mask, patch_mask=patch_mask)
            outputs.append((encoder, out))
        (sdpa_enc, sdpa_out), (eager_enc, eager_out) = outputs
        assert len(eager_enc.hidden_states) == len(sdpa_enc.hidden_states)
        for eager_h, sdpa_h in zip(eager_enc.hidden_states, sdpa_enc.hidden_states):
            torch.testing.assert_close(eager_h, sdpa_h)
        torch.testing.assert_close(eager_enc.last_hidden_state, sdpa_enc.last_hidden_state)
        torch.testing.assert_close(eager_out.logits, sdpa_out.logits)
        torch.testing.assert_close(eager_out.loss, sdpa_out.loss)


class TestBuildExperiments:
    """Each LightningModule builds from a tiny config or checkpoint."""

    def test_pretraining(self, tmp_path):
        config_path = tmp_path / "model_config.json"
        _tiny_config().to_json_file(config_path)
        experiment = PixelPretrainingExperiment(
            config_path=str(config_path),
            compile_model=False,
            patch_size=PATCH_SIZE,
            max_num_patches=MAX_NUM_PATCHES,
        )
        assert isinstance(experiment.model, MPIXELForPreTraining)

    @pytest.mark.parametrize("experiment_cls,kwargs,model_cls", [
        (TextClassificationExperiment, dict(num_labels=7), PixelForSequenceClassification),
        (GLUEExperiment, dict(num_labels=2, task_name="sst2"), PixelForSequenceClassification),
        (XNLIExperiment, dict(num_labels=3), PixelForSequenceClassification),
        (NERExperiment, dict(labels=["O", "B-PER", "I-PER"]), PixelForTokenClassification),
        (UDPExperiment, dict(), PixelForBiaffineParsing),
    ])
    def test_downstream_loads_encoder(self, checkpoint_path, experiment_cls, kwargs, model_cls):
        """The task model wraps the encoder weights from the checkpoint."""
        experiment = experiment_cls(model_id_or_path=checkpoint_path, **kwargs)
        assert isinstance(experiment.model, model_cls)
        state_dict = torch.load(checkpoint_path, weights_only=False)["state_dict"]
        for name, value in experiment.model.vit.state_dict().items():
            assert torch.equal(value, state_dict[f"model.vit.{name}"]), name


@pytest.mark.parametrize("module_name", [
    m.name for m in pkgutil.iter_modules(mpixel.experiments.__path__)
])
def test_import_experiment_module(module_name):
    importlib.import_module(f"mpixel.experiments.{module_name}")
