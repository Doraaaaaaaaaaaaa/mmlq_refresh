# Complexity Measurement Helpers

This folder contains lightweight adapters for measuring model complexity of
the baselines collected in `duibishiyan1`.

The intended workflow is:

```powershell
cd C:\Users\admin\Desktop\大三下\论文\mmlq_refresh
conda activate <your-torch-env>
pip install thop
python duibishiyan1\measure_complexity\measure_ours.py
python duibishiyan1\measure_complexity\measure_mmlq.py
python duibishiyan1\measure_complexity\measure_tanet.py
python duibishiyan1\measure_complexity\measure_charm.py
python duibishiyan1\measure_complexity\measure_aesmamba.py --variant viaa
python duibishiyan1\measure_complexity\measure_aesmamba.py --variant miaa
```

Or run all adapters and keep raw outputs:

```powershell
python duibishiyan1\measure_complexity\run_all.py
```

Each script prints:

- total parameters
- trainable parameters
- GMACs, when `thop` can trace the forward pass

Some baselines need extra local assets before the model can be built:

- `MMLQ`: LAVIS, BERT, CLIP/EVA weights, and CUDA because the released code
  moves tokenized text directly to `cuda`.
- `TANet`: `nni`, `tensorboardX`, and `code/AVA/resnet18_places365.pth.tar`.
- `Charm`: HuggingFace backbone weights, usually `facebook/dinov2-small`.
- `AesMamba`: VMamba/mamba dependencies and pretrained VMamba checkpoints under
  `Checkpoints/pretrain_model`.
- `VILA`: JAX/Flax/TFHub code; use reported or TFHub-derived complexity unless
  a separate JAX profiling setup is prepared.
- `NIMA`: Keras/TensorFlow code; measure separately with TensorFlow tooling or
  use reported implementation complexity.

The current default shell environment may not be the training environment. If
`ModuleNotFoundError: No module named 'torch'` appears, activate the conda/venv
used for the IAA experiments before running these scripts.

Recommended table note:

```text
Complexity is measured using released or author-provided code when the model can
be instantiated locally. Performance values are reported from the original
papers unless otherwise specified.
```
