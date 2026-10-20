# Known issues and troubleshooting

## Setup

**`401` / `GatedRepoError` when a container downloads FLUX.** The FLUX base models are gated. Accept
the licences of [FLUX.1 Kontext [dev]](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
and [FLUX.2 klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) with the account
whose token is stored in the `huggingface-secret` (and `huggingface-secret-2`) Modal secrets.

**`Secret 'huggingface-secret-2' not found`.** The FLUX.2 klein training launchers use a second
secret name; create it with the same token ([installation](installation.md#modal-secrets)).

**Modal authentication errors.** Run `modal token new`, or export `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET`. Scripts that call the `modal` CLI as a subprocess use `modal` from your
`PATH`; set `MODAL_BIN` to point elsewhere.

**GPU type not available.** Training scripts request B200 GPUs. Change `GPU_TYPE` (and, if needed,
the batch size in `TRAINING_CONFIG`) at the top of the launcher.

**`ModuleNotFoundError: segment_anything` / `inference` / `model`.** Run
`bash third_party/fetch_third_party.sh xmem sam`. The data pipeline uses its patched Segment
Anything checkout automatically; for the evaluator, `pip install segment-anything` (included in
`evaluator/environment.yml`) works as well. XMem's modules are imported from the checkout, so start
the scripts with `PYTHONPATH=./XMem` from inside `data_pipeline/` or `evaluator/`.

**`FileNotFoundError` for `sam_vit_h_4b8939.pth`, `XMem.pth` or `loftr_weights/`.** These weights are
not downloaded automatically; see [`third_party/README.md`](../../third_party/README.md#model-weights)
for where each one goes.

## Running

**`run_pipeline.py` fails to find the data or checkpoints.** Pass absolute paths: each step runs
from its own directory and uploads the folders to Modal.

**A long Modal job stopped part-way.** The FLUX.2 klein pipeline steps write outputs per example
and skip examples that already have them, and its driver retries a failed step, so re-running the
same command resumes. Training launchers resume from the latest checkpoint on their
volume (FLUX.1 Kontext: `--action resume`).

**`Token indices sequence length is longer than the specified maximum sequence length … (> 77)`**
during FLUX.1 Kontext inference. The CLIP text encoder truncates long prompts to 77 tokens. The
model's T5 encoder (512 tokens) still sees the full prompt, so the warning is expected.

**CUDA out of memory in the evaluator.** Run the dimension scripts with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `lgie_evaluator.py` already loads each group of
models once per phase and frees them before the next phase.

## SIMPLER-Bridge

**`ErrorExtensionNotPresent` / no Vulkan device.** Headless containers have no GPU display; the
rollout image renders with Mesa lavapipe on the CPU. Use the image defined in
`modal_rollout_gc.py`; `modal_vulkan_diag.py` reports what a container exposes.

**Segmentation fault in `compute_forward_kinematics`.** SAPIEN 2.2.2's bundled pinocchio crashes on
newer glibc. The rollout image swaps in `pinocchio_ik_backend.py`; `modal_test_ik_backend.py`
checks that the replacement works.

See the [SIMPLER guide](simpler.md) for the full setup.

## Results look wrong

- Check that the adapters match the dataset: BridgeDataV2 adapters for real tabletop scenes,
  CALVIN or LIBERO adapters for those simulators.
- SGE-Goal expects **atomic** instructions. Split composite instructions into atomic steps and run
  them in sequence ([quickstart](quickstart.md#multi-step-instructions)).

Still stuck? Open an [issue](https://github.com/Riddhi-Chatterjee/SGE-Goal/issues) with the details
listed in [CONTRIBUTING.md](../../CONTRIBUTING.md#reporting-a-problem).
