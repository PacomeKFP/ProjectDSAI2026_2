# ProjectDSAI2026 -- Detection-model inference optimization benchmark

A reproducible benchmark of GPU inference acceleration techniques
(TensorRT, torch.compile, TorchScript, CUDA Graphs, FP16 autocast)
applied to three families of object-detection models:

- **RetinaNet R50** (torchvision)
- **FCOS R50** (torchvision)
- **EfficientDet D4** (Ross Wightman's `effdet`)

Each (model, optimization variant) pair is benchmarked for speed
(`mean_ms`, `fps`), evaluated for COCO MAP on val2017, and profiled at
the operation level with `torch.profiler`. Runs are orchestrated on
the [Modal](https://modal.com) cloud platform with one T4 GPU container
per pair, ensuring full isolation between experiments.

## Headline results

Latest run: **Modal T4 16 GB, 16 June 2026** -- 3 models x 13 variants
(39 jobs, 4 expected failures). Full table:
[`docs/run_3_results.md`](docs/run_3_results.md).

| Model            | Best variant         | Speedup | AP COCO val2017 |
|------------------|----------------------|--------:|----------------:|
| EfficientDet D4  | `torchscript_fp16`   | x3.57   | 0.4480 (vs 0.4477) |
| RetinaNet R50    | `compile_fp16`       | x2.29   | 0.3774 (vs 0.3775) |
| FCOS R50         | `compile_fp16`       | x2.13   | 0.3336 (vs 0.3361) |

Cleanest TensorRT path (zone-based, AP preserved exactly):

| Model            | Variant            | Speedup |
|------------------|--------------------|--------:|
| EfficientDet D4  | `zone_trt_fp16`    | x2.30   |
| FCOS R50         | `zone_trt_folded`  | x1.93   |
| RetinaNet R50    | `zone_trt_folded`  | x1.66   |

## Repository layout

```
ProjectDSAI2026/
├── README.md                      this file
├── requirements.txt               base pip dependencies
├── setup.py                       one-shot project initialization
|
├── modal_runner.py                Modal orchestrator (1 container per run)
├── modal_test_env.py              Modal environment validation script
|
├── models/                        per-model specs (load + pre/post)
│   ├── retinanet_r50.py
│   ├── retinanet_r101.py
│   ├── fcos_r50.py
│   ├── efficientdet_d4.py
│   ├── efficientdet_d5.py
│   └── efficientdet_d6.py
|
├── optimizations/                 optimization toolbox
│   ├── runner.py                  OptimizationRunner orchestrator
│   ├── capability.py              environment capability detection
│   ├── fp16_half.py               FP16 autocast wrapper
│   ├── torchscript.py             TorchScript graph compilation
│   ├── torch_compile.py           torch.compile (inductor / cudagraphs)
│   ├── tensorrt_fp16.py           TensorRT FP16 via torch_tensorrt
│   ├── tensorrt_int8.py           TensorRT INT8 + PTQ calibration
│   ├── zones.py                   architecture-aware zone / sub-zone
│   ├── onnx_export.py             ONNX export
│   ├── ort_inference.py           ONNX Runtime CUDA EP
│   ├── inspect_zones.py           model-tree inspection
│   └── paths.py                   unified output-prefix helper
|
├── utils/                         shared utilities
│   ├── benchmark.py               benchmark_model + ModuleBenchmark
│   ├── data_loader.py             lazy COCO loader
│   ├── download_dataset.py        COCO val2017 downloader
│   └── tqdm_compat.py             tqdm fallback shim
|
├── eval/
│   └── map_eval.py                COCO MAP evaluation loop
|
├── profiler/
│   ├── pytorch_profiler.py        torch.profiler driver + tables
│   └── nsight_profiler.py         NVTX-annotated runner for Nsight Systems
|
├── docs/                          design docs and run reports
│   ├── run_3_results.md           full report on the latest run
│   ├── runs_modal.md              Modal run history (living document)
│   ├── cahier_implementation_v3.md  zone/sub-zone implementation plan
│   └── figures/                   PNG figures for the report
|
├── outputs/                       runs fetched from the Modal Volume
│   └── <run_id>/
│       ├── results.csv            aggregated summary
│       ├── bench/                 per-pair speed metrics (JSON)
│       ├── eval/                  per-pair COCO MAP (JSON)
│       ├── modules/               per-leaf-module timing (CSV)
│       ├── profiles/              per-operation profile (CSV)
│       ├── logs/                  per-pair container stdout/stderr
│       └── errors/                Python tracebacks on failure
|
├── datasets/                      COCO val2017 (populated by setup.py)
├── detectron2/                    Detectron2 package (cloned safeguard)
└── *.ipynb                        legacy local notebooks (pre-Modal)
```

## Prerequisites

| Requirement     | Notes                                                  |
|-----------------|--------------------------------------------------------|
| Python          | 3.10+ locally, 3.13 in the Modal image                 |
| CUDA            | NVIDIA GPU with CUDA 12+ (TensorRT requires Volta+, CC >= 7.0) |
| Modal account   | Required to reproduce the cloud runs (modal.com)       |
| Disk            | ~5 GB for COCO val2017 + model caches                  |

A `requirements.txt` covers the base dependencies. The optimization stack
(`torch-tensorrt`, `onnx`, `onnxruntime-gpu`, `onnxsim`) is installed
inside the Modal image (see [modal_runner.py](modal_runner.py)) or in
the legacy notebooks for local experiments.

## Installation

### Local

```bash
git clone https://github.com/PacomeKFP/ProjectDSAI2026_2.git
cd ProjectDSAI2026_2

# 1. Install base dependencies and download COCO val2017
python setup.py

# Skip steps with: --skip-deps  --skip-coco  --skip-d2
```

`setup.py` will:
1. install the base pip dependencies,
2. create the project directories,
3. download COCO val2017 (images + annotations, ~1 GB),
4. clone Detectron2 (kept as a safeguard, not required for the main path).

### Modal cloud

```bash
pip install modal
modal token new                      # one-time setup

# (Optional) Validate the cloud image
modal run modal_test_env.py
```

## Running benchmarks

### Full cloud run (recommended)

```bash
# Submit all (model, variant) pairs in parallel; ~1 h on T4
modal run --detach modal_runner.py
```

Each pair runs in its own container. Results are streamed to the
`dsai2026` Modal Volume.

### Selective run

```bash
# A subset of models
modal run modal_runner.py --models retinanet_r50,fcos_r50

# A subset of variants
modal run modal_runner.py --variants baseline,fp16,zone_trt_fp16

# Cap concurrent containers (cost does not change, only wall-time)
modal run modal_runner.py --parallel 4
```

### Fetching results

```bash
# Pull a finished run from the Modal Volume
modal volume get dsai2026 results/<run_id> ./outputs/

# Inspect
ls outputs/<run_id>/                 # results.csv + bench/ eval/ logs/ ...
```

### Local notebooks

The historical notebooks (`optimization_full.ipynb`,
`optimization_benchmark.ipynb`) still run end-to-end on a local CUDA GPU.
They use the same `optimizations/` and `utils/` modules as the cloud
runner.

## Reproducing the headline results

The latest published run lives at
[`outputs/20260616_122931/`](outputs/20260616_122931) and is fully
described in [`docs/run_3_results.md`](docs/run_3_results.md). To
reproduce:

```bash
modal run --detach modal_runner.py \
    --n-warmup 50 --n-measure 1000 \
    --n-profile 150 --n-profile-data 2000 --n-eval 2000
```

Same seed (42), same N_WARMUP / N_MEASURE, same image size (640x640) and
same Modal image (Debian slim Python 3.13 + pinned PyPI stack -- see
[modal_runner.py](modal_runner.py)).

## Documentation map

| Document                                                   | Audience               |
|------------------------------------------------------------|------------------------|
| [`docs/run_3_results.md`](docs/run_3_results.md)           | full report on the latest run, including the TRT submodule analysis, the cross-model accelerable-op ranking, and next steps |
| [`docs/runs_modal.md`](docs/runs_modal.md)                | living history of every Modal run, per-approach limits, applied fixes |
| [`docs/cahier_implementation_v3.md`](docs/cahier_implementation_v3.md) | design doc for the zone / sub-zone optimization framework |
| [`docs/figures/`](docs/figures)                            | rendered PNG figures used by the report |

## Project conventions

- Identifiers (variable, function, class, dict-key names) stay in English.
- All comments, docstrings and printed messages are in English.
- ASCII-only source files (no decorative Unicode).
- Each optimization variant is built lazily and isolated in its own
  container -- failures are recorded but never propagate.

## Citation

If this code is useful for your work, please cite the repository:

```
@misc{projectdsai2026,
  author = {Pacome K.F.P.},
  title  = {ProjectDSAI2026 -- Detection-model inference optimization benchmark},
  year   = {2026},
  url    = {https://github.com/PacomeKFP/ProjectDSAI2026_2}
}
```

## License

Academic project -- no public license attached. Contact the author for
reuse.
