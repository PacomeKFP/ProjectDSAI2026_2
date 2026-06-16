# ProjectDSAI2026 — Benchmark d'optimisation de modèles de détection

Comparaison empirique de techniques d'accélération inférence (TensorRT,
torch.compile, TorchScript, CUDA Graphs, autocast FP16) sur trois modèles
de détection : **RetinaNet R50**, **FCOS R50**, **EfficientDet D4**.

## État actuel

Dernier run : **Modal T4 16 GB, 16 juin 2026** — `outputs/20260616_122931/`.
3 modèles × 13 variantes = 39 jobs (4 échecs attendus, voir le rapport).

**Winners** :

| Modèle | Meilleure variante | Speedup vs baseline |
|---|---|---:|
| EfficientDet D4 | `torchscript_fp16` | **×3.57** |
| RetinaNet R50 | `compile_fp16` | ×2.29 |
| FCOS R50 | `compile_fp16` | ×2.13 |

**TRT « propre » (zone, AP conservée)** : ×2.30 (effdet), ×1.93 (FCOS via
`zone_trt_folded`), ×1.66 (RetinaNet via `zone_trt_folded`).

## Documents principaux

- [`docs/run_3_results.md`](docs/run_3_results.md) — **rapport complet du
  run actuel**, avec glossaire des variantes, conditions exactes, tableau
  37 lignes, analyse TRT sous-module sur EfficientDet, opérations les plus
  accélérables, et prochaines étapes.
- [`docs/runs_modal.md`](docs/runs_modal.md) — historique des runs Modal et
  limites observées par approche, document vivant.

## Organisation du repo

```
modal_runner.py            # orchestrateur Modal (1 conteneur par couple modèle×variante)
modal_test_env.py          # test d'environnement (à lancer avant un gros run)
optimizations/runner.py    # OptimizationRunner — boucle bench/profile/eval
optimizations/<technique>.py  # une voie d'optimisation par fichier
models/                    # spécification des 3 modèles (RetinaNet, FCOS, EffDet)
utils/                     # bench, profile, COCO eval
outputs/<run_id>/          # résultats d'un run (bench/, eval/, modules/, profiles/, logs/, errors/)
docs/                      # rapports et historiques
```

## Lancer un run

```bash
# 1. Vérifier l'environnement Modal (T4, image debian_slim+pip)
modal run modal_test_env.py

# 2. Lancer le bench complet (mode détaché conseillé — ~1h pour 39 jobs)
modal run --detach modal_runner.py

# 3. Récupérer les résultats
modal volume get dsai2026 results/<run_id> ./outputs/
```

Les paramètres (N_WARMUP, N_MEASURE, taille image, etc.) sont dans
`modal_runner.py`.
