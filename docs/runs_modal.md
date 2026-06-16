# Runs Modal — historique, résultats, limites

> Document vivant. Mis à jour après chaque run et chaque correction.
> Sert de référence pour le rapport et pour décider des prochaines actions.

---

## Glossaire des variantes

### Modèle complet (NMS dynamique inclus)
| Variante | Description |
|---|---|
| `baseline` | FP32 brut, référence |
| `fp16` | autocast (Tensor Cores), zéro compilation |
| `torchscript` | `jit.script` + `freeze` + `optimize_for_inference` |
| `compile` | `torch.compile` backend `inductor`, `dynamic=False` |
| `cudagraphs` | `torch.compile` backend `cudagraphs` (capture de graphe CUDA) |
| `compile_fp16` | autocast + inductor |
| `cudagraphs_fp16` | autocast + cudagraphs |
| `torchscript_fp16` | autocast + torchscript |
| `trt_fp16` | TensorRT FP16 via `torch_tensorrt`, modèle complet |

### Par zone (backbone+FPN+heads optimisé, NMS reste eager)
| Variante | Description |
|---|---|
| `zone_torchscript` | TorchScript sur zone statique |
| `zone_compile` | inductor sur zone statique |
| `zone_cudagraphs` | CUDA graphs sur zone statique |
| `zone_trt_fp16` | TensorRT FP16 sur backbone (pièce TRT propre — exigence prof) |
| `zone_trt_folded` | constant-fold (`jit.freeze`) **puis** TRT — pour BiFPN |
| `mixed_trt_bb__cudagraphs_rest` | TRT(backbone) + cudagraphs(FPN+heads) |

---

## Historique des runs

### Run #1 — 2026-06-15 21:00 (Colab T4)

**Contexte** : R50 sur Colab T4, sortie console seulement (pas de Volume).

**Résultats** (R50, full-model uniquement, 500 images éval) :

| Variante | ms | FPS | speedup | MAP | Statut |
|---|---|---|---|---|---|
| baseline | 58.55 | 17.1 | ×1.00 | 0.401 | OK |
| fp16 | 62.65 | 16.0 | **×0.94** | 0.401 | OK (régresse) |
| torchscript | 50.94 | 19.6 | ×1.15 | 0.401 | OK |
| compile | 61.02 | 16.4 | **×0.96** | — | OK (régresse) |
| cudagraphs | 64.92 | 15.4 | **×0.90** | — | OK (régresse) |
| **compile_fp16** | **32.47** | **30.8** | **×1.80** | 0.401 | **OK ⭐** |
| cudagraphs_fp16 | 43.39 | 23.0 | ×1.35 | 0.401 | OK |
| torchscript_fp16 | 44.75 | 22.3 | ~~×1.31~~ | 0.401 | **OK (FAUX)** |
| `trt_fp16` | — | — | — | — | **FAILED** |

### Run #2 — 2026-06-15 23:42 (Modal A100 80 GB)

**Contexte** : premier test Modal, A100 80 GB, image NGC 24.10 **avec** `add_python="3.11"`.

**Résultats** (R50, 500 images éval, 4 variantes seulement) :

| Variante | ms | speedup | MAP | Statut |
|---|---|---|---|---|
| baseline | 30.99 | ×1.00 | 0.401 | OK |
| fp16 | 42.76 | **×0.725** | 0.401 | OK (régresse) |
| `trt_fp16` | — | — | — | **SKIPPED** (TRT absent) |
| compile_fp16 | 19.85 | ×1.56 | 0.401 | OK |

**Anomalie majeure** : `torch_tensorrt` et `tensorrt` étaient absents du conteneur
malgré la base NGC. Cause : **`add_python="3.11"` réinstalle Python par-dessus
NGC** et casse les paquets natifs préinstallés. → **Fix appliqué dans Run #3**.

### Run #3 — à venir (Modal T4 16 GB, 3 modèles × ~15 variantes)

**Configuration** :
- Image : NGC 24.10 **sans** `add_python` (Python 3.10 natif préservé)
- GPU : T4 16 GB
- CPU/RAM : 8 vCPU / 32 GB par conteneur
- Volume `dsai2026` (1 TB inclus dans plan Starter)
- Caches partagés : `TORCH_HOME=/data/cache/torch`, `HF_HOME=/data/cache/hf`
- Paramètres : N_WARMUP=50, N_MEASURE=1000, N_PROFILE=150, N_PROFILE_DATA=2000, N_EVAL=2000
- 1 conteneur par couple (modèle, variante), parallélisme = 6
- 3 modèles × 12-15 variantes = ~42 jobs

**Tableau à remplir après le run**.

---

## Limites observées par approche

### baseline
- OK partout. Métrique de référence.

### fp16 (autocast pur)
- **Régresse** sur A100 (×0.725) et sur T4 (×0.94) **quand le baseline est rapide** : l'overhead des casts FP32↔FP16 dépasse le gain Tensor Cores.
- → FP16 seul n'est jamais le gagnant ; il faut **combiner avec une compilation** (compile, torchscript).

### torchscript (full)
- T4 : ×1.15 (gain modeste — fusion Conv+BN).
- Avec FP16 wrapper, la trace échoue (mauvais format input) → **doit lever une exception**, pas retourner le modèle original. **Fix appliqué.**

### compile (full)
- Le NMS dynamique cause des **recompilations en boucle** sur `decode_single`, `batched_nms`, `clip_boxes_to_image`.
- Atteint `recompile_limit (8)` → retombe en eager pour les shapes non vues.
- T4 : ×0.96 (régresse). N'est utile **qu'en combinaison avec FP16**.

### cudagraphs (full)
- **`skipping cudagraphs due to cpu device`** sur `anchor_generator.set_cell_anchors` et `_batched_nms_coordinate_trick`.
- **`CUDA Graph is empty`** — cudagraphs ne capture rien d'utile.
- T4 : ×0.90 (régresse). Inutilisable sur le modèle complet.

### compile_fp16 (full)
- **Le winner du modèle complet** : T4 ×1.80, A100 ×1.56.
- Subit aussi des recompiles NMS mais le gain backbone domine.
- À mettre en avant dans le rapport.

### trt_fp16 (full)
- **Échec systématique** : `out of bounds slice ... input dimensions = [0,1]` puis `Error while setting the input shape`.
- Cause racine : le NMS produit parfois zéro détection → TRT essaie de compiler avec input batch=0.
- TRT demande explicitement dans le log : *« consider constant fold the model first »* et *« set upper bound on dynamic shapes »*.
- → **Inutilisable sur le modèle complet.** La solution est `zone_trt_fp16`.

### zone_trt_fp16 — la voie TRT propre
- Optimise **uniquement** le backbone (shapes fixes 640×640), laisse le NMS en eager.
- Pas de shapes dynamiques → TRT compile proprement.
- **C'est ce qu'on doit présenter dans le rapport pour satisfaire l'exigence prof.**
- À mesurer dans Run #3.

### zone_trt_folded — pour le BiFPN (EfficientDet)
- `jit.freeze` propage les constantes des poids gelés → la **fusion pondérée** du BiFPN devient une addition pondérée standard → TRT peut la fusionner.
- TRT le réclamait explicitement (*« consider constant fold the model first »*).
- À mesurer dans Run #3.

### zone_cudagraphs
- ×1.71 sur R50 mesuré en local (RTX 5060) → la **suppression de l'overhead de lancement** des kernels est le levier sur GPU rapide.
- À confirmer sur T4 (gain probablement plus modeste car T4 plus lente, donc l'overhead pèse moins en relatif).

### mixed_trt_bb__cudagraphs_rest
- Hypothèse : TRT(backbone) + cudagraphs(FPN+heads) cumule les gains.
- À mesurer sur les 3 modèles. Risque : les transitions entre régions optimisées peuvent coûter (copies de buffers).

---

## Fixes appliqués

### Fix #1 — `torchscript` fail-loud (commité)
Avant : si `script` et `trace` échouent, `optimize_with_torchscript` retournait le modèle original silencieusement. Le runner mesurait alors le modèle eager et rapportait un **faux speedup** (cf. Run #1, `torchscript_fp16` ×1.31 fictif).
Après : lève `RuntimeError`. Le try/except du runner marque `FAILED` proprement.

### Fix #2 — Modal sans `add_python` (commité)
Avant : `add_python="3.11"` réinstallait Python par-dessus NGC, dégradant les paquets natifs (torch_tensorrt, tensorrt). → TRT indisponible.
Après : on garde le Python NGC natif. `effdet` et `timm` installés avec `--no-deps` pour ne pas toucher au `torch` NGC. TRT préservé.

### Fix #3 — caches partagés sur le Volume (commité)
Avant : chaque conteneur re-téléchargeait `retinanet_resnet50_fpn_v2_coco-5905b1c5.pth` (146 MB).
Après : `TORCH_HOME=/data/cache/torch` → téléchargé une fois, partagé entre tous les conteneurs Modal.

### Fix #4 — granularité 1 job = 1 (modèle, variante) (commité)
Avant : tous les variantes d'un modèle dans un seul conteneur → heartbeat timeouts (MAP eval bloquante), pollution d'état possible (résolu par reset mais fragile).
Après : un conteneur par couple. Plus d'isolation, plus de timeouts heartbeat, relance ciblée gratuite.

---

## À faire (demain ou plus tard)

Liste des chantiers identifiés, à coder une fois Run #3 analysé :

- [ ] **Bornage shapes dynamiques pour TRT** : `torch._dynamo.mark_dynamic` ou `torch.export.Dim` pour fixer un upper bound sur le nombre max de boîtes après NMS → permettrait `trt_fp16` full-model (peu prioritaire car `zone_trt_fp16` est la vraie voie).
- [ ] **Constant folding via onnxsim** pour la voie C2 du cahier (alternative à `jit.freeze` pour `zone_trt_folded`).
- [ ] **Réécriture éventuelle de `FpnCombine.forward` (BiFPN)** pour utiliser les coefficients pré-calculés à l'inférence (`relu(w)/Σw` → constantes en eval) → rend le BiFPN TRT-friendly sans dépendre de `freeze`.
- [ ] Analyse profilers : extraire les opérations les plus accélérables (point 2 du cahier des charges) à partir des CSV de `/data/results/<run_id>/profiles/`.

---

## Pour interpréter un run

Récupérer les résultats :
```bash
modal volume get dsai2026 results/<run_id> ./local_run/
```

Arborescence type :
```
local_run/
  results.csv                    ← tableau de synthèse (à ouvrir d'abord)
  bench/<model>_<variant>.json   ← métriques de vitesse brutes
  eval/<model>_<variant>.json    ← MAP/AR COCO complètes (par variante)
  modules/<model>_<variant>.csv  ← timing par module feuille (baseline/fp16)
  profiles/<model>_<variant>.csv ← table d'opérations (kernels/mémoire)
  logs/<model>_<variant>.log     ← stdout/stderr complet du conteneur
  errors/<model>_<variant>.txt   ← traceback Python si CONTAINER_FAILED
```

Pour comprendre pourquoi une variante a échoué :
1. Regarder `results.csv` → statut
2. Si `FAILED` ou `CONTAINER_FAILED` → ouvrir `errors/<model>_<variant>.txt` et `logs/<model>_<variant>.log`
3. Chercher dans le log les warnings spécifiques au framework (TRT, dynamo, cudagraphs)
