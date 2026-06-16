# Run #3 — Modal T4 16 GB, 3 modèles × 13 variantes

> **Run ID** : `20260616_122931`
>
> **Artefacts** : [`outputs/20260616_122931/`](../results/20260616_122931) -- dispobible chez @PacomeKFP
>
> **Référence des choix** : [`docs/runs_modal.md`](runs_modal.md)
>
> **Auteur des notes** : PacomeKFP
>
> **Date du run** : 16 juin 2026


Ce document a deux objectifs : 
(1) servir de **support pédagogique** pour qu'on
puisse, plus tard, retrouver pourquoi telle variante donne tel chiffre, et
(2) répondre point par point aux **trois attentes de l'encadrant** (mail du
16 juin) — TRT au niveau sous-module, extraction des opérations rapides, et
ouverture vers la suite.

Lecture conseillée : lire dans l'ordre. Les chiffres n'ont de sens qu'avec la
définition exacte de la variante qui les produit.

---

## 1. Préambule — ce qu'on cherche à mesurer

On a trois modèles de détection issus de familles différentes :

- **RetinaNet R50** — `torchvision`, ResNet-50 + FPN + heads classification/régression dense, NMS classique
- **FCOS R50** — `torchvision`, anchor-free, ResNet-50 + FPN, *centerness* head, NMS
- **EfficientDet D4** — `effdet` (Ross Wightman), EfficientNet-B4 + BiFPN + heads

Pour chacun on benchmarke un **baseline FP32** puis une série de variantes
d'optimisation. À chaque fois on mesure : **temps moyen par image** (ms),
**FPS**, **AP COCO** (sur 2000 images de val2017) et — pour les baselines et
fp16 — le profil sous-module et opération.

Trois questions qui structurent la suite :

1. **Quel levier domine sur chaque architecture ?** (réponse : pas la même
   selon le modèle — voir §4)
2. **TensorRT tient-il ses promesses au niveau opération ?** (réponse :
   oui sur EfficientDet, partiellement sur torchvision — voir §5)
3. **Quelles briques élémentaires sont les plus accélérables ?**
   (Conv+BN+activation, depthwise, addition pointwise — voir §6)

---

## 2. Glossaire des variantes — ce que chaque tag *fait réellement*

Cette section répond à la demande « précise la constitution de chaque tag ».
Pour chaque variante on donne **(a)** ce qu'elle fait en pratique, **(b)** la
motivation théorique, et **(c)** le cas où elle est censée briller.

### `baseline`
- **Quoi** : modèle FP32 chargé tel quel, en mode `eval()`, exécuté avec
  `torch.no_grad()`. Aucune compilation, aucune fusion explicite, aucune
  conversion de précision. cuDNN choisit ses kernels en autotune au premier
  forward (warmup de 50 itérations avant la mesure).
- **Pourquoi** : c'est la référence. Tout speedup est mesuré par rapport à
  ce point. C'est aussi ce qu'on obtient si on déploie « bêtement » un modèle
  exporté de PyTorch sans toucher au runtime.
- **Quand ça brille** : jamais — c'est la cible à battre.

### `fp16` (autocast pur)
- **Quoi** : enveloppe le forward dans `torch.autocast(device_type="cuda",
  dtype=torch.float16)`. Aucun graphe, aucune fusion. PyTorch insère
  dynamiquement des casts FP32↔FP16 autour de chaque opération.
- **Pourquoi** : les Tensor Cores du T4 (architecture Turing, capability 7.5)
  exécutent les mat-muls FP16 ~2× plus vite que FP32. L'autocast est censé
  capter ce gain sans changer le code.
- **Quand ça brille** : sur les modèles très *compute-bound* avec de gros
  GEMM (gros backbones, batch large). **Quand le modèle est petit et*
  *memory-bound, les casts coûtent plus que ce qu'ils font gagner**.

### `torchscript`
- **Quoi** : trois étapes successives.
  1. `torch.jit.script(model)` — analyse le code Python source et produit un
     graphe statique. Si script échoue (modèle non-scriptable), fallback en
     `torch.jit.trace` avec un input exemple.
  2. `torch.jit.freeze` — inline les poids comme constantes, supprime les
     attributs inutiles, ouvre la porte au constant folding.
  3. `torch.jit.optimize_for_inference` — passes spécifiques inférence :
     **fusion Conv+BatchNorm** (les stats BN figées sont repliées dans les
     poids de la conv → une couche au lieu de deux), fusion Conv+ReLU via le
     fuser NNC, élimination de code mort.
- **Pourquoi** : récupérer les fusions « faciles » sans dépendre de Triton
  (contrairement à `torch.compile`) ni de TRT. Marche partout, y compris
  sur Windows.
- **Quand ça brille** : sur les architectures avec **beaucoup de blocs
  Conv→BN→Activation séquentiels** (typiquement les EfficientNet/BiFPN avec
  leurs SiLU et depthwise convs). Sur les torchvision pures (RetinaNet/FCOS),
  le gain est modeste car une partie du graphe est en code Python dynamique.

### `compile`
- **Quoi** : `torch.compile(model, backend="inductor", mode="default",
  dynamic=False)`. *Inductor* est le compilateur PyTorch 2 : il prend le
  graphe FX capturé par TorchDynamo et génère du code Triton (kernels CUDA
  spécialisés) avec fusion agressive d'opérations pointwise et de réductions.
- **Pourquoi** : compilation moderne, gains importants sur les modèles
  réguliers. `dynamic=False` car nos shapes sont fixées à 640×640.
- **Quand ça brille** : seul, **rarement** — sur les détecteurs il subit des
  recompilations en boucle à cause du NMS dynamique (décodage qui produit
  un nombre variable de boîtes par image). En combinaison FP16, c'est
  une autre histoire (voir `compile_fp16`).

### `cudagraphs`
- **Quoi** : `torch.compile(model, backend="cudagraphs")`. CUDA Graphs
  enregistre une séquence de lancements de kernels comme un objet unique
  qu'on peut « rejouer » en un seul appel API → suppression du *kernel
  launch overhead* (typiquement ~5-10 µs par kernel × des centaines de
  kernels = potentiel important sur les petits modèles).
- **Pourquoi** : sur GPU rapide, le coût du lancement Python+CUDA peut
  dominer le temps de calcul effectif pour les petits kernels.
- **Quand ça brille** : **architectures à shapes complètement statiques**.
  Sur les détecteurs, FCOS et RetinaNet utilisent des tenseurs CPU dans
  `anchor_generator.set_cell_anchors` et `_batched_nms_coordinate_trick`,
  ce qui **casse la capture** (cudagraphs refuse les tenseurs CPU).
  Voir §4 pour les régressions désastreuses.

### `compile_fp16`, `cudagraphs_fp16`, `torchscript_fp16`
- **Quoi** : autocast FP16 + une des trois compilations ci-dessus. L'ordre :
  on entre dans `torch.autocast`, puis le forward compilé s'exécute en FP16.
- **Pourquoi** : combiner *gain Tensor Cores* + *gain fusion/compilation*.
  La combinaison est presque toujours meilleure que chaque ingrédient seul.
- **Quand ça brille** : `compile_fp16` est **le winner historique du modèle
  complet** sur les torchvision. Idem ici (×2.13 FCOS, ×2.29 RetinaNet).

### `trt_fp16` (modèle complet)
- **Quoi** : `torch_tensorrt.compile(model,
  enabled_precisions={torch.float16})` sur le modèle complet. Sous le capot
  TRT-Dynamo : capture du graphe via Dynamo → partitionnement en sous-graphes
  TRT-compatibles vs eager → compilation des sous-graphes en moteurs TRT (qui
  contiennent les kernels CUDA natifs avec fusion Conv+BN+activation,
  sélection de tactiques optimales par benchmark interne, kernel cache).
- **Pourquoi** : c'est l'optimisation la plus puissante théoriquement —
  TRT recompile en kernels natifs spécialisés pour le GPU cible, alors que
  TorchScript reste dans le runtime PyTorch.
- **Quand ça brille** : sur **modèles à shapes purement statiques**. Sur
  RetinaNet/FCOS, **le NMS produit un nombre variable de détections** →
  shapes dynamiques avec borne supérieure non bornée (`Infinity`) → bug
  sympy → crash de la compilation TRT (voir §3.2).

### `zone_torchscript`, `zone_compile`, `zone_cudagraphs`, `zone_trt_fp16`
- **Quoi** : on isole une **zone statique** du modèle (typiquement
  backbone+FPN, parfois heads), on la passe par la compilation, et on laisse
  le post-traitement (NMS) en eager. Pour `zone_trt_fp16` c'est précisément
  ce que TRT demande : un sous-graphe à shapes fixes.
- **Pourquoi** : contourner le problème des shapes dynamiques du NMS, tout
  en récupérant le gain sur les ~80-90% du temps qui se passent dans le
  backbone et la tête de classification.
- **Quand ça brille** : c'est *la* voie TRT propre que l'encadrant demande
  — pas de plantage, AP parfaitement conservée, gain net.

### `zone_trt_folded`
- **Quoi** : avant TRT, on applique `torch.jit.freeze` pour propager les
  constantes (poids gelés → les multiplications par les coefficients du
  BiFPN deviennent des additions pondérées avec coefficients pré-calculés).
  Puis on passe à TRT.
- **Pourquoi** : TRT le demandait explicitement dans les logs du Run #1 —
  « *consider constant fold the model first* ». La fusion pondérée du
  BiFPN d'EfficientDet et certains patterns FPN+heads sont des cibles
  typiques.
- **Quand ça brille** : sur les architectures qui font des combinaisons
  pondérées de feature maps (BiFPN). Surprise du run : ça brille aussi sur
  **RetinaNet et FCOS** (×1.66 et ×1.93), preuve que le constant folding
  débloque autre chose que le BiFPN.

### `mixed_trt_bb__cudagraphs_rest`
- **Quoi** : TRT sur le backbone uniquement + cudagraphs sur le reste
  (FPN+heads). L'idée est d'utiliser TRT là où il excelle (convolutions
  régulières) et cudagraphs là où l'overhead de lancement domine (les
  centaines de petits kernels des heads).
- **Pourquoi** : combiner les forces. Mais les transitions entre régions
  optimisées coûtent (copies de buffers, synchronisations).
- **Quand ça brille** : potentiellement quand TRT seul n'arrive pas à
  compiler tout, et que cudagraphs marche sur la suite. En pratique ici,
  comparable à TRT seul (cf. §4).

### `zone_trt_int8` — non exécuté
- **Quoi prévu** : calibration INT8 sur 300 images, conversion des poids/
  activations en entier 8 bits via Post-Training Quantization (PTQ).
- **Pourquoi non exécuté** : flag `do_int8=False` dans la config du run.
  TRT-Dynamo en INT8 demande `modelopt` (NVIDIA TensorRT-Model-Optimizer)
  qui n'est pas installé dans l'image. À activer dans le prochain run si
  on veut explorer cette piste.

---

## 3. Conditions exactes du run

### 3.1. Environnement

- **Image Modal** : `debian_slim(python_version="3.13")` + pip install des
  dépendances *de zéro* (comme on ferait sur Colab).
- **Versions clés** validées par [`modal_test_env.py`](../modal_test_env.py) :
  - `torch 2.8.0+cu128`
  - `torch_tensorrt 2.8.0`
  - `tensorrt 10.12.0.36`
  - `numpy 2.4.6`
  - `cv2 4.13.0`
- **GPU** : T4 16 GB (Turing, capability 7.5, FP16 Tensor Cores, **pas** de
  BF16 ni INT8 Tensor Cores au niveau hardware).
- **Granularité** : 1 conteneur Modal par couple (modèle, variante) — pas de
  pollution d'état entre variantes.
- **Caches partagés** sur Volume Modal : `TORCH_HOME=/data/cache/torch`,
  `HF_HOME=/data/cache/hf` → poids torchvision/effdet téléchargés une seule
  fois et réutilisés.

### 3.2. Paramètres de bench

| Paramètre | Valeur | Rôle |
|---|---|---|
| `N_WARMUP` | 50 | Itérations de chauffe (compile + cuDNN autotune) |
| `N_MEASURE` | 1000 | Itérations chronométrées pour la moyenne |
| `N_PROFILE` | 150 | Itérations sous `torch.profiler` pour les tables d'ops |
| `N_PROFILE_DATA` | 2000 | Images pour le profilage |
| `N_EVAL` | 2000 | Images pour la MAP COCO |
| Image size | 640×640 | Fixe pour les 3 modèles |

### 3.3. Variantes ratées et leur cause exacte

Quatre variantes échouent. Les fichiers `errors/<model>_<variant>.txt` et
les logs correspondants donnent la cause racine :

| Variante | Cause | Recommandation TRT lue dans le log |
|---|---|---|
| `retinanet_r50_trt_fp16` | `AttributeError: 'Infinity' object has no attribute '_mpf_'` dans `torch_tensorrt/dynamo/utils.py::extract_var_range_info` | Le NMS produit des shapes avec borne sup = `oo` (sympy infinity). TRT demande explicitement de borner les shapes dynamiques avec `torch._dynamo.mark_dynamic` ou `torch.export.Dim(min=, max=)`. |
| `fcos_r50_trt_fp16` | Idem | Idem |
| `retinanet_r50_torchscript_fp16` | `RuntimeError: TorchScript : ni script ni trace n'ont abouti` | L'autocast wrapper (`torch.autocast`) casse la trace : le forward devient un Python object non-scriptable. **C'est le bon comportement** : depuis le Fix #1 (Run #1), TorchScript ne retourne plus silencieusement le modèle eager — il échoue franchement. |
| `fcos_r50_torchscript_fp16` | Idem | Idem |

Les warnings TRT les plus fréquents dans les logs des variantes TRT qui
**ont** marché (zones, EfficientDet full) :

- `Both operands of the binary elementwise op index_shape_X are constant.
  In this case, please consider constant fold the model first.` → motive
  la variante `zone_trt_folded`.
- `Unable to import quantization op. Please install modelopt library` →
  bloque INT8 ; à résoudre via `pip install nvidia-modelopt` au prochain run.
- `TensorRT-LLM is not installed` → sans impact ici, ignorable.

---

## 4. Tableau complet des résultats

Lecture : la colonne **xBase** est la vitesse relative au baseline du même
modèle (>1 = plus rapide). MAP COCO sur 2000 images de val2017, n'est calculée
que pour les variantes qui produisent encore les sorties détection compatibles
(les variantes `zone_*` non eval sont marquées `—`).

### 4.1. Tableau complet

| Modèle | Variante | ms | FPS | xBase | AP | Commentaire condensé |
|---|---|---:|---:|---:|---:|---|
| efficientdet_d4 | baseline | 109.17 | 9.2 | ×1.00 | 0.4477 | référence FP32 |
| efficientdet_d4 | fp16 | 162.66 | 6.1 | **×0.67** | 0.4477 | régression : autocast seul plombe |
| efficientdet_d4 | torchscript | 42.15 | 23.7 | ×2.59 | 0.4477 | très bon : fusion Conv+BN+SiLU paie |
| efficientdet_d4 | **torchscript_fp16** | **30.54** | **32.7** | **×3.57** | 0.4480 | **🏆 winner global** |
| efficientdet_d4 | cudagraphs | 58.58 | 17.1 | ×1.86 | — | OK, seul cas où cudagraphs paie |
| efficientdet_d4 | cudagraphs_fp16 | 43.93 | 22.8 | ×2.48 | 0.4478 | bien |
| efficientdet_d4 | trt_fp16 | 47.07 | 21.2 | ×2.32 | 0.4474 | **seul TRT full-model qui marche** |
| efficientdet_d4 | zone_torchscript | 41.76 | 23.9 | ×2.61 | — | aussi bien que torchscript full |
| efficientdet_d4 | zone_cudagraphs | 58.48 | 17.1 | ×1.87 | — | ≈ cudagraphs full |
| efficientdet_d4 | zone_trt_fp16 | 47.50 | 21.1 | ×2.30 | 0.4480 | **TRT zone, AP préservée** |
| efficientdet_d4 | mixed_trt_bb__cudagraphs_rest | 52.23 | 19.1 | ×2.09 | — | combine bien, pas mieux |
| fcos_r50 | baseline | 50.79 | 19.7 | ×1.00 | 0.3361 | référence |
| fcos_r50 | fp16 | 48.07 | 20.8 | ×1.06 | 0.3337 | marginal |
| fcos_r50 | compile | 46.06 | 21.7 | ×1.10 | — | recompile NMS limite le gain |
| fcos_r50 | **compile_fp16** | **23.87** | **41.9** | **×2.13** | 0.3336 | **🏆 winner FCOS** |
| fcos_r50 | cudagraphs | 160.57 | 6.2 | **×0.32** | — | catastrophe (capture impossible) |
| fcos_r50 | cudagraphs_fp16 | 174.94 | 5.7 | **×0.29** | — | idem, pire |
| fcos_r50 | torchscript | 41.34 | 24.2 | ×1.23 | 0.3361 | bon |
| fcos_r50 | torchscript_fp16 | — | — | — | — | FAILED (autocast ⇒ trace KO) |
| fcos_r50 | trt_fp16 | — | — | — | — | FAILED (shapes dyn. NMS) |
| fcos_r50 | zone_torchscript | 49.80 | 20.1 | ×1.02 | — | quasi neutre |
| fcos_r50 | zone_compile | 54.13 | 18.5 | ×0.94 | — | régresse |
| fcos_r50 | zone_cudagraphs | 61.19 | 16.3 | ×0.83 | — | régresse |
| fcos_r50 | zone_trt_fp16 | 40.46 | 24.7 | ×1.26 | 0.3360 | TRT zone OK, AP conservée |
| fcos_r50 | **zone_trt_folded** | **26.25** | **38.1** | **×1.93** | 0.3363 | **TRT propre, presque le winner** |
| fcos_r50 | mixed_trt_bb__cudagraphs_rest | 41.83 | 23.9 | ×1.21 | — | TRT bb + cg rest |
| retinanet_r50 | baseline | 49.75 | 20.1 | ×1.00 | 0.3775 | référence |
| retinanet_r50 | fp16 | 35.38 | 28.3 | ×1.41 | 0.3774 | autocast paye plus que sur FCOS |
| retinanet_r50 | compile | 53.41 | 18.7 | ×0.93 | — | recompile NMS |
| retinanet_r50 | **compile_fp16** | **21.68** | **46.1** | **×2.29** | 0.3774 | **🏆 winner RetinaNet** |
| retinanet_r50 | cudagraphs | 68.23 | 14.7 | ×0.73 | — | régresse |
| retinanet_r50 | cudagraphs_fp16 | 50.60 | 19.8 | ×0.98 | 0.3774 | neutre |
| retinanet_r50 | torchscript | 44.61 | 22.4 | ×1.12 | 0.3773 | modeste |
| retinanet_r50 | torchscript_fp16 | — | — | — | — | FAILED (autocast ⇒ trace KO) |
| retinanet_r50 | trt_fp16 | — | — | — | — | FAILED (shapes dyn. NMS) |
| retinanet_r50 | zone_torchscript | 46.31 | 21.6 | ×1.07 | — | marginal |
| retinanet_r50 | zone_compile | 53.42 | 18.7 | ×0.93 | — | régresse |
| retinanet_r50 | zone_cudagraphs | 56.38 | 17.7 | ×0.88 | — | régresse |
| retinanet_r50 | zone_trt_fp16 | 47.98 | 20.8 | ×1.04 | 0.3775 | TRT zone neutre |
| retinanet_r50 | **zone_trt_folded** | **29.98** | **33.4** | **×1.66** | 0.3777 | **TRT propre RetinaNet** |
| retinanet_r50 | mixed_trt_bb__cudagraphs_rest | 49.42 | 20.2 | ×1.01 | — | neutre |

### 4.2. Visualisation — Pareto MAP vs FPS

![Pareto MAP vs FPS](figures/map_vs_fps.png)

Trois bandes horizontales — une par famille — car la MAP varie très peu d'une
variante à l'autre (au pire ~3‰ d'écart). C'est attendu : aucune des
optimisations ici ne change la sémantique du modèle, sauf le passage en FP16
qui peut introduire des erreurs d'arrondi marginales. **L'objectif est donc
de pousser chaque famille le plus à droite possible sans descendre la MAP**.

Quelques lectures :
- **EfficientDet D4** (rouge) part de 9 FPS et atteint ~33 FPS avec
  `torchscript_fp16` — c'est le plus gros gain absolu, mais le modèle reste
  le plus lent des trois en valeur finale.
- **RetinaNet R50** (bleu) atteint **46 FPS** avec `compile_fp16` —
  championne des FPS du run, AP intacte (0.3775).
- **FCOS R50** (vert) plafonne autour de 42 FPS (`compile_fp16`) — légère
  baisse AP en FP16 (0.3336 vs 0.3361 baseline, soit −0.7%).
- Les variantes **`fp16` seules** (carrés) sont parfois *à gauche* du
  baseline — EfficientDet régresse à 6 FPS, FCOS stagne. Confirmation qu'il
  faut **toujours combiner FP16 avec une compilation**.

### 4.3. Speedup par variante et par famille

![Speedup par variante](figures/speedup_par_variante.png)

Tous les speedups normalisés par le baseline de chaque modèle (ligne
pointillée à ×1.0). Lecture rapide :
- Les barres **hachurées** marquent une régression (vitesse plus faible que
  le baseline). Quatre variantes régressent sur FCOS (`cudagraphs`,
  `cudagraphs_fp16`, `zone_compile`, `zone_cudagraphs`) — c'est le modèle le
  plus capricieux à optimiser.
- Les **FAIL** sont en rouge vertical : `trt_fp16` et `torchscript_fp16` sur
  les torchvision (voir §3.3 pour les causes).
- `compile_fp16` est la barre la plus haute pour les torchvision
  (×2.13 et ×2.29).
- `torchscript_fp16` domine sur EfficientDet (×3.57), suivi de
  `zone_torchscript` et `torchscript` (×2.6) — la famille TorchScript
  est clairement la plus adaptée à cette architecture.
- `zone_trt_folded` est le **TRT le plus propre** : il dépasse `zone_trt_fp16`
  sur les deux torchvision (×1.93 et ×1.66 vs ×1.26 et ×1.04). Confirme que
  le constant folding débloque réellement la compilation TRT.

### 4.4. Winners

| Modèle | Winner global | Speedup | Meilleur TRT « propre » | Speedup TRT |
|---|---|---:|---|---:|
| EfficientDet D4 | `torchscript_fp16` | ×3.57 | `zone_trt_fp16` ou `trt_fp16` | ×2.30–2.32 |
| FCOS R50 | `compile_fp16` | ×2.13 | `zone_trt_folded` | ×1.93 |
| RetinaNet R50 | `compile_fp16` | ×2.29 | `zone_trt_folded` | ×1.66 |

Observation transverse : **`compile_fp16` est le meilleur pour les
torchvision**, **`torchscript_fp16` pour effdet**. Aucune variante n'est
universellement supérieure, ce qui montre que la stratégie d'optimisation
doit suivre la structure du modèle.

---

## 5. Réponse à l'attente n°1 — TensorRT au niveau sous-module

L'encadrant demande : « *comparez le modèle avant et après accélération au
niveau des sous-modules, et montrez clairement d'où viennent les speed-ups* ».

On prend **EfficientDet D4** comme cas d'étude car c'est le seul modèle où
`trt_fp16` full-model fonctionne — donc on peut comparer baseline et TRT
sur la **même trace de profilage**.

### 5.1. Vue globale

| Mesure | Baseline | TRT FP16 | Gain |
|---|---:|---:|---:|
| Wall-clock moyen / image | 109.17 ms | 47.07 ms | **×2.32** |
| `self_cuda_us` total (profil 150 itér) | 13 957 ms | 8 602 ms | **×1.62** |
| AP COCO val2017 (2000 img) | 0.4477 | 0.4474 | −0.0003 |

Le wall-clock gagne plus que le CUDA pur (×2.32 vs ×1.62) : la différence
vient de la **suppression de l'overhead Python** (le runtime TRT exécute
le moteur en un appel C++ unique, là où PyTorch lançait des centaines de
kernels via Python).

### 5.2. Disparition des kernels baseline (top 10 par temps économisé)

Les chiffres sont en **ms cumulés sur 150 forward passes**, métrique
`self_cuda_us` extraite des [profils CSV](../outputs/20260616_122931/profiles).

| Opération baseline | Baseline (ms) | TRT (ms) | Économie | Cause de la disparition |
|---|---:|---:|---:|---|
| `aten::cudnn_convolution` | 1 517 | 0 | **−1 517** | Remplacée par des kernels TRT spécialisés (`trt_volta_hcudnn_*`, `sm70_xmma_fprop_*`) qui intègrent Conv+BN+activation dans un seul lancement, en FP16. |
| `aten::cudnn_batch_norm` | 989 | 0 | **−989** | **Fusion Conv+BN** : en inférence, BN est une transformation affine constante repliable dans les poids de la conv précédente. TRT fait ce folding systématiquement. |
| `bn_fw_inf_1C11_kernel_NCHW` (cuDNN BN forward) | 989 | 0 | **−989** | Idem ci-dessus, c'est le kernel concret derrière `cudnn_batch_norm`. |
| `aten::_conv_depthwise2d` | 960 | 0 | **−960** | Les depthwise convs (cœur d'EfficientNet) sont remplacées par des kernels TRT optimisés FP16 inclus dans `tensorrt::execute_engine`. |
| `volta_sgemm_128x64_nn` (matmul FP32 Volta) | 930 | 0 | **−930** | Remplacé par des tactiques FP16 (`trt_volta_hcudnn_128x128_relu_*`) qui exploitent les Tensor Cores. |
| `aten::silu_` + son kernel elementwise | 751 + 751 | 0 | **−1 502** | **Fusion conv+silu** : SiLU (Swish, `x * sigmoid(x)`) est fusionnée dans le kernel TRT précédent en tant que activation epilog. |
| `aten::cat` (concat des features BiFPN) | 517 | 0.7 | **−516** | TRT élimine la plupart des concats en réordonnant les writes des kernels producteurs vers la zone mémoire concaténée — pas de copie séparée. |
| `aten::mul` (broadcast pointwise) | 443 | 0 | **−443** | Fusionné dans `generatedNativePointwise` (kernel TRT généré pour des chaînes d'ops elementwise). |

### 5.3. Visualisation — top 10 opérations économisées

![Top ops économisées par TRT FP16 sur EfficientDet](figures/trt_op_savings_effdet.png)

Les barres bleues sont les baselines FP32, les rouges sont ce qu'il **reste**
après TRT FP16. On voit que pour la plupart des opérations dominantes du
baseline (`cudnn_convolution`, `cudnn_batch_norm`, `_conv_depthwise2d`,
`silu_`, `cat`), la barre rouge est **invisible** : TRT les a fait
complètement disparaître au profit de ses propres kernels fusionnés
(voir §5.3 ci-dessous). C'est ce qui produit le gain ×1.62 en CUDA pur.

### 5.4. Ce que TRT ajoute (ops nouvelles)

| Op TRT | Temps (ms) | Rôle |
|---|---:|---|
| `tensorrt::execute_engine` | 3 826 | Wrapper de lancement du moteur TRT (englobe tous les kernels du sous-graphe compilé) |
| `generatedNativePointwise` | 1 124 | Kernels CUDA générés par TRT pour des séquences pointwise fusionnées (silu, add, mul…) |
| `trt_volta_hcudnn_128x128_relu_*` | 526 | Convolutions FP16 Volta avec ReLU/SiLU fusionnée, tactique tuile 128×128 |
| `sm70_xmma_fprop_implicit_gemm_f32f32_f32f32_f32_*` | 222 | Convolutions implicit-GEMM SM 7.0 (Turing) |
| `trt_volta_scudnn_128x64_relu_*` | 180 | Variante de tuile 128×64 |
| `cuSliceLayer::naiveSlice` | 105 | Slicing des feature maps (TRT layer dédiée) |

**Lecture :** `tensorrt::execute_engine` (3.8 s sur 150 forwards) **remplace
à lui seul** tous les kernels Conv/BN/Activation listés à la §5.2
(≈ 7.7 s économisés). Le reste — `generatedNativePointwise` — absorbe les
fusions d'opérations elementwise.

### 5.5. Ce qui *reste bloquant*

Le rapport baseline→TRT n'est que de ×1.62 en CUDA pur alors qu'on aurait
pu espérer plus. Pourquoi ?

1. **Une partie du modèle reste en eager** : le partitionneur TRT laisse en
   PyTorch tout sous-graphe contenant des ops non supportées ou des shapes
   dynamiques. Dans le log on voit `Both operands of the binary elementwise
   op index_shape_X are constant. In this case, please consider constant
   fold the model first` répété ~30× — chaque occurrence est un endroit où
   TRT a renoncé à fuser parce que l'IR contenait des opérations sur des
   shapes constantes non foldées. **C'est précisément ce que résout
   `zone_trt_folded`** (mais zone_trt_folded n'est pas mesuré sur effdet
   car effdet utilise une autre voie d'isolation).
2. **Pas d'INT8** : sur Turing les Tensor Cores INT8 doublent encore le débit
   par rapport à FP16. Pas activé dans ce run.
3. **NMS reste en eager Python** — pour effdet ce n'est pas critique car la
   tête de détection est déjà au format dense.

---

## 6. Réponse à l'attente n°2 — opérations les plus accélérables

L'encadrant demande une **vue transversale** : quelles briques élémentaires
deviennent les plus efficaces, indépendamment de la famille de modèle.

### 6.1. Méthode

On agrège les profils de `self_cuda_us` des trois baselines (RetinaNet,
FCOS, EfficientDet) pour identifier où va le temps. On regarde ensuite
comment ces ops se comportent quand on applique TRT FP16 ou
`torchscript_fp16`.

### 6.2. Top des consommateurs (baselines, ms cumulés sur 3 modèles)

| Op baseline | Total ms | Famille |
|---|---:|---|
| `aten::cudnn_convolution` | 10 690 | **Convolution** — ~75% du temps total |
| `_5x_cudnn_volta_scudnn_winograd_128x128_*` | 5 037 | Variante Winograd (conv 3×3) |
| `volta_sgemm_128x64_nn` | 2 162 | GEMM FP32 |
| `aten::cudnn_batch_norm` | 1 501 | **BatchNorm** |
| `aten::add_`, `aten::mul`, elementwise variants | ~2 700 | **Pointwise** |
| `aten::_conv_depthwise2d` | 960 | **Depthwise conv** (effdet only) |
| `aten::clamp_min_` (ReLU) | 783 | **Activation** |

### 6.3. Le classement « accélérable » qui répond au prof

Sur la base des transitions baseline → TRT FP16 / torchscript_fp16
observées dans les profils :

| Brique | Speedup typique | Pourquoi elle marche bien |
|---|---|---|
| **Conv 3×3 → BN → ReLU/SiLU séquentielles** | ×2 à ×4 | Triple fusion : (a) BN repliée dans les poids de la conv, (b) activation fusionnée comme epilog du kernel conv, (c) exécution FP16 sur Tensor Cores. **C'est la brique reine.** |
| **Depthwise conv** | ×2 à ×3 | TRT a des tactiques FP16 spécialisées pour les depthwise (ratio compute/memory faible → bénéficie surtout des Tensor Cores). |
| **GEMM (linear / 1×1 conv)** | ×2 (FP16) à ×4 (INT8 si activé) | Tensor Cores. |
| **Chaînes de pointwise (mul, add, relu, sigmoid)** | ×3 à ×5 | TRT les fusionne dans un kernel unique généré → suppression du round-trip mémoire entre les ops. C'est aussi ce que fait `torch.compile`+inductor. |
| **Concat (`aten::cat`)** | ×100 à ×∞ (élimination) | TRT élimine en réorganisant les writes des producteurs. |

| Brique | Speedup faible / nul | Pourquoi |
|---|---|---|
| **NMS et post-traitement détection** | ×1 (reste en eager) | Shapes dynamiques (nombre variable de boîtes), boucles Python, tenseurs CPU intermédiaires. |
| **Anchor generation (RetinaNet/FCOS)** | régresse parfois | Utilise des tenseurs CPU → casse cudagraphs ; reste en eager même sous TRT. |
| **Code Python explicite dans le forward** | ×1 | Ce qui n'est pas dans le graphe FX ne peut pas être compilé. |

### 6.4. Briques sur lesquelles parier pour un futur design

Synthèse pour la suite (cf. §8) : un réseau pensé pour être **rapide en
production GPU** devrait privilégier :

1. **Blocs Conv→BN→Activation** réguliers (pas de branche conditionnelle).
2. **Activations simples et fusionnables** : ReLU, SiLU/Swish, GELU, Hardswish.
3. **Depthwise + Pointwise** (MBConv) plutôt que des convolutions 3×3 lourdes
   quand le compute n'est pas le goulot.
4. **Shapes statiques** sur toute la chaîne (éviter les Top-K dynamiques,
   les sélections par seuil, les NMS en cœur de réseau).
5. **Pas de tenseurs CPU intermédiaires** (anchors, grids, etc. doivent être
   pré-calculés en GPU ou registrés comme buffers).
6. **Concat structurées** (en aval d'opérations TRT-friendly).

---

## 7. Réponse à l'attente n°3 — ouverture vers la suite

Dans de futurs travaux, on peut imaginer **construire un réseau de neurones
de détection en assemblant directement les briques que la §6 a identifiées
comme les plus accélérables**. Concrètement :

- **Backbone** type EfficientNet ou MobileNetV3 : empilement de MBConv
  (depthwise + pointwise + SiLU + SE-block) — chacune de ces sous-couches
  est dans le top du tableau de §6.3.
- **Neck** statique : un FPN simple (sans BiFPN à coefficients learnables
  dynamiques) ou un BiFPN avec coefficients **gelés à l'inférence** pour
  permettre le constant folding.
- **Tête de détection dense** type FCOS ou CenterNet, avec **borne supérieure
  fixe** sur le nombre de détections post-NMS (paddé à K = 100 par exemple)
  pour que tout reste en shapes statiques.
- **NMS** GPU-natif et statique (`torchvision.ops.batched_nms` avec un Top-K
  fixe), ou intégré comme plugin TRT.

Ce design garantit que **>95% du graphe soit compilable en un unique moteur
TRT FP16**, et serait un excellent terrain pour ajouter ensuite INT8 PTQ
(gain attendu ×1.5–2 supplémentaire sur Turing/Ampere).

---

## 8. Prochaines étapes concrètes

### 8.1. Pour le rendu du 24 juin

- [ ] Régler `do_int8=True` et installer `nvidia-modelopt` dans l'image
  Modal, puis relancer `zone_trt_int8` sur les 3 modèles. **Gain attendu**
  ×1.5–2 supplémentaire par rapport à FP16.
- [ ] Borner les shapes dynamiques des torchvision via
  `torch._dynamo.mark_dynamic` ou `torch.export.Dim(min=1, max=300)` sur
  la sortie du NMS, et **réessayer `trt_fp16` full-model** sur RetinaNet
  et FCOS. C'est ce que TRT réclame explicitement dans les logs.
- [ ] Relancer `torchscript_fp16` sur les 3 modèles avec une enveloppe
  autocast adaptée (cast manuel des inputs en FP16 avant le trace plutôt
  que `torch.autocast` qui casse la trace). Devrait débloquer le winner
  sur FCOS/RetinaNet par symétrie avec EfficientDet.

### 8.2. Pour la présentation

- Centrer le narratif autour de la **§5** (TRT au niveau sous-module sur
  EfficientDet) — c'est la demande la plus concrète de l'encadrant.
- Montrer le **tableau condensé §4.2** (winners) et expliquer pourquoi
  `torchscript_fp16` domine sur effdet (BiFPN + SiLU fusionnables) tandis
  que `compile_fp16` domine sur torchvision (gestion du NMS par Inductor
  meilleure que TorchScript).
- Conclure par la **§6.3** (briques accélérables) et la **§7** (design
  futur) pour répondre au point 3 du mail.

### 8.3. Pistes pour plus tard

- Réécrire `FpnCombine.forward` du BiFPN pour pré-calculer
  `relu(w)/Σw` en eval → coefficients constants → fusion possible sans
  passer par `jit.freeze`. Rendrait `trt_fp16` propre sans astuce.
- Étudier les profilers (CSV `profiles/`) avec un script qui extrait
  automatiquement les opérations dont la part baisse / monte de plus
  de X% en TRT vs baseline — automatiser §6.
- Tester `torch.export` (le futur de Dynamo) qui résout proprement le
  problème des shapes dynamiques avec `Dim`.

---

## 9. Où trouver quoi

| Question | Fichier |
|---|---|
| Tableau brut (37 lignes) | aggrégé dans la §4.1 ci-dessus depuis `outputs/20260616_122931/bench/*.json` |
| Logs complets par variante | [`outputs/20260616_122931/logs/`](../outputs/20260616_122931/logs) |
| Tracebacks des FAIL | [`outputs/20260616_122931/errors/`](../outputs/20260616_122931/errors) |
| Profil sous-module (baseline et fp16) | [`outputs/20260616_122931/modules/`](../outputs/20260616_122931/modules) |
| Profil par opération (toutes variantes) | [`outputs/20260616_122931/profiles/`](../outputs/20260616_122931/profiles) |
| Métriques MAP COCO | [`outputs/20260616_122931/eval/`](../outputs/20260616_122931/eval) |
| Log d'orchestration | [`outputs/20260616_122931/run.log`](../outputs/20260616_122931/run.log) |
