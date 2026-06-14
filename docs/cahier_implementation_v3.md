# Cahier d'implémentation — Optimisation par zone et sous-zone (v3)

> Statut : **à valider**. Une fois validé, implémentation dans l'ordre du §12.
> Priorité directrice : **gain maximal pour effort minimal**.

---

## 0. Objet

Étendre le pipeline d'optimisation actuel (`optimizations/`) pour :

1. **Optimiser aussi les têtes** (aujourd'hui laissées en eager côté torchvision).
2. Introduire l'**optimisation par sous-zone** : appliquer à chaque sous-module
   l'outil adapté à *son* architecture (le module qui pose souci reçoit un
   traitement spécifique).
3. Ajouter pour le BiFPN les **deux voies** décidées :
   - **cudagraphs** (contourne la fusion — déjà en place),
   - **constant-fold + TensorRT** (restaure la fusion — recommandé par TRT).

Le tout sans casser l'existant : les variantes actuelles restent valides.

---

## 1. Acquis validés (ne pas refaire)

| Fait établi | Preuve |
|---|---|
| L'optimisation par **zone statique** (backbone) débloque le gain | cudagraphs R50 ×1.71 vs ×0.96 sur modèle complet |
| Le **NMS** doit rester eager (shapes dynamiques) | recompilations / `size mismatch` sur modèle complet |
| **FP16** = optimisation **modèle complet** (autocast), pas zone | fuite FP16 → tête FP32 → mismatch de dtype |
| **FP16 pénalise effdet** (depthwise, pas de Tensor Cores) | ×0.79 mesuré sur D4 |
| **cudagraphs** est le bon levier pour effdet (petits kernels) | signal ×4.82 (bruité) sur D4 |
| Le **resample** gêne `inductor` (sympy), **pas** TRT | logs sympy ; TRT supporte `Resize` |
| Le **frein TRT du BiFPN** = la **fusion pondérée** (op non standard) | `FpnCombine` + warnings « consider constant fold » |
| Le profiler doit **retourner** `prof`, sauvegarde dans l'appelant | trace introuvable sinon ; fait dans le runner |

**Structure des modèles (vérifiée) :**

```
torchvision (R50, FCOS)          effdet (D4/D5/D6)
model.backbone : BackboneWithFPN  model.model : EfficientDet
  .body  (ResNet)                   .backbone (EfficientNet)
  .fpn   (FPN)                      .fpn      (BiFPN)
model.head                          .class_net
  .classification_head              .box_net
  .regression_head                model.anchors
model.transform, .anchor_generator
→ décodage + NMS = dynamique        → décodage + NMS = dynamique
```

---

## 2. Principes de conception (les règles)

1. **Frontière statique/dynamique** : tout ce qui précède le décodage/NMS est
   statique à 640×640 → optimisable. Le décodage + NMS reste **toujours** eager.
   La zone statique = `backbone + FPN + têtes` (les têtes incluses).
2. **Entrée verrouillée à 640×640** : toute optimisation suppose cette taille fixe.
   Hors de cette taille → rupture (dure pour cudagraphs/TRT, recompilation pour compile).
3. **Par sous-zone** : on peut remplacer indépendamment `backbone`, `fpn`, `head`.
   On affecte à chacun l'outil adapté à son architecture (cf. §5.2).
4. **Pas d'imbrication de `torch.compile`** : les sous-modules optimisés doivent
   être **disjoints** (jamais TRT *dans* un cudagraph). Glue eager entre eux.
5. **Robustesse** : chaque optimisation est tentée en try/except dans le runner.
   Un échec d'une sous-zone → log + fallback eager pour cette sous-zone, le reste continue.
6. **Symétrie des familles** : après ces travaux, torchvision et effdet optimisent
   tous deux `backbone + fpn + têtes` (aujourd'hui seul effdet inclut les têtes).

---

## 3. Travaux — vue d'ensemble

| Tâche | Intitulé | Gain | Effort | Dépend de |
|---|---|---|---|---|
| **A** | Accès sous-zone + zone élargie aux têtes | moyen-élevé | faible | — |
| **B** | Variante mixte `mixed_trt_bb__cudagraphs_rest` | à mesurer | moyen | A |
| **C** | `zone_trt_folded` (constant-fold + TRT) | élevé (effdet) | moyen | — |
| **D** | `ort_full` (NMS dynamique en ONNX Runtime) | test ciblé | moyen | — (optionnel) |

Ordre retenu : **A → C → B → D** (A débloque B ; C est la pièce « anales » indépendante).

---

## 4. Tâche A — Accès sous-zone + zone élargie aux têtes

### 4.1 Cartographie des sous-zones (`zones.py`)

```python
SUBZONES = {
    "torchvision": ["backbone", "fpn", "head"],
    "effdet":      ["backbone", "fpn", "class_net", "box_net"],
}
```

### 4.2 API à ajouter dans `zones.py`

```python
def get_subzone(model, family, name) -> tuple[nn.Module, Callable]:
    """Retourne (sous_module, setter) pour une sous-zone nommée.
    torchvision: backbone→model.backbone.body, fpn→model.backbone.fpn, head→model.head
    effdet:      backbone→model.model.backbone, fpn→model.model.fpn,
                 class_net→model.model.class_net, box_net→model.model.box_net
    """

def capture_subzone_inputs(model, family, device, size=(640,640)) -> dict[str, tuple]:
    """Fait passer UN dummy 640×640 dans le modèle complet et capture, via des
    forward_pre_hooks, l'entrée réelle de chaque sous-zone (pour le tracing/TRT
    qui exigent un exemple). Retourne {name: (args...)}.
    - dummy torchvision : [torch.zeros(3,H,W)]   (List[Tensor])
    - dummy effdet      : torch.zeros(1,3,H,W)
    Les hooks sont retirés en fin d'appel.
    """

def apply_subzone_plan(model, family, plan: dict[str, Callable|None],
                       ctx: dict, device="cuda", size=(640,640)) -> nn.Module:
    """Applique un plan {sous_zone: optimiseur ou None}.
    None = laisser la sous-zone en eager.
    - Capture les entrées intermédiaires UNIQUEMENT si un optimiseur en a besoin
      (torchscript/TRT). cudagraphs/compile n'ont pas besoin d'exemple.
    - Remplace chaque sous-module in-place via son setter.
    Retourne le modèle (même API).
    """
```

### 4.3 Besoin d'exemple par optimiseur

| Optimiseur | Exemple requis ? | Source de l'exemple |
|---|---|---|
| `opt_cudagraphs` | non (capture au 1er appel) | — |
| `opt_compile` | non | — |
| `opt_torchscript` (trace) | **oui** | `capture_subzone_inputs` |
| `opt_trt_fp16` | **oui** | `capture_subzone_inputs` |
| `opt_trt_int8` | oui + calib | idem + calib loader |

→ `apply_subzone_plan` n'appelle `capture_subzone_inputs` que si au moins un
optimiseur du plan est dans la liste « exemple requis ».

### 4.4 Zone élargie aux têtes — intégration

`apply_zone_optimization` (existant) devient un cas particulier de
`apply_subzone_plan` : **le même optimiseur** est appliqué à toutes les sous-zones
statiques, têtes comprises.

```python
def apply_zone_optimization(model, family, optimizer, ctx, device, size,
                            include_heads=True):
    """Applique `optimizer` à backbone + fpn (+ têtes si include_heads).
    Implémenté via apply_subzone_plan avec le même optimiseur partout.
    include_heads=True par défaut → symétrie torchvision/effdet."""
    names = list(SUBZONES[family])
    if not include_heads:
        names = [n for n in names if n not in ("head","class_net","box_net")]
    plan = {n: optimizer for n in names}
    return apply_subzone_plan(model, family, plan, ctx, device, size)
```

> Conséquence : les variantes `zone_*` existantes optimisent désormais **aussi les
> têtes** côté torchvision. C'est l'amélioration « zone élargie ».

### 4.5 Critères d'acceptation (tâche A)

- [ ] `get_subzone` retourne le bon module pour les 6 sous-zones (3 tv + ... effdet).
- [ ] `capture_subzone_inputs` renvoie une entrée non vide pour chaque sous-zone.
- [ ] `apply_zone_optimization(..., include_heads=True)` sur R50 produit une sortie
      `List[Dict]` correcte (test : forward sur 1 image dummy).
- [ ] `zone_cudagraphs` sur R50 avec têtes incluses : pas de crash, speedup ≥ version backbone-seul.
- [ ] Tests locaux : cudagraphs + torchscript ; **risque connu** : tracing d'une tête
      à entrée `dict` (RetinaNetHead). Si échec → fallback tête eager (try/except runner).

---

## 5. Tâche B — Variante mixte `mixed_trt_bb__cudagraphs_rest`

### 5.1 Design

```
entrée 640 → [TRT FP16] backbone → [cudagraphs] fpn + têtes → [eager] décodage+NMS
```

**Astuce clé** : TRT n'est appliqué qu'au **backbone** (entrée connue = `[1,3,640,640]`,
pas de capture nécessaire) ; le reste (fpn + têtes) part en **cudagraphs**
(pas d'exemple requis). → la variante mixte **n'a pas besoin** de
`capture_subzone_inputs`. Implémentation directe.

### 5.2 Plan d'optimisation par famille (le « bon outil par module »)

```python
PLAN_MIXED = {
  "torchvision": {"backbone": opt_trt_fp16, "fpn": opt_cudagraphs, "head": opt_cudagraphs},
  "effdet":      {"backbone": opt_trt_fp16, "fpn": opt_cudagraphs,
                  "class_net": opt_cudagraphs, "box_net": opt_cudagraphs},
}
```

> Pour torchvision, `backbone` = `model.backbone.body` (le ResNet seul) ; le FPN
> part en cudagraphs avec les têtes.

### 5.3 Builder runner

```python
def build_mixed_trt_cudagraphs(model, mspec, ctx):
    from optimizations.zones import apply_subzone_plan, opt_trt_fp16, opt_cudagraphs
    plan = PLAN_MIXED[mspec.family]
    return apply_subzone_plan(model, mspec.family, plan, _zone_ctx(ctx),
                              ctx.config.device, ctx.config.size)
```

Variante : `VariantSpec("mixed_trt_bb__cudagraphs_rest", build_mixed_trt_cudagraphs,
do_map=True, with_modules=False, profile=True, requires="trt")`.

### 5.4 Critères d'acceptation (tâche B)

- [ ] La variante tourne sur Colab (TRT) sans crash, sortie correcte.
- [ ] On compare dans les chiffres : `zone_cudagraphs` (tout cudagraphs) vs
      `zone_trt_fp16` (tout TRT) vs `mixed` → tableau de décision par famille.
- [ ] En local (sans TRT) : la variante est **SKIPPED** proprement (`requires="trt"`).

---

## 6. Tâche C — `zone_trt_folded` (constant-fold + TensorRT)

### 6.1 Le mécanisme

À l'inférence, les poids du BiFPN sont gelés → `relu(w_i)/(ε+Σrelu(w_j))` sont des
**constantes**. Les replier transforme la fusion pondérée en **addition pondérée
par constantes**, que TRT sait fusionner. C'est ce que réclamaient les warnings
TRT (« consider constant fold the model first »).

### 6.2 Voie C1 — `freeze` + TRT frontend TorchScript (primaire, faible effort)

`torch.jit.freeze` fait de la **propagation de constantes** : il traite les
paramètres gelés comme constantes et replie `relu(param)`, la somme et la division.

```python
def opt_trt_fp16_folded(zone, ex, ctx):
    import torch_tensorrt
    zone.eval()
    scripted = torch.jit.trace(zone, ex, strict=False)
    frozen   = torch.jit.freeze(scripted)         # ← constant-fold
    return torch_tensorrt.compile(
        frozen, ir="torchscript", inputs=[ex],
        enabled_precisions={torch.float16},
        truncate_long_and_double=True,
    )
```

- **Exemple requis** → via `capture_subzone_inputs` (zone effdet = `model.model`,
  entrée = `[1,3,640,640]`, donc connue ; pas de capture nécessaire si on l'applique
  à la zone entière `model.model`).
- Drop-in : reste en monde torch.

### 6.3 Voie C2 — ONNX + `onnxsim` + TRT (production, optionnelle)

```
export ONNX (model.model, 640 fixe) → onnxsim (simplify + constant fold)
  → TensorRT engine (via torch_tensorrt ONNX frontend OU ORT-TRT EP)
```

- `onnxsim` fait un constant-fold plus agressif que `freeze`.
- Intégration drop-in plus lourde (wrapper ORT/engine). **À ne faire que si C1
  ne suffit pas** (mesure de la fusion via le profiler avant/après).
- Référence : sample officiel NVIDIA EfficientDet (graph-surgeon + EfficientNMS).

### 6.4 Critères d'acceptation (tâche C)

- [ ] `zone_trt_folded` (C1) tourne sur Colab, sortie correcte, MAP@640 ≈ `zone_trt_fp16`.
- [ ] **Mesure clé** : via le profiler avant/après, le **nombre d'opérations
      distinctes diminue** et le temps GPU du BiFPN baisse vs `zone_trt_fp16` non foldé.
      → c'est la preuve que le constant-fold a restauré la fusion.
- [ ] Comparaison effdet : `zone_cudagraphs` vs `zone_trt_fp16` vs `zone_trt_folded`.

---

## 7. Tâche D — `ort_full` (optionnel)

Export du **modèle complet** (NMS inclus) en ONNX → ONNX Runtime CUDA EP.
Teste si un runtime à **shapes dynamiques** passe le mur du NMS là où TRT bute.

```python
def build_ort_full(model, mspec, ctx):
    # 1. export_full_detection(model, onnx_path, image_size=(640,640))
    # 2. ORTModelFull(onnx_path) : wrapper qui expose la même API (List[Dict] / Tensor)
    ...
```

- **Risque** : l'export ONNX d'un détecteur torchvision complet (avec NMS) est
  délicat ; effdet via `DetBenchPredict` aussi. → variante marquée *expérimentale*,
  `requires="ort"`, fallback propre si l'export échoue.
- Bench seulement (la comparaison MAP nécessiterait un postprocess dédié) → `do_map=False`.

---

## 8. Jeu de variantes final

| Variante | backbone | fpn | têtes | NMS | MAP | Profil | requires | Familles |
|---|---|---|---|---|:---:|:---:|---|---|
| `baseline` | — | — | — | eager | ✓ | ✓ | — | toutes |
| `fp16` | autocast modèle complet | ✓ | ✓ | cuda | toutes |
| `zone_torchscript` | TS | TS | TS | eager | ✗ | ✓ | — | toutes |
| `zone_compile` | compile | compile | compile | eager | ✗ | ✓ | compile | tv (effdet opt-in) |
| `zone_cudagraphs` | cg | cg | cg | eager | ✗ | ✓ | cuda | toutes ⭐ |
| `zone_trt_fp16` | TRT | TRT | TRT | eager | ✓ | ✓ | trt | toutes |
| `zone_trt_int8` | TRT-int8 | … | … | eager | ✓ | ✓ | trt+int8 | toutes (opt) |
| **`mixed_trt_bb__cudagraphs_rest`** | TRT | cg | cg | eager | ✓ | ✓ | trt | toutes ⭐ |
| **`zone_trt_folded`** | TRT(foldé) | TRT(foldé) | TRT(foldé) | eager | ✓ | ✓ | trt | effdet ⭐ |
| `ort_full` *(opt)* | ONNX Runtime modèle complet | — | ✗ | ✓ | ort | toutes |

(cg = cudagraphs ; ⭐ = pièces à fort intérêt pour le rapport)

**Affectation par famille (notebook) :**

```python
VARIANTS_TV     = [baseline, fp16, zone_torchscript, zone_compile, zone_cudagraphs,
                   zone_trt_fp16, mixed_trt_bb__cudagraphs_rest, zone_trt_int8(opt)]
VARIANTS_EFFDET = [baseline, fp16, zone_torchscript, zone_cudagraphs,
                   zone_trt_fp16, zone_trt_folded, mixed_trt_bb__cudagraphs_rest,
                   zone_trt_int8(opt)]   # zone_compile opt-in
```

---

## 9. Modifications par fichier

| Fichier | Modification |
|---|---|
| `optimizations/zones.py` | + `SUBZONES`, `get_subzone`, `capture_subzone_inputs`, `apply_subzone_plan`, `opt_trt_fp16_folded` ; `apply_zone_optimization` réécrit via plan + `include_heads` |
| `optimizations/runner.py` | + builders `build_mixed_trt_cudagraphs`, `build_zone_trt_folded`, (opt) `build_ort_full` ; + variantes dans `DEFAULT_VARIANTS` ; + `PLAN_MIXED` |
| `optimizations/ort_inference.py` | (tâche D) + `export_full_detection` usage + wrapper `ORTModelFull` |
| `optimizations/__init__.py` | exports des nouvelles fonctions |
| `optimization_full.ipynb` | listes `VARIANTS_TV`/`VARIANTS_EFFDET` mises à jour ; cellule d'analyse comparative `cudagraphs vs trt vs folded vs mixed` |
| `docs/cahier_implementation_v3.md` | ce document |

---

## 10. Plan de tests

**Local (Windows, conda `base`, CUDA, SANS TRT) :**
- [ ] `get_subzone` : 6 sous-zones résolues correctement.
- [ ] `capture_subzone_inputs` : entrées capturées (R50 + D4).
- [ ] `apply_zone_optimization(include_heads=True)` cudagraphs sur R50/FCOS/D4 :
      forward correct + speedup mesuré.
- [ ] tracing de la tête (`opt_torchscript` sur `head`) : OK ou fallback propre.
- [ ] `mixed_*` partie cudagraphs (backbone laissé eager en local faute de TRT) : forward OK.

**Colab (Linux, TRT) :**
- [ ] `mixed_trt_bb__cudagraphs_rest` : forward + MAP@640 + profil.
- [ ] `zone_trt_folded` (C1) : forward + MAP@640 + **profil avant/après** (preuve de fusion).
- [ ] (opt) `ort_full`.

Chaque test : un petit script `python -u` via `conda` base, n_iter réduit, sortie filtrée.

---

## 11. Risques & décisions ouvertes

| Risque | Mitigation |
|---|---|
| Tracing/TRT d'une tête à entrée **dict** (RetinaNetHead) échoue | fallback tête eager (try/except runner) ; cudagraphs/compile sur la tête n'ont pas ce souci |
| `freeze` ne replie pas **tout** le BiFPN | mesurer au profiler ; si insuffisant → voie C2 (onnxsim) |
| Coût des **frontières** dans `mixed` annule le gain | mesurer ; si négatif → conclure « cudagraphs-tout gagne » (résultat valable) |
| Imbrication accidentelle de `torch.compile` | garantir des sous-modules **disjoints** dans le plan |
| Capture d'entrée fausse si le dummy ne suit pas le bon chemin | utiliser l'input réel du pipeline (collate du modèle) pour le dummy |

**Décisions ouvertes à trancher avec toi :**
1. `ort_full` (tâche D) : on l'inclut ou on le garde pour plus tard ?
2. Pour `zone_trt_folded` : on se limite à C1 (freeze) d'abord, C2 (onnxsim) seulement si C1 insuffisant — OK ?
3. La zone élargie aux têtes : on l'active **par défaut** sur toutes les variantes
   `zone_*` (symétrie), ou on garde une variante `zone_*_no_head` pour comparer
   l'apport des têtes ? (je penche pour : têtes incluses par défaut + 1 mesure
   ponctuelle backbone-seul sur R50 pour quantifier l'apport.)

---

## 12. Ordre d'implémentation & jalons

1. **Jalon 1 — Tâche A** : sous-zones + zone élargie aux têtes.
   Test local cudagraphs/TS. → fige l'API sous-zone.
2. **Jalon 2 — Tâche C (C1)** : `zone_trt_folded` (freeze + TRT TS).
   Test Colab + profil avant/après (preuve de fusion).
3. **Jalon 3 — Tâche B** : variante mixte. Test Colab + tableau comparatif.
4. **Jalon 4 — Tâche D (si retenue)** : `ort_full`.
5. **Jalon 5** : mise à jour notebook + cellule d'analyse comparative finale.

Après chaque jalon : test isolé (local ou Colab selon dispo), puis intégration runner.
