# CODE_STYLE — Order Book Mind Reader

## 1. Naming

| Thing | Convention | Example |
|---|---|---|
| Modules | `snake_case`, noun | `labels.py`, `baselines.py` |
| Functions | `snake_case`, verb | `event_time_labels`, `block_bootstrap_diff` |
| Classes | `PascalCase` | `StoikovMicroprice`, `Normalizer`, `BookTransformer` |
| Constants | `UPPER_SNAKE_CASE` | `FEATURE_COLUMNS`, `EMBARGO` |
| Frozen values | `config.py` or `configs/base.yaml` | `CONTEXT`, `FEE_BPS` |
| Fold labels | `foldK: train[0:a] test[b:c]` | printed, not parsed |

Standard ML shorthand is allowed: `x`, `y`, `p`, `auc`, `bps`. No other
abbreviations.

## 2. Formatting

- `ruff format` (100 cols), `ruff check`.
- Import groups: stdlib / third-party / local, one blank line between.
- `from __future__ import annotations` at the top of every module.
- Optional heavy imports (`torch`, `lightgbm`, `arch`) go **inside** the
  function that needs them, or behind a guarded module-level try, so that
  importing the package never requires them.

## 3. Types

- Full annotations on public functions.
- `np.ndarray` / `pd.Series` / `pd.DataFrame` in signatures, never `list[float]`
  for array data.
- Frozen dataclasses for records: `LabelSet`, `Score`, `Fold`, `ModelConfig`.
- No `Any`.

## 4. Structure

- `features.py` must never import `labels.py`. Direction of information is the
  one structural guarantee against leakage that no reviewer has to trust.
- `evaluate.py` computes numbers; `config.py` holds the thresholds. A metric
  module that also decides pass/fail is a metric module with an opinion.
- `scripts/` wires stages and prints; it holds no formulas.
- Functions ≤ 40 lines except the fold loops, which are linear and readable.

## 5. Comments and docstrings

State the *why* and the units. Examples of the required style:

```python
# shift(1): at row t we know the previous move, not the one after this event.
```

```python
def edge_in_bps(y, p, forward_move_bps, threshold=0.5):
    """Signed mid move captured per traded event, in basis points of mid.

    Only events the model wants are counted: p > threshold goes long,
    p < 1 - threshold goes short, everything else is untraded.
    """
```

Negative results get a docstring too. `verdict()` returning "predictable but
below the cost line" is a documented outcome, not an error path.

## 6. Errors

- Name the fixing command in the message.
- Never swallow an exception; the archive reader reports dropped rows, it does
  not hide them.
- A missing optional dependency is a clean exit 1 with the install line.

## 7. Randomness and determinism

- `np.random.default_rng(config.SEED)`.
- Torch: `torch.manual_seed(seed)`, `num_workers=0`, no `cudnn.benchmark`.
- Any new stochastic step must take a `seed` argument.

## 8. Git

`feat:` `fix:` `docs:` `test:` `data:` `chore:`.
`Freeze pre-registration` must precede the first `src/` commit; changes to the
bar afterwards are `RE-REGISTER: <reason>`.
