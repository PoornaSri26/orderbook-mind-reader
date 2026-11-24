# DESIGN_SYSTEM — Order Book Mind Reader

The visual deliverables are: a **baseline-vs-model table in bp**, an
**imbalance-response curve** (model vs micro-price), and a **patching heatmap**.
All three must be legible in a PDF and in a GitHub README.

## 1. Colour tokens (`src/micro/style.py`)

| Token | Hex | Use |
|---|---|---|
| `ink` | `#111318` | text, axes |
| `grid` | `#D6DAE1` | gridlines, 0.6 pt |
| `baseline` | `#6B7280` | every baseline series (they are one comparison class) |
| `model` | `#3B6EA5` | the transformer, always this colour |
| `microprice` | `#8A5A00` | Stoikov micro-price series |
| `pass` | `#2E7D32` | bar met |
| `fail` | `#B3261E` | bar missed |
| `muted` | `#6B7280` | annotations |

Rules: the model is always blue, baselines always grey, the micro-price always
amber — in every figure, in every report. A reader should recognise the series
without the legend.

## 2. Figure geometry

| Figure | Size (in) | Notes |
|---|---|---|
| baseline table | — | markdown table, not an image |
| imbalance-response | 6.5 × 4.2 | x = imbalance bucket, y = P(up) |
| patching heatmap | 5.0 × 4.0 | layers × tokens, single-hue ramp |
| calibration | 5.0 × 5.0 | 45° reference line, error bars per bin |

DPI 160 (PNG), 300 (print). Font 9 pt ticks, 11 pt bold title. Top/right spines
removed. No figure-level frame.

## 3. Axis conventions

- **Imbalance-response:** x is the imbalance bucket centre in [0,1], y is
  P(up) in [0,1]; plot the 45° "no information" line at 0.5.
- **Calibration:** x predicted probability, y realised frequency, with Wilson
  intervals per bin; the diagonal is the reference.
- **Patching:** x = token position (relative to the prediction token), y =
  layer; colour = recovery in [−1, 1] on a diverging ramp centred at 0.
- Every axis label carries units. `bp` for costs and edges, `ticks` for spread,
  `nats` for log-loss.

## 4. Required annotations

1. sample size (`n = 412,880 events`, `5 folds`);
2. the cost line drawn as a dashed horizontal rule with its value, on every
   edge/bp chart;
3. a title that states the conclusion ("the gain is a fifth of the spread", not
   "log-loss by model").

## 5. Tables

- Models in a fixed order: `imbalance_logit`, `weighted_mid`, `stoikov`,
  `lightgbm`, `transformer` (last, so it has to earn its place at the bottom).
- The bar row is bolded, never coloured.
- Intervals in square brackets, three significant figures.

## 6. Accessibility

- [ ] Greyscale-safe: series distinguishable by dash pattern and marker too.
- [ ] Heatmap uses a perceptually uniform ramp; never rainbow.
- [ ] Contrast: ink on white ≥ 7:1.
- [ ] Alt text states the conclusion, plus `n` and the folds.

## 7. Implementation

`micro.style.apply_style()` is called at the top of every plotting script;
`micro.style.TOKENS` is the only place a hex value may appear.
