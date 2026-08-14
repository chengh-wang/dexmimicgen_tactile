# Drawer Cleanup B Residual Sweep - 2026-08-02

## Context

Task: `drawer_cleanup`

Experiment: B residual world-model action sweep, 50 evaluation episodes per setting.

Output root:

```text
outputs/drawer_cleanup_B_swap_50ep_same_setup_20260802
```

The old 8-policy flow-matching training/eval goal is no longer being pursued unless explicitly requested again.

## Results

| Rank | Setting | Success |
|---:|---|---:|
| 1 | `basis4_l21e-03` | **47/50 = 94%** |
| 2 | `basis6_l21e-04` | 43/50 = 86% |
| 2 | `basis8_l21e-03` | 43/50 = 86% |
| 4 | `basis10_l21e-03` | 42/50 = 84% |
| 4 | `basis10_l21e-04` | 42/50 = 84% |
| 6 | `basis6_l21e-03` | 41/50 = 82% |
| 6 | `basis8_l21e-04` | 41/50 = 82% |
| 6 | `basis12_l21e-03` | 41/50 = 82% |
| 9 | `basis12_l21e-04` | 40/50 = 80% |
| 10 | `basis4_l21e-04` | 36/50 = 72% |

## Conclusion

Best setting:

```text
basis4_l21e-03: 47/50 = 94%
```

The strongest result came from the smallest tested basis count with the stronger regularization setting in this sweep. Larger basis counts were still strong but did not beat `basis4_l21e-03`; most landed in the 80-86% range.

## Notes

- The final local 02 service for this sweep finished and became inactive.
- Confirmed local completed summaries for:
  - `02_B_basis4_l21e-03`
  - `02_B_basis4_l21e-04`
  - `02_B_basis6_l21e-03`
  - `02_B_basis6_l21e-04`
- Remaining table entries were the completed remote sweep results reported during the same run.
