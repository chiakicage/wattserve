# n1 Timeline Plots

These plots use the first 5s of the final 20s analysis window from the `n1` steady-window sweep.

- States: `g1` short GEMM phase, `g16` low-clock middle phase, `g20` recovered phase.
- Shading: blue = GEMM phase, orange = norm phase.
- Timeline: [n1_three_state_clock_power_timeline_tail5s.png](n1_three_state_clock_power_timeline_tail5s.png)

| GEMM steps | GEMM phase ms | Norm phase ms | GEMM clock MHz | GEMM power W | TFLOPs/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.1 | 0.3895 | 1262.0 | 397.8 | 265.93 |
| 16 | 533.0 | 0.3887 | 1253.3 | 398.5 | 264.05 |
| 20 | 591.9 | 0.3888 | 1410.0 | 353.1 | 297.21 |
