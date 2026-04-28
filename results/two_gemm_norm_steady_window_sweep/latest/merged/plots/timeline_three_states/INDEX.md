# Three-State Clock/Power Timelines

These plots use existing steady-window results. Each panel shows representative cycles from the final-20s tail window, not startup.

- States: `g1` short GEMM phase, `g16` middle low-clock phase, `g20` recovered phase.
- Norm configs: `n128`, `n512`, `n1024`.
- Shading: blue = GEMM phase, orange = norm phase.
- Power and clock y-axis limits are shared across all panels.

- Detail: [n128_three_state_clock_power_timeline_tail5s.png](n128_three_state_clock_power_timeline_tail5s.png)
- Detail: [n512_three_state_clock_power_timeline_tail5s.png](n512_three_state_clock_power_timeline_tail5s.png)
- Detail: [n1024_three_state_clock_power_timeline_tail5s.png](n1024_three_state_clock_power_timeline_tail5s.png)

| Norm steps | Norm phase ms | GEMM steps | GEMM phase ms | GEMM clock MHz | GEMM power W | TFLOPs/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 51.0 | 1 | 30.3 | 1352.9 | 367.6 | 290.09 |
| 128 | 51.0 | 16 | 529.0 | 1266.9 | 392.4 | 266.02 |
| 128 | 51.0 | 20 | 592.4 | 1410.0 | 338.7 | 296.94 |
| 512 | 204.1 | 1 | 30.0 | 1404.2 | 291.9 | 292.99 |
| 512 | 204.1 | 16 | 526.8 | 1280.5 | 389.2 | 267.14 |
| 512 | 204.2 | 20 | 592.5 | 1410.0 | 329.9 | 296.90 |
| 1024 | 408.3 | 1 | 29.9 | 1409.2 | 299.3 | 293.71 |
| 1024 | 408.3 | 16 | 527.1 | 1281.7 | 385.6 | 267.02 |
| 1024 | 408.4 | 20 | 592.5 | 1410.0 | 329.6 | 296.90 |

## Final-Window 5s Views

These use the first 5s of the final 20s analysis window. This keeps the x-axis comparable across panels while avoiding the density of the full 20s view.
