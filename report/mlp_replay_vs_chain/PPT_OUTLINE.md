# PPT Outline: MLP Replay vs State-Chain

## Slide 1: Title

Title: State-Carrying MLP Chains Expose a Power/Clock Effect

Subtitle: Why `fused_add_rmsnorm` changes GEMM speed without being counted in GEMM time

Speaker note:
We are isolating the stack-level effect into a minimal MLP experiment.

## Slide 2: Question

Main question:

Why does removing norm/residual boundaries in a full Llama stack improve GEMM
throughput and clock behavior more than norm self-time alone can explain?

Key framing:

- Not asking whether norm has a runtime cost.
- Asking whether norm changes the operating point of surrounding GEMMs.

## Slide 3: Minimal Workload

Workload:

`gate + up -> silu_and_mul -> down`

Variants:

- without `fused_add_rmsnorm`
- with `fused_add_rmsnorm`

Shape:

- Llama-13B
- prompt length 8192
- hidden 5120
- intermediate 13824
- bfloat16

## Slide 4: Two Benchmark Semantics

Core distinction:

Both modes run the same MLP kernels. They differ only in whether the MLP output
is fed back as the next hidden state.

## Slide 5: Fixed Replay Diagram

`fixed_replay` semantics:

- always reads the same fixed hidden source
- output is overwritten
- no stack-like state carry

Diagram:

```text
source ──MLP──> out  discarded
source ──MLP──> out  discarded
source ──MLP──> out  discarded
...
```

With norm enabled:

```text
source ──gate/up/down GEMMs──> tmp ──fused_add_rmsnorm──> normed_out
   ^                                                           |
   |                                                           v
   +---------------- next step still reads source          discarded
```

Buffer view:

```text
source:      read-only, reused every step
next_hidden: overwritten every step
```

Pseudocode:

```python
for _ in range(steps):
    run_mlp_once(source, next_hidden)
```

Speaker note:
This mode is a control. It has the same GEMM shapes and optional norm kernels,
but it does not preserve stack-like dataflow. Even if norm changes `next_hidden`,
that tensor is not used as the next GEMM input.

## Slide 6: State Chain Diagram

`state_chain`:

- each MLP output becomes the next hidden input
- state carries across steps and repeats
- stack-like operator cadence

Diagram:

```text
hidden_0 ──MLP──> hidden_1 ──MLP──> hidden_2 ──MLP──> hidden_3 ...
```

With norm enabled:

```text
hidden_i ──gate/up/down GEMMs──> tmp ──fused_add_rmsnorm──> hidden_i+1
   ^                                                               |
   |                                                               v
   +--------------------- next step GEMMs read hidden_i+1
```

Buffer view:

```text
step 1: current=A, next_hidden=B -> write B -> swap -> current=B
step 2: current=B, next_hidden=A -> write A -> swap -> current=A
step 3: current=A, next_hidden=B -> write B -> swap -> current=B
```

Pseudocode:

```python
for _ in range(steps):
    run_mlp_once(current, next_hidden)
    current, next_hidden = next_hidden, current
```

Speaker note:
The output tensor is not just measured and thrown away. It becomes the next
large GEMM input, like a stack of layers. Therefore the norm boundary is not
only an extra kernel; it can change the operating point seen by later GEMMs.

## Slide 7: Why The Difference Matters

Comparison:

| Property | fixed replay | state chain |
| --- | --- | --- |
| Step input | same source every time | previous step output |
| Output use | discarded | next input |
| Stack-like cadence | no | yes |
| What it controls for | local kernel cost | operating-point feedback |

Message:

If norm were only local kernel self-time, both modes should react similarly.
They do not.

## Slide 8: Measurement Rule

GEMM TFLOPs/s is computed only from GEMM kernel time.

Included:

- gate GEMM
- up GEMM
- down GEMM

Excluded:

- `silu_and_mul`
- activation copy / concat
- `fused_add_rmsnorm`
- Python loop overhead

Speaker note:
This is essential. The throughput drop cannot be explained by accidentally
putting norm or activation time in the GEMM denominator.

## Slide 9: Four-Row Result

| Mode | Norm | GEMM TFLOPs/s | Power (W) | Clock (MHz) |
| --- | --- | ---: | ---: | ---: |
| fixed replay | no | 268.38 | 400.97 | 1289.11 |
| fixed replay | yes | 268.78 | 392.11 | 1290.00 |
| state chain | no | 292.98 | 316.99 | 1410.00 |
| state chain | yes | 270.26 | 398.39 | 1295.46 |

Suggested visual:

Grouped bar chart with two panels:

- GEMM TFLOPs/s
- GPU clock

## Slide 10: Fixed Replay Does Not React

Adding norm in fixed replay:

- GEMM TFLOPs/s: `+0.15%`
- power: `-8.86 W`
- clock: `+0.89 MHz`

Message:

When the input is replayed rather than carried forward, inserting the norm
boundary does not meaningfully affect surrounding GEMMs.

## Slide 11: State Chain Reacts Strongly

Adding norm in state chain:

- GEMM TFLOPs/s: `-7.76%`
- power: `+81.41 W`
- clock: `-114.54 MHz`

Message:

The same norm boundary is enough to move the state-carrying chain into a
higher-power, lower-clock operating point.

## Slide 12: The Critical Contrast

Without norm:

- state chain is much faster than fixed replay
- `292.98` vs `268.38` GEMM TFLOPs/s
- `1410 MHz` vs `1289 MHz`

With norm:

- state chain collapses back near fixed replay
- `270.26` vs `268.78` GEMM TFLOPs/s
- `1295 MHz` vs `1290 MHz`

Message:

The norm boundary erases the state-chain high-clock advantage.

## Slide 13: Interpretation

Main interpretation:

The effect is a workload-state effect, not a norm self-time effect.

Reasoning:

- Norm and activation are excluded from GEMM TFLOPs/s.
- Fixed replay plus norm does not show a GEMM slowdown.
- State chain plus norm does show a GEMM slowdown.
- Therefore the state-carrying norm/residual cadence changes the operating
  point seen by later GEMMs.

## Slide 14: Relation to Full Stack Results

This minimal MLP experiment supports the stack-level hypothesis:

- full stack reproduces power/clock split
- naive repeated single block can differ because benchmark dataflow/reset
  structure differs
- state-carrying chains are the durable reproducer

Message:

The stack effect can be reproduced without full attention, final norm, or
lm_head.

## Slide 15: Caveats

Caveats:

- Result is from A100 40G SXM.
- `steps=40` is used for presentation; the sweep showed similar qualitative
  behavior at smaller step counts.
- The focused result uses real `silu_and_mul`; broader no-silu tests show silu
  is not strictly required.

## Slide 16: Takeaway

Takeaway:

In a state-carrying Llama MLP chain, inserting `fused_add_rmsnorm` makes the
following GEMMs run at lower clock and lower GEMM-only throughput. Replaying
the same fixed input does not show this behavior.

Closing line:

The important benchmark variable is not only which kernels are present, but
whether the workload preserves stack-like state-carrying cadence.
