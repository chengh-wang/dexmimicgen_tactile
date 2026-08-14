# Tactile WM Residual Roadmap

Date: 2026-07-30

## Goal

Train a shared tactile encoder and task-specific base policies, collect policy-rollout success and failure data, train task-specific world models, then use progress-conditioned success values for residual action optimization.

## Pipeline Gates

### Gate 1: Shared Tactile VAE Encoder

Use one shared frozen tactile VAE encoder across all tasks.

Requirements:
- All task h5 files can be encoded by the same tactile encoder.
- Tactile preprocessing is consistent between train and eval.
- Use raw simulation tactile values with normalization, not `/2200`.
- Flow policy keeps the tactile VAE frozen and only trains the downstream policy/fusion MLP.

### Gate 2: Base Policy Training And Evaluation

Train base flow policies for all tasks using the shared tactile VAE.

For each task, record:
- best checkpoint
- best NFE
- 20/50 episode success rate
- action contract: delta vs absolute, action dimension, rot6d conversion details
- observation contract: RGB, lowdim, tactile keys and encoder dimensions

Task should not enter the WM/residual stage unless its base policy is nonzero and behavior is visually reasonable.

Minimum criterion:
- Prefer `>= 5/20` success.
- Bare minimum `>= 2/20` if the task is hard but clearly not broken.
- Full-zero tasks must be debugged before data collection.

### Gate 3: Collect Success And Failure Rollouts

For each task, collect rollout data from the trained base policy:
- 250 successful rollouts
- 250 failed rollouts

Each rollout should save:
- observation sequence: RGB, lowdim, tactile
- action sequence
- success label
- progress `tau`
- terminal success/reward
- optional sanity-check video

Important:
- Success and failure should come from the same base-policy distribution.
- Failure should include full trajectories, especially late failures.
- Avoid mixing human-demo success with policy-rollout failure unless explicitly tracked as different distributions.

### Gate 4: Train Task-Specific WM

Train one world model per task using that task's success and failure rollouts.

WM target:
- input: history latent and action chunk
- output: future latent / rollout latent

Checks before residual:
- WM rollout latent does not visibly diverge.
- Success-vs-failure probe AUC is meaningfully above random, ideally `> 0.75`.
- Value curves along successful trajectories are smoother and more interpretable than the old terminal setpoint value.

### Gate 5: Progress-Conditioned Residual Value

Use WM rollout plus a progress-conditioned value to optimize residual actions.

#### B1: Kernel-Smoothed Time-Varying Direction

For each progress `tau`, compute a local success direction:

```text
v(tau) = normalize(
    sum_i K_h(tau_i - tau) z_i_success
  - sum_j K_h(tau_j - tau) z_j_failure
)
```

This is a smooth version of progress-binned `v(tau)`.

#### B2: Progress-Conditioned Logistic Probe

Train:

```text
p(success | z, tau) = sigmoid(w(tau)^T z + b(tau))
w(tau) = W phi(tau)
```

where `phi(tau)` can be an RBF, polynomial, or Fourier basis.

This gives a continuous, monotonic value and avoids the old symmetric Gaussian overshoot penalty.

#### Residual Controller

At eval time:

```text
U_base = base_policy(obs_history)
z_H = WM_rollout(z_history, U_base + delta_U)
V = value(z_H, tau_H)
g = dV / dU
delta_U = clip(g / lambda, -trust_delta, trust_delta)
a_exec = a_base[0] + delta_U[0]
```

## Current Status

We are currently in **Gate 2 / early Gate 3**, not fully past Gate 3.

Current base-policy eval status:
- `three_piece`: nonzero; best currently around `50k nfe8 = 12/20`.
- `can_sort_random`: nonzero; best currently around `20k nfe8 = 13/20`.
- `pouring`: nonzero; best currently around `20k nfe16 = 13/20`.
- `box_cleanup`: nonzero; best currently around `40k nfe4 = 9/20`.
- `threading`: weak but nonzero; best currently around `20k nfe8 = 5/20`.
- `lift_tray`: currently all zero under 400-step eval; max700 90k/100k debug eval is running.
- `transport`: currently all zero under 400-step eval; max700 90k/100k debug eval is running.

Interpretation:
- We have not yet completed the full base-policy gate for all tasks.
- Some tasks are ready candidates for rollout collection.
- `lift_tray` and `transport` must be debugged or fixed before collecting success/failure data.

## Next Step

Finish the base-policy matrix and identify one reliable config per task.

Then:
1. Start success/failure rollout collection for nonzero tasks.
2. Keep debugging `lift_tray` and `transport` until they have nonzero, visually reasonable base policies.
3. Use two pilot tasks first for the full WM + B1/B2 residual pipeline:
   - one stronger task, e.g. `can_sort_random` or `three_piece`
   - one weaker task, e.g. `threading`

Do not start all-task WM/residual training until the base-policy and rollout collection gates are clean.
