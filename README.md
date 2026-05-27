# Market-Making with Jump Processes

A stochastic optimal control framework for intraday market-making, implemented and backtested on TAQ (Trade and Quote) data. The agent posts limit orders and issues market orders according to a policy derived from solving a Hamilton-Jacobi-Bellman (HJB) PDE over a four-dimensional state space.

---

## Project Structure

```
Primary Model (Highest PnL)/
    Main_Algorithm.py      # HJB solver, backtest engine, MLE calibration
    HPO_Optimizer.py       # Monte Carlo hyperparameter search

Initial Model (Weak)/
    Momentum_Algorithm.py  # Simpler threshold-based strategy
    Momentum_Backtester.py # Backtesting for initial model
    mle_calibration.py     # Structural MLE for signal parameters

Real-Time LOB Simulator/
    simulator.py           # 8-dim Hawkes LOB simulator + GUI
    Real_Time_Algorithm_test.py  # HJB agent wired into the simulator
    calibrate_simulator.py # Fits Hawkes parameters from TAQ data
    taq_parser.py / taq_extractor.py
```

---

## Mathematical Framework

### Momentum Signal

Each incoming market order updates a latent price-pressure signal `alpha`. Between events, `alpha` decays exponentially toward zero:

```
alpha(t) = alpha(t0) * exp(-kappa * (t - t0))
```

When a market order of volume `V` arrives against prevailing liquidity `lambda` (the average quoted size in lots), the signal jumps:

```
alpha -> alpha + eta * (V / lambda^gamma)   [buy market order]
alpha -> alpha - eta * (V / lambda^gamma)   [sell market order]
```

The parameters `kappa` (mean-reversion speed), `eta` (impact coefficient), and `gamma` (concavity in liquidity) are calibrated daily from TAQ data.

### Liquidity State

A second state variable tracks the prevailing order book depth. Each market order of volume `V` consumes liquidity:

```
lambda -> max(lambda_min, lambda - V_lots)
```

Between market orders, `lambda` is pulled back toward its long-run mean `lambda_bar` with coefficient `0.5`. The state space for `lambda` is discretized over `[1, 20]` lots.

### HJB PDE

The agent's value function `V(t, alpha, lambda, q)` satisfies the HJB equation. Stepping backward in time from the terminal condition:

```
dV/dt + (-kappa * alpha) * dV/dalpha
       + 0.5 * (lambda_bar - lambda) * dV/dlambda
       - phi * q^2
       + q * S * alpha * alpha_scale
       + H_ask(z_ask*) + H_bid(z_bid*)
       = 0
```

The spatial derivatives are approximated with an upwind finite difference scheme: the sign of the drift coefficient selects which neighboring grid point to use, preserving stability.

Terminal condition at `t = T`:

```
V(T, alpha, lambda, q) = -|q| * S * upsilon_mo - psi * q^2
```

The first term forces liquidation of residual inventory via market order at cost `upsilon_mo` per share. The second penalizes squared terminal inventory.

### Limit Order Control

At each grid point, the agent optimizes the quantity `z` to post on each side. For the ask side, the expected value of posting `z` lots is:

```
H_ask(z) = (lam_arrival / 2) * E[ V(alpha', lambda', q - Y) - V(alpha, lambda, q) + Y * S * half_spread ]
```

where the expectation is over incoming market order sizes drawn from a discrete distribution `{1, 3, 10}` lots with probabilities `{0.7, 0.2, 0.1}`. The fractional fill `Y` is:

```
Y = z * min(1, V_mo / lambda)
```

The state transitions after a fill are:

```
alpha' = alpha + eta * (V_mo / lambda^gamma)
lambda' = max(lambda_min, lambda - V_mo)
```

The control `z*` is found by exhaustive search over feasible lot sizes, subject to the inventory constraint `|q + z| <= Q_bar`.

Bilinear interpolation in the `(alpha, lambda)` dimensions is used to evaluate `V` at the post-jump state, which generally falls off the grid.

### Impulse Control (Market Orders)

At each step, the optimal policy is compared against the value of immediately issuing a market order to reduce inventory:

```
M(q) = max over m in {1, ..., |q|} of V(alpha, lambda, q - sign(q)*m) - m * S * upsilon_mo
```

If `M(q) > V_unconstrained`, the agent executes the market order rather than posting limit orders.

### Parameter Calibration

After each trading day, the structural parameters are re-estimated by maximum likelihood. The log-likelihood for the signal model treats mid-price jumps as a doubly-stochastic point process:

```
log L = sum over up-jump times of log(max(alpha_t, 0) + theta)
      + sum over down-jump times of log(max(-alpha_t, 0) + theta)
      - integral of [max(alpha_t, 0) + theta + max(-alpha_t, 0) + theta] dt
```

The baseline rate `theta` prevents zero intensities when `alpha` is near zero. Optimization uses L-BFGS-B. The updated estimates are blended with the prior parameters via exponential moving average to limit day-to-day drift:

```
kappa_new = (1 - alpha_struct) * kappa_old + alpha_struct * kappa_MLE
```

with `alpha_struct = 0.15` for structural parameters and `alpha_obs = 0.35` for observable rates (`lam_arrival`, `lambda_bar`).

### Intraday Re-solving

Within each trading day, the algorithm re-solves the HJB PDE every 30 minutes using updated estimates of `lam_arrival` and `lambda_bar` computed from the most recent 30-minute window. This allows the policy to adapt to intraday variation in market activity.

---

## Real-Time LOB Simulator

The simulator generates synthetic order flow using an 8-dimensional multivariate Hawkes process with a power-law kernel:

```
lambda_i(t) = mu_i * f(t) + sum over past events j: alpha_ij * (1 + gamma_ij * (t - t_j))^(-beta_ij)
```

The 8 event types are: limit order arrivals at `ask0` and `bid0`, cancellations at `ask0` and `bid0`, market buy and sell orders, and in-spread quote insertions on both sides. The spread-dependence of in-spread events is modeled as `(spread - 1)^beta_spread`. Thinning (Ogata's method) is used for simulation.

The LOB itself tracks four queue levels (`ask+1`, `ask0`, `bid0`, `bid-1`) with proportional fill mechanics: a market order of size `V` consumes a fraction `V / total_queue_size` of each resting order.

Hawkes parameters are fit from historical TAQ data and stored per ticker in a JSON configuration file.

---

## Hyperparameter Optimization

`HPO_Optimizer.py` runs a Monte Carlo search over the four cost function parameters:

| Parameter | Role |
|-----------|------|
| `phi` | Running inventory penalty coefficient |
| `psi` | Terminal inventory penalty coefficient |
| `alpha_scale` | Scaling factor for the alpha signal in the PDE |
| `max_clip_lots` | Maximum lot size for limit orders |

Each candidate is scored on total PnL with a drawdown penalty applied if maximum drawdown exceeds $1,500.

---

## Data Format

The backtest expects NYSE TAQ-format CSVs with columns `TIME_M`, `BID`, `ASK`, `BIDSIZ`, `ASKSIZ` for quotes and `TIME_M`, `PRICE`, `SIZE` for trades. Trade side is inferred by comparing trade price to the prevailing mid.

---

## Dependencies

```
numpy, pandas, scipy, numba, matplotlib
PyQt6, pyqtgraph  (for the real-time LOB simulator)
```

The HJB solver and backtest engine are JIT-compiled with Numba and parallelized across the alpha dimension.
