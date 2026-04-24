import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

dt = 0.0005            # 0.0005 second time step
steps_per_frame = 70   # Calculate 70ms of data per visual frame update
window_size = 5000     # Keep the last 2.5 seconds of data on screen

sigma = 0.01           # Tick size ($0.01)
theta = 1.0            # Baseline arrival rate of price ticks
kappa = 15.0           # Mean reversion speed of the imbalance
xi = 1.0               # Volatility of the continuous order flow noise

mu_plus = 5.0          # Baseline exogenous buy order rate
mu_minus = 5.0         # Baseline exogenous sell order rate
gamma = 100.0          # Market decay rate

beta_pp = 25.0         # Buys exciting buys 
beta_mm = 25.0         # Sells exciting sells
beta_pm = 20.0         # High cross-excitation
beta_mp = 20.0         # Buys exciting sells

eta = 0.02             # Market impact scaling factor for g(V)

def draw_realistic_volumes(num_orders):
    if num_orders == 0:
        return np.array([])
        
    rands = np.random.rand(num_orders)
    volumes = np.zeros(num_orders)
    
    # 70% chance of a standard round lot (100 shares)
    idx_round = rands < 0.70
    volumes[idx_round] = 100.0
    
    # 20% chance of a small odd lot (10 to 99 shares)
    idx_odd = (rands >= 0.70) & (rands < 0.90)
    if np.sum(idx_odd) > 0:
        volumes[idx_odd] = np.random.randint(10, 100, size=np.sum(idx_odd))
        
    # 10% chance of a massive block trade (Pareto Tail)
    idx_tail = rands >= 0.90
    if np.sum(idx_tail) > 0:
        pareto_alpha = 1.5 
        base_block_size = 200
        volumes[idx_tail] = (np.random.pareto(pareto_alpha, size=np.sum(idx_tail)) + 1) * base_block_size
        
    return volumes

current_t = 0.0
current_lambda_plus = mu_plus
current_lambda_minus = mu_minus
current_alpha = 0.0
current_S = 100.0      

t_data = deque([current_t], maxlen=window_size)
lp_data = deque([current_lambda_plus], maxlen=window_size)
lm_data = deque([current_lambda_minus], maxlen=window_size)
alpha_data = deque([current_alpha], maxlen=window_size)
S_data = deque([current_S], maxlen=window_size)

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.canvas.manager.set_window_title('Market Microstructure (Pareto Volume)')

line_lp, = ax0.plot([], [], color='green', linewidth=1, label=r'Buy Rate ($\lambda^+$)')
line_lm, = ax0.plot([], [], color='red', linewidth=1, label=r'Sell Rate ($\lambda^-$)')
ax0.set_title('Hawkes Market Order Arrival Rates')
ax0.set_ylabel('Orders / Sec')
ax0.legend(loc='upper right')
ax0.grid(True, alpha=0.3)

line_alpha, = ax1.plot([], [], color='purple', linewidth=1)
ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
ax1.set_title(r'Order Flow Imbalance ($\alpha_t$)')
ax1.set_ylabel('Imbalance')
ax1.grid(True, alpha=0.3)

line_S, = ax2.step([], [], color='blue', linewidth=1.5, where='post')
ax2.set_title('Stock Mid-Price ($S_t$)')
ax2.set_xlabel('Time (Seconds)')
ax2.set_ylabel('Price ($)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

def update_simulation(frame):
    global current_t, current_lambda_plus, current_lambda_minus, current_alpha, current_S
    
    for _ in range(steps_per_frame):
        # Market Order Arrivals
        dN_plus = np.random.poisson(current_lambda_plus * dt)
        dN_minus = np.random.poisson(current_lambda_minus * dt)
        
        # Update Hawkes Intensities
        d_lp = gamma * (mu_plus - current_lambda_plus) * dt + beta_pp * dN_plus + beta_pm * dN_minus
        d_lm = gamma * (mu_minus - current_lambda_minus) * dt + beta_mm * dN_minus + beta_mp * dN_plus
        
        current_lambda_plus = max(current_lambda_plus + d_lp, mu_plus)
        current_lambda_minus = max(current_lambda_minus + d_lm, mu_minus)
        
        # Volume Pareto/Round-Lot function
        impact_plus = 0.0
        if dN_plus > 0:
            V_plus = draw_realistic_volumes(dN_plus)
            impact_plus = np.sum(eta * np.log(1 + V_plus))
            
        impact_minus = 0.0
        if dN_minus > 0:
            V_minus = draw_realistic_volumes(dN_minus)
            impact_minus = np.sum(eta * np.log(1 + V_minus))
        
        # Update Alpha 
        dW = np.random.normal(0, np.sqrt(dt))
        d_alpha = -kappa * current_alpha * dt + xi * dW + impact_plus - impact_minus
        current_alpha += d_alpha
        
        # Update Mid-Price Tick Intensities
        mu_plus_tick = max(current_alpha, 0) + theta
        mu_minus_tick = max(-current_alpha, 0) + theta
        
        # Mid-Price Jumps
        dJ_plus = np.random.poisson(mu_plus_tick * dt)
        dJ_minus = np.random.poisson(mu_minus_tick * dt)
        current_S += sigma * (dJ_plus - dJ_minus)
        
        # Update Time and Append
        current_t += dt
        t_data.append(current_t)
        lp_data.append(current_lambda_plus)
        lm_data.append(current_lambda_minus)
        alpha_data.append(current_alpha)
        S_data.append(current_S)

    # Update Graphics
    line_lp.set_data(t_data, lp_data)
    line_lm.set_data(t_data, lm_data)
    line_alpha.set_data(t_data, alpha_data)
    line_S.set_data(t_data, S_data)
    
    ax0.set_xlim(t_data[0], t_data[-1])
    ax1.set_xlim(t_data[0], t_data[-1])
    ax2.set_xlim(t_data[0], t_data[-1])
    
    l_max = max(max(lp_data), max(lm_data))
    ax0.set_ylim(0, l_max * 1.1)
    
    alpha_min, alpha_max = min(alpha_data), max(alpha_data)
    alpha_pad = max(abs(alpha_max - alpha_min) * 0.1, 1.0)
    ax1.set_ylim(alpha_min - alpha_pad, alpha_max + alpha_pad)
    
    S_min, S_max = min(S_data), max(S_data)
    S_pad = max((S_max - S_min) * 0.2, sigma * 2) 
    ax2.set_ylim(S_min - S_pad, S_max + S_pad)
    
    return line_lp, line_lm, line_alpha, line_S

ani = animation.FuncAnimation(
    fig, update_simulation, interval=33, blit=False, cache_frame_data=False
)

plt.show()