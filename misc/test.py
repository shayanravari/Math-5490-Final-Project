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

lambda_plus = 30.0     # Rate of exogenous buy market orders per second
lambda_minus = 30.0    # Rate of exogenous sell market orders per second
mean_vol = 100.0       # Mean volume of an incoming order

eta = 0.5              # Market impact scaling factor for g(V)

current_t = 0.0
current_alpha = 0.0
current_S = 100.0

t_data = deque([current_t], maxlen=window_size)
alpha_data = deque([current_alpha], maxlen=window_size)
S_data = deque([current_S], maxlen=window_size)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
fig.canvas.manager.set_window_title('Microscopic Market Microstructure Simulation')

line_alpha, = ax1.plot([], [], color='purple', linewidth=1)
ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
ax1.set_title(r'Order Flow Imbalance ($\alpha_t$) Driven by Compound Poisson Jumps')
ax1.set_ylabel('Imbalance')
ax1.grid(True, alpha=0.3)

line_S, = ax2.step([], [], color='blue', linewidth=1.5, where='post')
ax2.set_title('Stock Mid-Price ($S_t$)')
ax2.set_xlabel('Time (Seconds)')
ax2.set_ylabel('Price ($)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

def update_simulation(frame):
    global current_t, current_alpha, current_S
    
    for _ in range(steps_per_frame):
        dW = np.random.normal(0, np.sqrt(dt))

        # Market Order Arrivals
        dN_plus = np.random.poisson(lambda_plus * dt)
        dN_minus = np.random.poisson(lambda_minus * dt)
        
        # Impact of buy orders
        impact_plus = 0.0
        if dN_plus > 0:
            V_plus = np.random.exponential(scale=mean_vol, size=dN_plus)
            impact_plus = np.sum(eta * np.log(1 + V_plus)) # g(V)
            
        # Impact of sell orders
        impact_minus = 0.0
        if dN_minus > 0:
            V_minus = np.random.exponential(scale=mean_vol, size=dN_minus)
            impact_minus = np.sum(eta * np.log(1 + V_minus)) # g(V)
        
        # Update alpha
        d_alpha = -kappa * current_alpha * dt + xi * dW + impact_plus - impact_minus
        current_alpha += d_alpha
        
        # Update Mid-Price Tick Intensities
        mu_plus = max(current_alpha, 0) + theta
        mu_minus = max(-current_alpha, 0) + theta
        
        # Mid-Price Jumps
        dJ_plus = np.random.poisson(mu_plus * dt)
        dJ_minus = np.random.poisson(mu_minus * dt)
        
        # Update Time and Append
        current_S += sigma * (dJ_plus - dJ_minus)
        current_t += dt
        t_data.append(current_t)
        alpha_data.append(current_alpha)
        S_data.append(current_S)

    # Update Graphics
    line_alpha.set_data(t_data, alpha_data)
    line_S.set_data(t_data, S_data)
    ax1.set_xlim(t_data[0], t_data[-1])
    ax2.set_xlim(t_data[0], t_data[-1])
    
    alpha_min, alpha_max = min(alpha_data), max(alpha_data)
    alpha_pad = max(abs(alpha_max - alpha_min) * 0.1, 1.0)
    ax1.set_ylim(alpha_min - alpha_pad, alpha_max + alpha_pad)
    
    S_min, S_max = min(S_data), max(S_data)
    S_pad = max((S_max - S_min) * 0.2, sigma * 2) 
    ax2.set_ylim(S_min - S_pad, S_max + S_pad)
    
    return line_alpha, line_S

ani = animation.FuncAnimation(
    fig, update_simulation, interval=33, blit=False, cache_frame_data=False
)

plt.show()