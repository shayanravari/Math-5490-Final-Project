import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import time
import json
import numpy as np
import pyqtgraph as pg
from collections import deque
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont
from Real_Time_Algorithm_test import HJBMarketMaker


class OrderSizeModel:
    def __init__(self, lo_config=None, mo_config=None, co_config=None, empty_prob=0.0):
        self.lo_cfg = lo_config or {'round_numbers':[100], 'round_probs':[1.0], 'geometric_p': 0.05, 'p_round': 1.0}
        self.mo_cfg = mo_config or {'round_numbers':[100], 'round_probs': [1.0], 'geometric_p': 0.05, 'p_round': 1.0}
        self.co_cfg = co_config or {'round_numbers':[100], 'round_probs': [1.0], 'geometric_p': 0.05, 'p_round': 1.0}
        
        self.empty_prob = empty_prob

    def _sample(self, cfg):
        if np.random.rand() < cfg['p_round']:
            return np.random.choice(cfg['round_numbers'], p=cfg['round_probs'])
        else:
            return np.random.geometric(cfg['geometric_p'])

    def sample_lo(self): return self._sample(self.lo_cfg)
    def sample_mo(self): return self._sample(self.mo_cfg)
    def sample_co(self): return self._sample(self.co_cfg)

    def sample_cancel_order(self, queue_orders):
        sim_indices = [i for i, order in enumerate(queue_orders) if order['type'] == 'sim']
        if not sim_indices: return 0
        idx = np.random.choice(sim_indices)
        return queue_orders.pop(idx)['size']
        
    def sample_q_plus2(self):
        if np.random.rand() < self.empty_prob:
            return 0
        else:
            return self.sample_lo()

class SixLevelLOB:
    def __init__(self, tick_size=0.01, initial_price=100.00, size_model=None):
        self.tick_size = tick_size
        self.size_model = size_model or OrderSizeModel()
        
        self.p_ask0 = round(initial_price, 4)
        self.p_bid0 = round(initial_price - self.tick_size, 4)
        
        self.q_ask_plus1 =[{'type': 'sim', 'size': 100}, {'type': 'sim', 'size': 50}]
        self.q_ask0 =[{'type': 'sim', 'size': 50}, {'type': 'sim', 'size': 50}]
        self.q_bid0 =[{'type': 'sim', 'size': 50}, {'type': 'sim', 'size': 50}]
        self.q_bid_minus1 =[{'type': 'sim', 'size': 100}, {'type': 'sim', 'size': 50}]

    @property
    def spread_in_ticks(self):
        return max(1, round(round(self.p_ask0 - self.p_bid0, 4) / self.tick_size))

    @property
    def mid_price(self):
        return round((self.p_ask0 + self.p_bid0) / 2.0, 4)

    def total_q(self, queue):
        return sum(order['size'] for order in queue)

    def trigger_in_spread(self, side, volume, order_type='sim'):
        if side == 'ask':
            self.q_ask_plus1 = self.q_ask0.copy()
            self.q_ask0 = [{'type': order_type, 'size': volume}]
            self.p_ask0 = round(self.p_ask0 - self.tick_size, 4)
        else: 
            self.q_bid_minus1 = self.q_bid0.copy()
            self.q_bid0 = [{'type': order_type, 'size': volume}]
            self.p_bid0 = round(self.p_bid0 + self.tick_size, 4)

    def trigger_queue_depletion(self, side, remaining_vol):
        user_fills =[] 
        remaining_vol = int(remaining_vol) 
        resting_side = 'sell' if side == 'ask' else 'buy'

        if self.total_q(self.q_ask0 if side == 'ask' else self.q_bid0) == 0:
            if side == 'ask':
                self.q_ask0 = self.q_ask_plus1.copy()
                self.p_ask0 = round(self.p_ask0 + self.tick_size, 4)
                self.q_ask_plus1 =[{'type': 'sim', 'size': self.size_model.sample_q_plus2()}]
            else:
                self.q_bid0 = self.q_bid_minus1.copy()
                self.p_bid0 = round(self.p_bid0 - self.tick_size, 4)
                self.q_bid_minus1 =[{'type': 'sim', 'size': self.size_model.sample_q_plus2()}]

        while remaining_vol > 0 and remaining_vol >= self.total_q(self.q_ask0 if side == 'ask' else self.q_bid0):
            target_q = self.q_ask0 if side == 'ask' else self.q_bid0
            current_q_vol = self.total_q(target_q)
            
            for order in target_q:
                if order['type'] == 'user':
                    user_fills.append({'side': resting_side, 'price': self.p_ask0 if side == 'ask' else self.p_bid0, 'size': order['size']})
            
            if side == 'ask':
                remaining_vol -= current_q_vol
                self.q_ask0 = self.q_ask_plus1.copy()
                self.p_ask0 = round(self.p_ask0 + self.tick_size, 4)
                self.q_ask_plus1 =[{'type': 'sim', 'size': self.size_model.sample_q_plus2()}]
            else:
                remaining_vol -= current_q_vol
                self.q_bid0 = self.q_bid_minus1.copy()
                self.p_bid0 = round(self.p_bid0 - self.tick_size, 4)
                self.q_bid_minus1 =[{'type': 'sim', 'size': self.size_model.sample_q_plus2()}]

        if remaining_vol > 0:
            target_q = self.q_ask0 if side == 'ask' else self.q_bid0
            while remaining_vol > 0 and target_q:
                if target_q[0]['size'] <= remaining_vol:
                    filled_order = target_q.pop(0)
                    remaining_vol -= filled_order['size']
                    if filled_order['type'] == 'user':
                        user_fills.append({'side': resting_side, 'price': self.p_ask0 if side == 'ask' else self.p_bid0, 'size': filled_order['size']})
                else:
                    target_q[0]['size'] -= remaining_vol
                    if target_q[0]['type'] == 'user':
                        user_fills.append({'side': resting_side, 'price': self.p_ask0 if side == 'ask' else self.p_bid0, 'size': remaining_vol})
                    remaining_vol = 0
                    
        return user_fills

    def place_custom_limit_order(self, side, price, volume, order_type='user'):
        price = round(price, 4)
        user_fills =[]
        if side == 'buy':
            if price >= self.p_ask0:
                self.execute_market_order('ask', volume)
                user_fills.append({'side': 'buy', 'price': self.p_ask0, 'size': volume})
            elif self.p_bid0 < price < self.p_ask0:
                self.q_bid_minus1 = self.q_bid0.copy()
                self.q_bid0 = [{'type': order_type, 'size': volume}]
                self.p_bid0 = price
            elif price == self.p_bid0:
                self.q_bid0.append({'type': order_type, 'size': volume})
            elif price == round(self.p_bid0 - self.tick_size, 4):
                self.q_bid_minus1.append({'type': order_type, 'size': volume})
                
        elif side == 'sell':
            if price <= self.p_bid0:
                self.execute_market_order('bid', volume)
                user_fills.append({'side': 'sell', 'price': self.p_bid0, 'size': volume})
            elif self.p_bid0 < price < self.p_ask0:
                self.q_ask_plus1 = self.q_ask0.copy()
                self.q_ask0 =[{'type': order_type, 'size': volume}]
                self.p_ask0 = price
            elif price == self.p_ask0:
                self.q_ask0.append({'type': order_type, 'size': volume})
            elif price == round(self.p_ask0 + self.tick_size, 4):
                self.q_ask_plus1.append({'type': order_type, 'size': volume})
                
        return user_fills

    def add_limit_order(self, queue_name, volume, order_type='sim'):
        order = {'type': order_type, 'size': volume}
        if queue_name == 'ask+1': self.q_ask_plus1.append(order)
        elif queue_name == 'ask0': self.q_ask0.append(order)
        elif queue_name == 'bid0': self.q_bid0.append(order)
        elif queue_name == 'bid-1': self.q_bid_minus1.append(order)

    def execute_cancel_order(self, queue_name):
        if queue_name == 'ask0': target_q = self.q_ask0
        elif queue_name == 'ask+1': target_q = self.q_ask_plus1
        elif queue_name == 'bid0': target_q = self.q_bid0
        elif queue_name == 'bid-1': target_q = self.q_bid_minus1
        else: return

        # Find all simulated orders
        sim_indices =[i for i, order in enumerate(target_q) if order['type'] == 'sim']
        
        if not sim_indices:
            return 
            
        idx = np.random.choice(sim_indices)
        target_q.pop(idx)

    def execute_market_order(self, side, volume):
        return self.trigger_queue_depletion(side, volume)
    
    def cancel_user_orders(self, side=None):
        if side in ['ask', None]:
            self.q_ask_plus1 =[o for o in self.q_ask_plus1 if o.get('type') != 'user']
            self.q_ask0 =[o for o in self.q_ask0 if o.get('type') != 'user']
        if side in ['buy', None]:
            self.q_bid0 =[o for o in self.q_bid0 if o.get('type') != 'user']
            self.q_bid_minus1 =[o for o in self.q_bid_minus1 if o.get('type') != 'user']

class EightDimHawkesSimulator:
    def __init__(self, lob, mu, alpha, gamma, beta_kernel, f_Qt_func, beta_spread):
        self.lob = lob
        self.mu = np.array(mu)
        self.alpha = np.array(alpha)
        self.gamma = np.array(gamma)
        self.beta_kernel = np.array(beta_kernel)
        self.f_Qt_func = f_Qt_func
        self.beta_spread = beta_spread
        self.dim = 8
        
        self.max_hist = 500
        self.hist_t = np.zeros(self.max_hist)
        self.hist_a = np.zeros((8, self.max_hist))
        self.hist_g = np.zeros((8, self.max_hist))
        self.hist_b = np.zeros((8, self.max_hist))
        
        self.head = 0
        self.hist_len = 0

    def calculate_intensities(self, t):
        s = self.lob.spread_in_ticks
        f_qt = self.f_Qt_func(t)
        lambda_t = np.copy(self.mu)
        
        if self.hist_len > 0:
            dt = t - self.hist_t[:self.hist_len]
            a = self.hist_a[:, :self.hist_len]
            g = self.hist_g[:, :self.hist_len]
            b = self.hist_b[:, :self.hist_len]
            
            kernel_vals = a * (1.0 + g * dt)**(-b)
            lambda_t += np.sum(kernel_vals, axis=1)

        lambda_t *= f_qt
        
        if s > 1:
            is_multiplier = (s - 1) ** self.beta_spread
            lambda_t[3] *= is_multiplier
            lambda_t[4] *= is_multiplier
        else:
            lambda_t[3] = 0.0
            lambda_t[4] = 0.0
            
        return np.maximum(0, lambda_t)

    def _add_to_history(self, t, event_id):
        self.hist_t[self.head] = t
        self.hist_a[:, self.head] = self.alpha[:, event_id]
        self.hist_g[:, self.head] = self.gamma[:, event_id]
        self.hist_b[:, self.head] = self.beta_kernel[:, event_id]
        
        self.head = (self.head + 1) % self.max_hist
        if self.hist_len < self.max_hist:
            self.hist_len += 1

    def simulate_next_event(self, t_current):
        t = t_current
        while True:
            lambda_t = self.calculate_intensities(t)
            lambda_sum = np.sum(lambda_t)
            
            if lambda_sum == 0:
                t += 0.1
                continue
                
            tau = np.random.exponential(1.0 / lambda_sum)
            t += tau
            
            lambda_t_new = self.calculate_intensities(t)
            lambda_sum_new = np.sum(lambda_t_new)
            
            if np.random.uniform(0, lambda_sum) <= lambda_sum_new:
                probs = lambda_t_new / lambda_sum_new
                event_id = np.random.choice(self.dim, p=probs)
                self._add_to_history(t, event_id)
                return t, event_id

class LOBBacktestEnv:
    def __init__(self, lob, simulator, size_model):
        self.lob = lob
        self.simulator = simulator
        self.size_model = size_model
        self.current_time = 0.0
        self.recent_trades =[] 
        
        self.inventory = 0        
        self.cash = 100000.0      
        self.realized_pnl = 0.0

    @property
    def total_pnl(self):
        unrealized = self.inventory * self.lob.mid_price
        return (self.cash - 100000.0) + unrealized

    def process_user_fills(self, fills):
        for fill in fills:
            if fill['side'] == 'buy':
                self.inventory += fill['size']
                self.cash -= fill['size'] * fill['price']
            elif fill['side'] == 'sell':
                self.inventory -= fill['size']
                self.cash += fill['size'] * fill['price']
                
            print(f"[{self.current_time:.2f}] AGENT FILLED! {fill['side'].upper()} {fill['size']} @ ${fill['price']:.2f} | Inv: {self.inventory} | PnL: ${self.total_pnl:.2f}")

    def step(self, user_actions=None):
        if user_actions:
            for action in user_actions:
                if action['type'] == 'cancel':
                    self.lob.cancel_user_orders(action.get('side'))
                elif action['type'] == 'limit':
                    fills = self.lob.place_custom_limit_order(
                        side=action['side'], price=action['price'], 
                        volume=action['size'], order_type='user'
                    )
                    if fills:
                        self.process_user_fills(fills)
                        for fill in fills:
                            self.recent_trades.append({
                                'time': self.current_time, 
                                'side': 'BUY' if action['side'] == 'buy' else 'SELL', 
                                'price': fill['price'], 
                                'size': fill['size'],
                                'is_agent': True
                            })
                            
                elif action['type'] == 'market':
                    side = action['side']
                    exec_price = self.lob.p_ask0 if side == 'buy' else self.lob.p_bid0
                    self.recent_trades.append({
                        'time': self.current_time, 
                        'side': 'BUY' if side == 'buy' else 'SELL', 
                        'price': exec_price, 
                        'size': action['size'],
                        'is_agent': True
                    })
                    
                    self.process_user_fills([{'side': side, 'price': exec_price, 'size': action['size']}])
                    self.lob.execute_market_order('ask' if side == 'buy' else 'bid', action['size'])

        next_time, event_id = self.simulator.simulate_next_event(self.current_time)
        self.current_time = next_time
        
        if event_id in[0, 5]:
            vol = self.size_model.sample_lo()
            if event_id == 0: self.lob.add_limit_order('ask0', vol)
            elif event_id == 5: self.lob.add_limit_order('bid0', vol)
            
        elif event_id in [3, 4]:
            vol = self.size_model.sample_lo()
            if event_id == 3: self.lob.trigger_in_spread('ask', vol)
            elif event_id == 4: self.lob.trigger_in_spread('bid', vol)
            
        elif event_id in[2, 7]:
            vol = self.size_model.sample_mo()
            user_fills =[]
            
            if event_id == 2: 
                self.recent_trades.append({'time': self.current_time, 'side': 'BUY', 'price': self.lob.p_ask0, 'size': vol, 'is_agent': False})
                user_fills = self.lob.execute_market_order('ask', vol)
            elif event_id == 7: 
                self.recent_trades.append({'time': self.current_time, 'side': 'SELL', 'price': self.lob.p_bid0, 'size': vol, 'is_agent': False})
                user_fills = self.lob.execute_market_order('bid', vol)
                
            if user_fills:
                self.process_user_fills(user_fills)

        elif event_id in [1, 6]:
            if event_id == 1:
                self.lob.execute_cancel_order('ask0')
                if self.lob.total_q(self.lob.q_ask0) == 0: 
                    self.lob.trigger_queue_depletion('ask', 0)
                    
            elif event_id == 6:
                self.lob.execute_cancel_order('bid0')
                if self.lob.total_q(self.lob.q_bid0) == 0: 
                    self.lob.trigger_queue_depletion('bid', 0)

        return event_id

class SimpleMarketMaker:
    def __init__(self, order_size=100, action_interval=0.2):
        self.order_size = order_size
        self.action_interval = action_interval
        self.last_action_time = 0.0
        self.last_mo_time = 0.0

    def get_actions(self, env):
        if env.current_time - self.last_action_time < self.action_interval:
            return None
            
        self.last_action_time = env.current_time
        actions =[]
        
        if env.current_time - self.last_mo_time >= 10.0:
            self.last_mo_time = env.current_time
            if env.inventory == 0:
                actions.append({'type': 'market', 'side': 'buy', 'size': self.order_size})
                print(f"[{env.current_time:.2f}] BOT: Placed BUY MO at {self.order_size} shares.")
            else:
                actions.append({'type': 'market', 'side': 'sell', 'size': env.inventory})
                print(f"[{env.current_time:.2f}] BOT: Placed SELL MO at {env.inventory} shares.")
            
            return actions

        spread = env.lob.spread_in_ticks
        if spread >= 2:
            aggressive_bid = round(env.lob.p_bid0 + env.lob.tick_size, 4)
            aggressive_ask = round(env.lob.p_ask0 - env.lob.tick_size, 4)
            
            if env.inventory == 0:
                actions.append({'type': 'limit', 'side': 'buy', 'price': aggressive_bid, 'size': self.order_size})
                print(f"[{env.current_time:.2f}] BOT: Placed BUY LO at {aggressive_bid:.2f}")
                
            elif env.inventory >= self.order_size:
                actions.append({'type': 'limit', 'side': 'sell', 'price': aggressive_ask, 'size': self.order_size})
                print(f"[{env.current_time:.2f}] BOT: Placed SELL LO at {aggressive_ask:.2f}")

        return actions

class SimulationThread(QThread):
    update_signal = pyqtSignal(dict)

    def __init__(self, env, agent=None, playback_speed=1.0, target_fps=60):
        super().__init__()
        self.env = env
        self.agent = agent
        self.playback_speed = playback_speed
        self.is_running = True
        self.frame_interval = 1.0 / target_fps 
        self.gui_ready = True 

    def run(self):
        last_real_time = time.perf_counter()
        target_sim_time = self.env.current_time
        last_emit_time = time.perf_counter()

        while self.is_running:
            if not self.gui_ready:
                last_real_time = time.perf_counter()
                time.sleep(0.01)
                continue

            current_time_real = time.perf_counter()
            dt_real = current_time_real - last_real_time
            last_real_time = current_time_real
            
            if dt_real > 0.1:
                dt_real = 0.1
                
            target_sim_time += (dt_real * self.playback_speed)

            while self.env.current_time <= target_sim_time:
                user_actions = None
                if self.agent:
                    user_actions = self.agent.get_actions(self.env)
                self.env.step(user_actions=user_actions)

            if current_time_real - last_emit_time >= self.frame_interval: 
                self.gui_ready = False
                
                lob = self.env.lob
                trades_to_emit = list(self.env.recent_trades)[-50:]
                self.env.recent_trades.clear()

                active_orders =[]
                def scan_queue(q, q_name, side, price):
                    for order in q:
                        if order.get('type') == 'user':
                            active_orders.append({'side': side, 'queue': q_name, 'price': price, 'size': order['size']})
                            
                scan_queue(lob.q_ask_plus1, 'Ask+1', 'SELL', lob.p_ask0 + lob.tick_size)
                scan_queue(lob.q_ask0, 'Ask0', 'SELL', lob.p_ask0)
                scan_queue(lob.q_bid0, 'Bid0', 'BUY', lob.p_bid0)
                scan_queue(lob.q_bid_minus1, 'Bid-1', 'BUY', lob.p_bid0 - lob.tick_size)

                state = {
                    'time': target_sim_time, 
                    'mid': lob.mid_price,
                    'spread': lob.spread_in_ticks,
                    'tick_size': lob.tick_size,
                    'p_ask0': round(lob.p_ask0, 4),
                    'p_bid0': round(lob.p_bid0, 4),
                    'q_ask0': lob.total_q(lob.q_ask0),
                    'q_ask1': lob.total_q(lob.q_ask_plus1),
                    'q_bid0': lob.total_q(lob.q_bid0),
                    'q_bid1': lob.total_q(lob.q_bid_minus1),
                    'has_user_ask0': any(o['type'] == 'user' for o in lob.q_ask0),
                    'has_user_ask1': any(o['type'] == 'user' for o in lob.q_ask_plus1),
                    'has_user_bid0': any(o['type'] == 'user' for o in lob.q_bid0),
                    'has_user_bid1': any(o['type'] == 'user' for o in lob.q_bid_minus1),
                    'trades': trades_to_emit,
                    'inventory': self.env.inventory,
                    'pnl': self.env.total_pnl,
                    'active_orders': active_orders
                }
                self.update_signal.emit(state)
                last_emit_time = current_time_real
                
            time.sleep(0.001)

    def stop(self):
        self.is_running = False

class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        strings =[]
        for v in values:
            total_seconds = int(v) + (9 * 3600) + (30 * 60)
            h = (total_seconds // 3600) % 24
            m = (total_seconds % 3600) // 60
            s = total_seconds % 60
            strings.append(f"{h:02d}:{m:02d}:{s:02d}")
        return strings

class PriceAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return[f"{v:.2f}" for v in values]

class PriceLadderDOM(QMainWindow):
    def __init__(self, env, agent):
        super().__init__()
        self.setWindowTitle("Hawkes Process - Complete Algo Terminal")
        self.resize(1750, 850) 
        self.setStyleSheet("background-color: #121212;")

        self.num_rows = 41
        self.center_idx = self.num_rows // 2
        nearest_tick = round(env.lob.mid_price / env.lob.tick_size) * env.lob.tick_size
        self.center_price = round(nearest_tick, 4)

        self.cell_cache = {i: {'price': None, 'bid': None, 'ask': None} for i in range(self.num_rows)}
        self.last_time_str = ""

        self.time_data = deque(maxlen=600)
        self.mid_data = deque(maxlen=600)
        self.ask_data = deque(maxlen=600)
        self.bid_data = deque(maxlen=600)
        self.pnl_data = deque(maxlen=600) 
        
        self.current_y_min = None
        self.current_y_max = None

        self.init_ui()

        self.sim_thread = SimulationThread(env, agent=agent, playback_speed=10.0, target_fps=60)
        self.sim_thread.update_signal.connect(self.update_dom)
        self.sim_thread.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        agent_widget = QWidget()
        agent_widget.setFixedWidth(280)
        agent_layout = QVBoxLayout(agent_widget)
        agent_layout.setContentsMargins(10, 0, 10, 0)

        agent_title = QLabel("AGENT DASHBOARD")
        agent_title.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        agent_title.setStyleSheet("color: white; background-color: #1A1A1A; padding: 5px;")
        agent_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        agent_layout.addWidget(agent_title)

        self.lbl_inv = QLabel("INVENTORY: 0")
        self.lbl_inv.setFont(QFont("Courier", 14, QFont.Weight.Bold))
        self.lbl_inv.setStyleSheet("color: #B0B0B0;")
        self.lbl_inv.setAlignment(Qt.AlignmentFlag.AlignCenter)
        agent_layout.addWidget(self.lbl_inv)

        self.lbl_pnl = QLabel("PNL: $0.00")
        self.lbl_pnl.setFont(QFont("Courier", 16, QFont.Weight.Bold))
        self.lbl_pnl.setStyleSheet("color: white;")
        self.lbl_pnl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        agent_layout.addWidget(self.lbl_pnl)

        self.pnl_plot = pg.PlotWidget()
        self.pnl_plot.setFixedHeight(120)
        self.pnl_plot.setBackground('#0A0A0A')
        self.pnl_plot.showGrid(x=True, y=True, alpha=0.15)
        self.pnl_plot.getAxis('left').setPen('#333333')
        self.pnl_plot.hideAxis('bottom') 

        self.pnl_curve = self.pnl_plot.plot(
            pen=pg.mkPen(color=(255, 215, 0, 255), width=2.0),
            fillLevel=0, 
            brush=pg.mkBrush(255, 215, 0, 30)
        )
        agent_layout.addWidget(self.pnl_plot)

        ao_title = QLabel("ACTIVE LIMIT ORDERS")
        ao_title.setFont(QFont("Courier", 10, QFont.Weight.Bold))
        ao_title.setStyleSheet("color: gray; margin-top: 10px;")
        ao_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        agent_layout.addWidget(ao_title)

        self.ao_table = QTableWidget(0, 3)
        self.ao_table.setHorizontalHeaderLabels(["SIDE(Q)", "PRICE", "RESTING"])
        self.ao_table.verticalHeader().setVisible(False)
        self.ao_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ao_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.ao_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.ao_table.setShowGrid(False)
        self.ao_table.setStyleSheet("QTableWidget { background-color: #0A0A0A; border: 1px solid #1A1A1A; } QHeaderView::section { background-color: #1A1A1A; color: gray; font-weight: bold; border: none; }")
        
        ao_header = self.ao_table.horizontalHeader()
        ao_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch) 
        self.ao_table.setColumnWidth(1, 75) 
        self.ao_table.setColumnWidth(2, 65) 

        self.ao_table.setRowCount(10)
        for i in range(10):
            for col in range(3):
                item = QTableWidgetItem("")
                item.setFont(QFont("Courier", 10, QFont.Weight.Bold))
                item.setBackground(pg.mkColor("#1E1E1E"))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.ao_table.setItem(i, col, item)
            self.ao_table.setRowHidden(i, True)

        agent_layout.addWidget(self.ao_table)
        main_layout.addWidget(agent_widget)

        dom_widget = QWidget()
        dom_widget.setFixedWidth(400)
        dom_layout = QVBoxLayout(dom_widget)
        dom_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_time = QLabel("09:30:00.000")
        self.lbl_time.setFont(QFont("Courier", 16, QFont.Weight.Bold))
        self.lbl_time.setStyleSheet("color: #00E676;")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_spread = QLabel("Spread: 0 ticks")
        self.lbl_spread.setFont(QFont("Courier", 10))
        self.lbl_spread.setStyleSheet("color: gray;")
        self.lbl_spread.setAlignment(Qt.AlignmentFlag.AlignCenter)

        dom_layout.addWidget(self.lbl_time)
        dom_layout.addWidget(self.lbl_spread)

        header_layout = QHBoxLayout()
        for text in["BID SIZE", "PRICE", "ASK SIZE"]:
            lbl = QLabel(text)
            lbl.setFont(QFont("Courier", 10, QFont.Weight.Bold))
            lbl.setStyleSheet("color: gray; background-color: #1A1A1A; padding: 5px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header_layout.addWidget(lbl)
        dom_layout.addLayout(header_layout)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(1)
        self.rows =[]

        font_large = QFont("Courier", 11, QFont.Weight.Bold)
        font_med = QFont("Courier", 11)

        for i in range(self.num_rows):
            lbl_bid = QLabel("")
            lbl_price = QLabel("")
            lbl_ask = QLabel("")

            for lbl in (lbl_bid, lbl_price, lbl_ask):
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setMinimumHeight(18) 

            lbl_bid.setFont(font_large)
            lbl_price.setFont(font_med)
            lbl_ask.setFont(font_large)

            grid_layout.addWidget(lbl_bid, i, 0)
            grid_layout.addWidget(lbl_price, i, 1)
            grid_layout.addWidget(lbl_ask, i, 2)

            self.rows.append({'bid': lbl_bid, 'price': lbl_price, 'ask': lbl_ask})

        dom_layout.addLayout(grid_layout)
        main_layout.addWidget(dom_widget)

        ts_widget = QWidget()
        ts_widget.setFixedWidth(280)
        ts_layout = QVBoxLayout(ts_widget)
        ts_layout.setContentsMargins(10, 0, 10, 0)

        ts_title = QLabel("TIME & SALES")
        ts_title.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        ts_title.setStyleSheet("color: gray; padding: 5px;")
        ts_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ts_layout.addWidget(ts_title)

        self.ts_table = QTableWidget(0, 3)
        self.ts_table.setHorizontalHeaderLabels(["TIME", "PRICE", "SIZE"])
        self.ts_table.verticalHeader().setVisible(False)
        self.ts_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ts_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.ts_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.ts_table.setShowGrid(False)
        self.ts_table.setStyleSheet("QTableWidget { background-color: #0A0A0A; border: 1px solid #1A1A1A; } QHeaderView::section { background-color: #1A1A1A; color: gray; font-weight: bold; border: none; }")

        header = self.ts_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch) 
        self.ts_table.setColumnWidth(1, 80) 
        self.ts_table.setColumnWidth(2, 60) 

        ts_layout.addWidget(self.ts_table)
        main_layout.addWidget(ts_widget)

        time_axis = TimeAxisItem(orientation='bottom')
        price_axis = PriceAxisItem(orientation='left')
        
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': time_axis, 'left': price_axis})
        self.plot_widget.setBackground('#0A0A0A') 
        self.plot_widget.setTitle("Real-Time Spreads & Mid-Price", color="#B0B0B0", size="12pt")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.getAxis('bottom').setPen('#333333')
        self.plot_widget.getAxis('left').setPen('#333333')

        self.ask_curve = self.plot_widget.plot(pen=pg.mkPen(color=(255, 82, 82, 200), width=1.5))
        self.bid_curve = self.plot_widget.plot(pen=pg.mkPen(color=(0, 230, 118, 200), width=1.5))
        self.mid_curve = self.plot_widget.plot(pen=pg.mkPen(color=(255, 255, 255, 255), width=2.0))

        self.spread_fill = pg.FillBetweenItem(curve1=self.ask_curve, curve2=self.bid_curve, brush=(255, 255, 255, 25))
        self.plot_widget.addItem(self.spread_fill)

        dashed_pen = pg.mkPen(color=(255, 255, 255, 100), width=1.5, style=Qt.PenStyle.DashLine)
        self.current_price_line = pg.InfiniteLine(angle=0, pen=dashed_pen)
        self.plot_widget.addItem(self.current_price_line)

        main_layout.addWidget(self.plot_widget, stretch=1)

    def format_time(self, seconds_since_open):
        total_seconds = int(seconds_since_open) + (9 * 3600) + (30 * 60)
        h = (total_seconds // 3600) % 24
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        ms = int(round(seconds_since_open % 1, 3) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def update_dom(self, state):
        sim_time = state['time']

        self.time_data.append(sim_time)
        self.pnl_data.append(state['pnl'])
        self.mid_data.append(state['mid'])
        self.ask_data.append(state['p_ask0'])
        self.bid_data.append(state['p_bid0'])
        
        self.lbl_inv.setText(f"INVENTORY: {state['inventory']}")
        pnl = state['pnl']
        sign = "+" if pnl >= 0 else "-"
        color = "#00E676" if pnl > 0 else ("#FF5252" if pnl < 0 else "white")
        self.lbl_pnl.setText(f"PNL: {sign}${abs(pnl):.2f}")
        self.lbl_pnl.setStyleSheet(f"color: {color};")

        self.pnl_curve.setData(list(self.time_data), list(self.pnl_data))
        self.pnl_plot.setXRange(sim_time - 10.0, sim_time, padding=0)

        # RESTORED FIX: Object Pooling for Active Orders!
        self.ao_table.setUpdatesEnabled(False)
        for i in range(10):
            if i < len(state['active_orders']):
                ao = state['active_orders'][i]
                c = "#00E676" if ao['side'] == 'BUY' else "#FF5252"
                
                self.ao_table.item(i, 0).setText(f"{ao['side']} ({ao['queue']})")
                self.ao_table.item(i, 0).setForeground(pg.mkColor(c))
                
                self.ao_table.item(i, 1).setText(f"{ao['price']:.2f}")
                self.ao_table.item(i, 1).setForeground(pg.mkColor(c))
                
                self.ao_table.item(i, 2).setText(str(ao['size']))
                self.ao_table.item(i, 2).setForeground(pg.mkColor(c))
                
                self.ao_table.setRowHidden(i, False)
            else:
                self.ao_table.setRowHidden(i, True)
        self.ao_table.setUpdatesEnabled(True)

        if state['trades']:
            self.ts_table.setUpdatesEnabled(False)
            for trade in state['trades']:
                self.ts_table.insertRow(0)
                t_str = self.format_time(trade['time'])
                p_str = f"{trade['price']:.2f}"
                s_str = f"{trade['size']}"
                
                text_color = "#00E676" if trade['side'] == 'BUY' else "#FF5252"
                bg_color = "#0A0A0A"
                if trade['size'] >= 200: bg_color = "#1A4A29" if trade['side'] == 'BUY' else "#4A1A1A"
                if trade.get('is_agent'): bg_color = "#5C4A00"
                
                item_t = QTableWidgetItem(t_str)
                item_p = QTableWidgetItem(p_str)
                item_s = QTableWidgetItem(s_str)
                
                for item in (item_t, item_p, item_s):
                    item.setFont(QFont("Courier", 10, QFont.Weight.Bold))
                    item.setForeground(pg.mkColor(text_color))
                    item.setBackground(pg.mkColor(bg_color))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                self.ts_table.setItem(0, 0, item_t)
                self.ts_table.setItem(0, 1, item_p)
                self.ts_table.setItem(0, 2, item_s)
                
                if self.ts_table.rowCount() > 50:
                    self.ts_table.removeRow(50)
            
            self.ts_table.scrollToTop()
            self.ts_table.setUpdatesEnabled(True)

        self.mid_curve.setData(list(self.time_data), list(self.mid_data))
        self.ask_curve.setData(list(self.time_data), list(self.ask_data))
        self.bid_curve.setData(list(self.time_data), list(self.bid_data))
        self.plot_widget.setXRange(sim_time - 10.0, sim_time, padding=0)

        if len(self.ask_data) > 0:
            target_y_max = max(self.ask_data) + (4 * state['tick_size'])
            target_y_min = min(self.bid_data) - (4 * state['tick_size'])
            
            if target_y_max - target_y_min < 10 * state['tick_size']:
                mid = (target_y_max + target_y_min) / 2
                target_y_max = mid + 5 * state['tick_size']
                target_y_min = mid - 5 * state['tick_size']

            if self.current_y_min is None:
                self.current_y_min = target_y_min
                self.current_y_max = target_y_max
            else:
                smoothing = 0.1 
                self.current_y_min += (target_y_min - self.current_y_min) * smoothing
                self.current_y_max += (target_y_max - self.current_y_max) * smoothing
                
            self.plot_widget.setYRange(self.current_y_min, self.current_y_max, padding=0)

        self.current_price_line.setValue(state['mid'])

        # 4. Update DOM Ladder
        new_time_str = self.format_time(sim_time)
        if new_time_str != self.last_time_str:
            self.lbl_time.setText(new_time_str)
            self.last_time_str = new_time_str

        self.lbl_spread.setText(f"Spread: {state['spread']} ticks")

        if abs(state['mid'] - self.center_price) > (15 * state['tick_size']):
            self.center_price = round(round(state['mid'] / state['tick_size']) * state['tick_size'], 4)

        active_bids = {
            state['p_bid0']: {'vol': state['q_bid0'], 'user': state['has_user_bid0']}
        }
        active_asks = {
            state['p_ask0']: {'vol': state['q_ask0'], 'user': state['has_user_ask0']}
        }

        for i in range(self.num_rows):
            tick_offset = self.center_idx - i
            row_price = round(self.center_price + (tick_offset * state['tick_size']), 4)
            
            price_text = f"{row_price:.2f}"
            if row_price == state['p_ask0']:
                price_style = "color: #FF5252; background-color: #1E1E1E; font-weight: bold;"
            elif row_price == state['p_bid0']:
                price_style = "color: #00E676; background-color: #1E1E1E; font-weight: bold;"
            else:
                price_style = "color: #B0B0B0; background-color: #1E1E1E;"

            if self.cell_cache[i]['price'] != (price_text, price_style):
                self.rows[i]['price'].setText(price_text)
                self.rows[i]['price'].setStyleSheet(price_style)
                self.cell_cache[i]['price'] = (price_text, price_style)

            if row_price in active_bids and active_bids[row_price]['vol'] > 0:
                bid_text = str(active_bids[row_price]['vol'])
                if active_bids[row_price]['user']: 
                    bid_style = "color: black; background-color: #FFD700; border-right: 2px solid #00E676; font-weight: bold;"
                else:
                    bid_style = "color: white; background-color: #1A4A29; border-right: 2px solid #00E676;"
            else:
                bid_text = ""
                bid_style = "background-color: #0D2614;"

            if self.cell_cache[i]['bid'] != (bid_text, bid_style):
                self.rows[i]['bid'].setText(bid_text)
                self.rows[i]['bid'].setStyleSheet(bid_style)
                self.cell_cache[i]['bid'] = (bid_text, bid_style)

            if row_price in active_asks and active_asks[row_price]['vol'] > 0:
                ask_text = str(active_asks[row_price]['vol'])
                if active_asks[row_price]['user']: 
                    ask_style = "color: black; background-color: #FFD700; border-left: 2px solid #FF5252; font-weight: bold;"
                else:
                    ask_style = "color: white; background-color: #661818; border-left: 2px solid #FF5252;"
            else:
                ask_text = ""
                ask_style = "background-color: #331818;"

            if self.cell_cache[i]['ask'] != (ask_text, ask_style):
                self.rows[i]['ask'].setText(ask_text)
                self.rows[i]['ask'].setStyleSheet(ask_style)
                self.cell_cache[i]['ask'] = (ask_text, ask_style)

        self.sim_thread.gui_ready = True

    def closeEvent(self, event):
        self.sim_thread.is_running = False
        self.sim_thread.wait(500)
        event.accept()

if __name__ == "__main__":
    print("Starting Simulator.")
    
    TARGET_TICKER = "IBM"
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    CONFIG_FILE = os.path.join(ROOT_DIR, "Extracted_Market_Data", TARGET_TICKER, f"{TARGET_TICKER}_config.json")
    
    if not os.path.exists(CONFIG_FILE):
        print(f"{CONFIG_FILE} not found.")
        sys.exit(1)
        
    with open(CONFIG_FILE, 'r') as f:
        cfg = json.load(f)
        
    u_shape_bins = {int(k): v for k, v in cfg['u_shape_bins'].items()}
    def f_Qt(t_seconds):
        return u_shape_bins.get(min(int(t_seconds // 1800), 12), 1.0) 

    size_model = OrderSizeModel(
        lo_config=cfg.get('lo_size_model'), 
        mo_config=cfg.get('mo_size_model'),
        co_config=cfg.get('co_size_model'),
        empty_prob=cfg.get('empty_level_prob', 0.0)
    )

    lob = SixLevelLOB(tick_size=0.01, initial_price=cfg.get('initial_price', 100.00), size_model=size_model)
    
    sim = EightDimHawkesSimulator(
        lob=lob, mu=cfg['mu_base'], alpha=cfg['alpha_matrix'], 
        gamma=cfg['gamma_matrix'], beta_kernel=cfg['beta_matrix'], 
        f_Qt_func=f_Qt, 
        beta_spread=cfg.get('beta_spread', 1.5)
    )
    
    env = LOBBacktestEnv(lob, sim, size_model)
    trading_bot = HJBMarketMaker(action_interval=0.1, lot_size=1)

    app = QApplication(sys.argv)
    window = PriceLadderDOM(env, agent=trading_bot)
    window.setWindowTitle(f"Trading Simulator - {TARGET_TICKER} Terminal")
    window.show()
    sys.exit(app.exec())