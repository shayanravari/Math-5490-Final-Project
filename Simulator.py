import sys
import time
import numpy as np
import pandas as pd
import pyqtgraph as pg
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont



class OrderSizeModel:
    def __init__(self):
        self.round_numbers =[1, 10, 50, 100, 200, 500]
        self.round_probs =[0.1, 0.3, 0.2, 0.2, 0.1, 0.1]
        self.geometric_p = 0.05
        self.p_round = 0.75

    def sample_limit_market_order(self):
        if np.random.rand() < self.p_round:
            return np.random.choice(self.round_numbers, p=self.round_probs)
        else:
            return np.random.geometric(self.geometric_p)

    def sample_cancel_order(self, queue_orders):
        sim_indices =[i for i, order in enumerate(queue_orders) if order['type'] == 'sim']
        
        if not sim_indices:
            return 0
            
        idx = np.random.choice(sim_indices)
        canceled_order = queue_orders.pop(idx)
        return canceled_order['size']
        
    def sample_q_plus2(self):
        return self.sample_limit_market_order() * 5

class SixLevelLOB:
    def __init__(self, tick_size=0.01, initial_price=100.00, size_model=None):
        self.tick_size = tick_size
        self.size_model = size_model or OrderSizeModel()
        
        self.p_ask0 = round(initial_price, 4)
        self.p_bid0 = round(initial_price - self.tick_size, 4)
        
        # Queues are now lists of dictionaries
        self.q_ask_plus1 =[{'type': 'sim', 'size': 100}, {'type': 'sim', 'size': 50}]
        self.q_ask0 =[{'type': 'sim', 'size': 50}, {'type': 'sim', 'size': 50}]
        self.q_bid0 =[{'type': 'sim', 'size': 50}, {'type': 'sim', 'size': 50}]
        self.q_bid_minus1 =[{'type': 'sim', 'size': 100}, {'type': 'sim', 'size': 50}]

    @property
    def spread_in_ticks(self):
        spread_value = round(self.p_ask0 - self.p_bid0, 4)
        return max(1, round(spread_value / self.tick_size))

    @property
    def mid_price(self):
        return round((self.p_ask0 + self.p_bid0) / 2.0, 4)

    def total_q(self, queue):
        # Sums up the 'size' key from every dictionary in the list
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
        user_fills = []
        
        while remaining_vol > 0 and remaining_vol >= self.total_q(self.q_ask0 if side == 'ask' else self.q_bid0):
            target_q = self.q_ask0 if side == 'ask' else self.q_bid0
            
            # Check for user fills before wiping the whole queue
            for order in target_q:
                if order['type'] == 'user':
                    user_fills.append({'price': self.p_ask0 if side == 'ask' else self.p_bid0, 'size': order['size']})

            if side == 'ask':
                remaining_vol -= self.total_q(self.q_ask0)
                self.q_ask0 = self.q_ask_plus1.copy()
                self.p_ask0 = round(self.p_ask0 + self.tick_size, 4)
                self.q_ask_plus1 =[{'type': 'sim', 'size': self.size_model.sample_q_plus2()}]
            else:
                remaining_vol -= self.total_q(self.q_bid0)
                self.q_bid0 = self.q_bid_minus1.copy()
                self.p_bid0 = round(self.p_bid0 - self.tick_size, 4)
                self.q_bid_minus1 =[{'type': 'sim', 'size': self.size_model.sample_q_plus2()}]

        if remaining_vol > 0:
            target_q = self.q_ask0 if side == 'ask' else self.q_bid0
            while remaining_vol > 0 and target_q:
                if target_q[0]['size'] <= remaining_vol:
                    filled_order = target_q.pop(0)
                    remaining_vol -= filled_order['size']
                    # Log if this specific order belonged to the user
                    if filled_order['type'] == 'user':
                        user_fills.append({'price': self.p_ask0 if side == 'ask' else self.p_bid0, 'size': filled_order['size']})
                else:
                    target_q[0]['size'] -= remaining_vol
                    # Log the partial fill
                    if target_q[0]['type'] == 'user':
                        user_fills.append({'price': self.p_ask0 if side == 'ask' else self.p_bid0, 'size': remaining_vol})
                    remaining_vol = 0
                    
        return user_fills

    def add_limit_order(self, queue_name, volume, order_type='sim'):
        order = {'type': order_type, 'size': volume}
        if queue_name == 'ask+1': self.q_ask_plus1.append(order)
        elif queue_name == 'ask0': self.q_ask0.append(order)
        elif queue_name == 'bid0': self.q_bid0.append(order)
        elif queue_name == 'bid-1': self.q_bid_minus1.append(order)

    def execute_market_order(self, side, volume):
        return self.trigger_queue_depletion(side, volume)

class TwelveDimHawkesSimulator:
    def __init__(self, lob, mu, alpha, gamma, beta_kernel, f_Qt_func, beta_spread=1.5):
        self.lob = lob
        self.mu = np.array(mu)
        
        self.alpha = np.array(alpha)
        self.gamma = np.array(gamma)
        self.beta_kernel = np.array(beta_kernel)
        
        self.f_Qt_func = f_Qt_func
        self.beta_spread = beta_spread
        self.dim = 12
        
        self.max_hist = 500
        self.hist_t = np.zeros(self.max_hist)
        self.hist_j = np.zeros(self.max_hist, dtype=int)
        self.hist_len = 0

    def calculate_intensities(self, t):
        s = self.lob.spread_in_ticks
        f_qt = self.f_Qt_func(t)
        lambda_t = np.copy(self.mu)
        
        if self.hist_len > 0:
            t_past = self.hist_t[:self.hist_len]
            j_past = self.hist_j[:self.hist_len]
            
            dt = t - t_past 
            
            # self.alpha[:, j_past] grabs the columns corresponding to past event types
            # Resulting arrays are shape (12, N)
            a = self.alpha[:, j_past]
            g = self.gamma[:, j_past]
            b = self.beta_kernel[:, j_past]
            
            # Calculate all 12xN kernel values
            kernel_vals = a * (1 + g * dt)**(-b)
            
            # Sum across the history axis to get the 12 base values
            lambda_t += np.sum(kernel_vals, axis=1)

        lambda_t *= f_qt
        
        # Spread modifier for in-spread events
        if s > 1:
            is_multiplier = (s - 1) ** self.beta_spread
            lambda_t[5] *= is_multiplier
            lambda_t[6] *= is_multiplier
        else:
            lambda_t[5] = 0.0
            lambda_t[6] = 0.0
            
        return np.maximum(0, lambda_t)

    def _add_to_history(self, t, event_id):
        # Fast rolling history buffer
        if self.hist_len < self.max_hist:
            self.hist_t[self.hist_len] = t
            self.hist_j[self.hist_len] = event_id
            self.hist_len += 1
        else:
            self.hist_t[:-1] = self.hist_t[1:]
            self.hist_j[:-1] = self.hist_j[1:]
            self.hist_t[-1] = t
            self.hist_j[-1] = event_id

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
            
            # Acceptance prob
            if np.random.uniform(0, lambda_sum) <= lambda_sum_new:
                probs = lambda_t_new / lambda_sum_new
                event_id = np.random.choice(self.dim, p=probs)
                self._add_to_history(t, event_id)
                return t, event_id
            
class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        """ Converts simulation seconds on the X-axis into HH:MM:SS """
        strings =[]
        for v in values:
            total_seconds = int(v) + (9 * 3600) + (30 * 60) # Start at 09:30:00
            h = (total_seconds // 3600) % 24
            m = (total_seconds % 3600) // 60
            s = total_seconds % 60
            strings.append(f"{h:02d}:{m:02d}:{s:02d}")
        return strings

class PriceAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return[f"{v:.2f}" for v in values]
    
class LOBBacktestEnv:
    def __init__(self, lob, simulator, size_model):
        self.lob = lob
        self.simulator = simulator
        self.size_model = size_model
        self.current_time = 0.0
        self.recent_trades =[] 

    def step(self, user_actions=None):
        # 1. PROCESS USER ALGORITHM ACTIONS FIRST
        if user_actions:
            for action in user_actions:
                if action['type'] == 'limit':
                    # Agent places an order at a specific price
                    fills = self.lob.place_custom_limit_order(
                        side=action['side'], 
                        price=action['price'], 
                        volume=action['size'], 
                        order_type='user'
                    )
                    
                    # If the limit order was priced so aggressively it crossed the spread
                    if fills:
                        for fill in fills:
                            print(f"\n[!] LO Crossed Spread Instantly: {fill['size']} shares at ${fill['price']:.2f}")
                            
                            # Log to Time & Sales Blotter
                            self.recent_trades.append({
                                'time': self.current_time, 
                                'side': 'BUY' if action['side'] == 'buy' else 'SELL', 
                                'price': fill['price'], 
                                'size': fill['size']
                            })
                            
                elif action['type'] == 'market':
                    # Standard Market Order
                    side = action['side']
                    exec_price = self.lob.p_ask0 if side == 'buy' else self.lob.p_bid0
                    
                    self.recent_trades.append({
                        'time': self.current_time, 
                        'side': 'BUY' if side == 'buy' else 'SELL', 
                        'price': exec_price, 
                        'size': action['size']
                    })
                    
                    self.lob.execute_market_order('ask' if side == 'buy' else 'bid', action['size'])

        next_time, event_id = self.simulator.simulate_next_event(self.current_time)
        self.current_time = next_time
        
        if event_id in[0, 2, 7, 10]:
            vol = self.size_model.sample_limit_market_order()
            if event_id == 0: self.lob.add_limit_order('ask+1', vol)
            elif event_id == 2: self.lob.add_limit_order('ask0', vol)
            elif event_id == 7: self.lob.add_limit_order('bid0', vol)
            elif event_id == 10: self.lob.add_limit_order('bid-1', vol)
            
        elif event_id in[5, 6]:
            vol = self.size_model.sample_limit_market_order()
            if event_id == 5: self.lob.trigger_in_spread('ask', vol)
            elif event_id == 6: self.lob.trigger_in_spread('bid', vol)
            
        elif event_id in [4, 9]:
            vol = self.size_model.sample_limit_market_order()
            user_fills =[]
            
            if event_id == 4: # Simulated Market Buy
                self.recent_trades.append({'time': self.current_time, 'side': 'BUY', 'price': self.lob.p_ask0, 'size': vol})
                user_fills = self.lob.execute_market_order('ask', vol)
            elif event_id == 9: # Simulated Market Sell
                self.recent_trades.append({'time': self.current_time, 'side': 'SELL', 'price': self.lob.p_bid0, 'size': vol})
                user_fills = self.lob.execute_market_order('bid', vol)
                
            if user_fills:
                for fill in user_fills:
                    print(f"\n[!] User LO Fill: {fill['size']} shares executed at ${fill['price']:.2f}")
                
        elif event_id in[1, 3, 8, 11]:
            if event_id == 1: self.size_model.sample_cancel_order(self.lob.q_ask_plus1)
            elif event_id == 3:
                self.size_model.sample_cancel_order(self.lob.q_ask0)
                if self.lob.total_q(self.lob.q_ask0) == 0: self.lob.trigger_queue_depletion('ask', 0)
            elif event_id == 8:
                self.size_model.sample_cancel_order(self.lob.q_bid0)
                if self.lob.total_q(self.lob.q_bid0) == 0: self.lob.trigger_queue_depletion('bid', 0)
            elif event_id == 11: self.size_model.sample_cancel_order(self.lob.q_bid_minus1)

        return event_id

class SimulationThread(QThread):
    update_signal = pyqtSignal(dict)

    def __init__(self, env, playback_speed=1.0, target_fps=144):
        super().__init__()
        self.env = env
        self.playback_speed = playback_speed
        self.is_running = True
        self.frame_interval = 1.0 / target_fps 

    def run(self):
        start_time_real = time.perf_counter()
        start_time_sim = self.env.current_time
        last_emit_time = time.perf_counter()

        while self.is_running:
            current_time_real = time.perf_counter()
            elapsed_real = (current_time_real - start_time_real) * self.playback_speed
            target_sim_time = start_time_sim + elapsed_real

            while self.env.current_time <= target_sim_time:
                self.env.step()

            if current_time_real - last_emit_time >= self.frame_interval: 
                lob = self.env.lob
                trades_to_emit = list(self.env.recent_trades)
                self.env.recent_trades.clear()

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
                    'trades': trades_to_emit
                }
                self.update_signal.emit(state)
                last_emit_time = current_time_real
                
            time.sleep(0)

    def stop(self):
        self.is_running = False
        self.wait()

class PriceLadderDOM(QMainWindow):
    def __init__(self, env):
        super().__init__()
        self.setWindowTitle("Simulator Terminal")
        self.resize(1500, 800) 
        self.setStyleSheet("background-color: #121212;")

        self.num_rows = 25
        self.center_idx = self.num_rows // 2
        
        nearest_tick = round(env.lob.mid_price / env.lob.tick_size) * env.lob.tick_size
        self.center_price = round(nearest_tick, 4)

        self.cell_cache = {i: {'price': None, 'bid': None, 'ask': None} for i in range(self.num_rows)}
        self.last_time_str = ""

        self.time_data = deque(maxlen=1500)
        self.mid_data = deque(maxlen=1500)
        self.ask_data = deque(maxlen=1500)
        self.bid_data = deque(maxlen=1500)

        self.init_ui()

        self.sim_thread = SimulationThread(env, playback_speed=1.0, target_fps=144)
        self.sim_thread.update_signal.connect(self.update_dom)
        self.sim_thread.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

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
        for text in ["BID SIZE", "PRICE", "ASK SIZE"]:
            lbl = QLabel(text)
            lbl.setFont(QFont("Courier", 10, QFont.Weight.Bold))
            lbl.setStyleSheet("color: gray; background-color: #1A1A1A; padding: 5px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header_layout.addWidget(lbl)
        dom_layout.addLayout(header_layout)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(1)
        self.rows =[]

        font_large = QFont("Courier", 12, QFont.Weight.Bold)
        font_med = QFont("Courier", 12)

        for i in range(self.num_rows):
            lbl_bid = QLabel("")
            lbl_price = QLabel("")
            lbl_ask = QLabel("")

            for lbl in (lbl_bid, lbl_price, lbl_ask):
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setMinimumHeight(24)

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

        # Setup Table
        self.ts_table = QTableWidget(0, 3)
        self.ts_table.setHorizontalHeaderLabels(["TIME", "PRICE", "SIZE"])
        self.ts_table.verticalHeader().setVisible(False)
        self.ts_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ts_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.ts_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.ts_table.setShowGrid(False)

        # Table Styling
        self.ts_table.setStyleSheet("""
            QTableWidget { background-color: #0A0A0A; border: 1px solid #1A1A1A; }
            QHeaderView::section { background-color: #1A1A1A; color: gray; font-weight: bold; border: none; }
        """)

        # Column Widths
        header = self.ts_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch) # Time
        self.ts_table.setColumnWidth(1, 80) # Price
        self.ts_table.setColumnWidth(2, 60) # Size

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

        if state['trades']:
            self.ts_table.setUpdatesEnabled(False)
            
            for trade in state['trades']:
                self.ts_table.insertRow(0)
                
                t_str = self.format_time(trade['time'])
                p_str = f"{trade['price']:.2f}"
                s_str = f"{trade['size']}"
                
                text_color = "#00E676" if trade['side'] == 'BUY' else "#FF5252"
                bg_color = "#0A0A0A"
                
                if trade['size'] >= 200:
                    bg_color = "#1A4A29" if trade['side'] == 'BUY' else "#4A1A1A"
                
                # Create Table Items
                item_t = QTableWidgetItem(t_str)
                item_p = QTableWidgetItem(p_str)
                item_s = QTableWidgetItem(s_str)
                
                # Style them
                for item in (item_t, item_p, item_s):
                    item.setFont(QFont("Courier", 10, QFont.Weight.Bold))
                    item.setForeground(pg.mkColor(text_color))
                    item.setBackground(pg.mkColor(bg_color))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                self.ts_table.setItem(0, 0, item_t)
                self.ts_table.setItem(0, 1, item_p)
                self.ts_table.setItem(0, 2, item_s)
                
                if self.ts_table.rowCount() > 500:
                    self.ts_table.removeRow(500)
            
            self.ts_table.scrollToTop()
            self.ts_table.setUpdatesEnabled(True)

        # Update the chart data
        self.time_data.append(sim_time)
        self.mid_data.append(state['mid'])
        self.ask_data.append(state['p_ask0'])
        self.bid_data.append(state['p_bid0'])

        self.mid_curve.setData(self.time_data, self.mid_data)
        self.ask_curve.setData(self.time_data, self.ask_data)
        self.bid_curve.setData(self.time_data, self.bid_data)

        self.plot_widget.setXRange(sim_time - 10.0, sim_time, padding=0)

        if len(self.ask_data) > 0:
            max_visible = max(self.ask_data)
            min_visible = min(self.bid_data)
            min_range = 10 * state['tick_size'] 
            current_range = max_visible - min_visible
            
            if current_range < min_range:
                padding = (min_range - current_range) / 2
                max_visible += padding
                min_visible -= padding
                
            self.plot_widget.setYRange(min_visible - (2 * state['tick_size']), max_visible + (2 * state['tick_size']), padding=0)

        self.current_price_line.setValue(state['mid'])

        # UPDATE the DOM Ladder
        new_time_str = self.format_time(sim_time)
        if new_time_str != self.last_time_str:
            self.lbl_time.setText(new_time_str)
            self.last_time_str = new_time_str

        self.lbl_spread.setText(f"Spread: {state['spread']} ticks")

        if abs(state['mid'] - self.center_price) > (8 * state['tick_size']):
            self.center_price = round(round(state['mid'] / state['tick_size']) * state['tick_size'], 4)

        active_bids = {
            state['p_bid0']: state['q_bid0'],
            round(state['p_bid0'] - state['tick_size'], 4): state['q_bid1']
        }
        active_asks = {
            state['p_ask0']: state['q_ask0'],
            round(state['p_ask0'] + state['tick_size'], 4): state['q_ask1']
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

            if row_price in active_bids and active_bids[row_price] > 0:
                bid_text = str(active_bids[row_price])
                bid_style = "color: white; background-color: #1A4A29; border-right: 2px solid #00E676;"
            else:
                bid_text = ""
                bid_style = "background-color: #0D2614;"

            if self.cell_cache[i]['bid'] != (bid_text, bid_style):
                self.rows[i]['bid'].setText(bid_text)
                self.rows[i]['bid'].setStyleSheet(bid_style)
                self.cell_cache[i]['bid'] = (bid_text, bid_style)

            if row_price in active_asks and active_asks[row_price] > 0:
                ask_text = str(active_asks[row_price])
                ask_style = "color: white; background-color: #661818; border-left: 2px solid #FF5252;"
            else:
                ask_text = ""
                ask_style = "background-color: #331818;"

            if self.cell_cache[i]['ask'] != (ask_text, ask_style):
                self.rows[i]['ask'].setText(ask_text)
                self.rows[i]['ask'].setStyleSheet(ask_style)
                self.cell_cache[i]['ask'] = (ask_text, ask_style)

    def closeEvent(self, event):
        self.sim_thread.stop()
        event.accept()

if __name__ == "__main__":
    print("Starting LOB...")
    
    def f_Qt(t): return 1.0 
    mu_base =[0.5, 0.4, 0.8, 0.7, 0.2, 0.1, 0.1, 0.8, 0.7, 0.2, 0.5, 0.4]
    
    alpha_matrix = np.full((12, 12), 0.01)
    np.fill_diagonal(alpha_matrix, 0.05)
    gamma_matrix = np.full((12, 12), 1.0)
    beta_matrix = np.full((12, 12), 1.5)

    size_model = OrderSizeModel()
    lob = SixLevelLOB(tick_size=0.01, initial_price=100.00, size_model=size_model)
    sim = TwelveDimHawkesSimulator(lob, mu_base, alpha_matrix, gamma_matrix, beta_matrix, f_Qt, beta_spread=1.5)
    env = LOBBacktestEnv(lob, sim, size_model)

    app = QApplication(sys.argv)
    window = PriceLadderDOM(env)
    window.show()
    sys.exit(app.exec())
    


    u_shape_bins = {0: 2.0, 1: 1.5, 2: 1.2, 3: 1.0, 4: 0.8, 5: 0.6, 6: 0.5, 
                    7: 0.6, 8: 0.8, 9: 1.0, 10: 1.2, 11: 1.6, 12: 2.5}
    def f_Qt(t_seconds):
        return u_shape_bins[min(int(t_seconds // 1800), 12)]
    
    # Base intensities and matrices
    mu_base =[0.5, 0.4, 0.8, 0.7, 0.2, 0.1, 0.1, 0.8, 0.7, 0.2, 0.5, 0.4]
    alpha_matrix = np.full((12, 12), 0.01)
    np.fill_diagonal(alpha_matrix, 0.05)
    gamma_matrix = np.full((12, 12), 1.0)
    beta_matrix = np.full((12, 12), 1.5)

    # Initialize components
    size_model = OrderSizeModel()
    lob = SixLevelLOB(tick_size=0.01, initial_price=100.00, size_model=size_model)
    sim = TwelveDimHawkesSimulator(lob, mu_base, alpha_matrix, gamma_matrix, beta_matrix, f_Qt, beta_spread=1.5)
    env = LOBBacktestEnv(lob, sim, size_model)

    TOTAL_TRADING_SECONDS = 23400.0 # 6.5 hours
    daily_history =[]
    event_count = 0
    
    print("Starting Simulation...")
    
    while env.current_time < TOTAL_TRADING_SECONDS:
        event_id, event_name = env.step()
        event_count += 1

        if event_count % 10 == 0:
            daily_history.append({
                'time_sec': env.current_time,
                'mid_price': lob.mid_price,
                'spread_ticks': lob.spread_in_ticks,
                'ask_vol': lob.total_q(lob.q_ask0),
                'bid_vol': lob.total_q(lob.q_bid0)
            })

    print(f"Generated {event_count} total events.")

    df = pd.DataFrame(daily_history)

    market_open = pd.to_datetime('2026-04-26 09:30:00')
    df['timestamp'] = market_open + pd.to_timedelta(df['time_sec'], unit='s')
    df.set_index('timestamp', inplace=True)
    
    df_minute = df.resample('1min').agg({
        'mid_price': 'last',
        'spread_ticks': 'mean',
        'ask_vol': 'mean',
        'bid_vol': 'mean'
    }).ffill()


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    time_fmt = mdates.DateFormatter('%H:%M')
    ax1.plot(df_minute.index, df_minute['mid_price'], label='Simulated Mid Price', color='blue', linewidth=1.5)
    
    # TODO: When you have empirical data, update this ...
    # ax1.plot(empirical_df.index, empirical_df['mid_price'], label='Empirical Mid Price', color='orange', alpha=0.7)
    
    ax1.set_title("Simulated Limit Order Book (T = 6.5 hours)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Mid Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(df_minute.index, df_minute['spread_ticks'], label='Simulated Spread (Ticks)', color='red', linewidth=1.2)
    # ax2.plot(empirical_df.index, empirical_df['spread_ticks'], label='Empirical Spread', color='orange', alpha=0.7)
    ax2.set_ylabel("Spread (Ticks)")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax3.plot(df_minute.index, df_minute['ask_vol'], label='Ask0 Volume', color='green', alpha=0.7)
    ax3.plot(df_minute.index, df_minute['bid_vol'], label='Bid0 Volume', color='purple', alpha=0.7)
    ax3.set_ylabel("Queue Volume")
    ax3.set_xlabel("Time of Day (EST)")
    ax3.xaxis.set_major_formatter(time_fmt)
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()