# -*- coding: utf-8 -*-
# ml/rl_trader.py

import numpy as np
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


class RLTrader:
    """معامله‌گر با یادگیری تقویتی برای بهینه‌سازی تصمیم‌گیری"""

    def __init__(self, state_size, action_size=3):  # 0: نگه داشتن, 1: خرید, 2: فروش
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # ضریب تخفیف
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """ساخت مدل شبکه عصبی برای DQN"""
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """ذخیره تجربه در حافظه"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """انتخاب عمل بر اساس حالت فعلی"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)  # exploration

        state = state.reshape(1, -1)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # exploitation

    def replay(self, batch_size=32):
        """آموزش مدل بر اساس تجربیات گذشته"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = next_state.reshape(1, -1)
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state, verbose=0)[0]
                )

            state = state.reshape(1, -1)
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_state(self, df, current_step, window_size=10):
        """دریافت حالت فعلی از داده‌های بازار"""
        if current_step < window_size:
            return np.zeros(self.state_size)

        state = []

        # داده‌های قیمت
        prices = df["close"].iloc[current_step - window_size : current_step].values
        state.extend(prices / prices[0] - 1)  # نرمال‌سازی

        # حجم
        volumes = df["volume"].iloc[current_step - window_size : current_step].values
        state.extend(volumes / (volumes.mean() + 1e-10))

        # اندیکاتورهای تکنیکال
        state.append(df["RSI"].iloc[current_step] / 100)  # نرمال‌سازی RSI
        state.append(df["MACD"].iloc[current_step])
        state.append(df["ADX"].iloc[current_step] / 100)  # نرمال‌سازی ADX

        # موقعیت فعلی (اگر داریم)
        if "position" in df.columns:
            state.append(df["position"].iloc[current_step])
        else:
            state.append(0)

        return np.array(state)


class RLIntegration:
    """یکپارچه‌سازی یادگیری تقویتی با سیستم موجود"""

    def __init__(self, state_size=25):
        self.rl_traders = {}  # {symbol: RLTrader}
        self.state_size = state_size

    def initialize_rl_for_symbol(self, symbol):
        """مقداردهی اولیه RL برای یک نماد"""
        if symbol not in self.rl_traders:
            self.rl_traders[symbol] = RLTrader(self.state_size)
            print(f"🤖 RL Trader برای {symbol} مقداردهی شد")

    def train_rl_trader(self, df, symbol, episodes=100):
        """آموزش RL تریدر بر اساس داده‌های تاریخی"""
        from config import RL_SETTINGS

        episodes = RL_SETTINGS["episodes"]

        self.initialize_rl_for_symbol(symbol)
        rl_trader = self.rl_traders[symbol]

        print(f"🎯 شروع آموزش RL برای {symbol}...")

        for episode in range(episodes):
            total_reward = 0
            position = 0
            entry_price = 0

            for step in range(20, len(df) - 1):
                state = rl_trader.get_state(df, step)
                action = rl_trader.act(state)

                # اجرای عمل
                current_price = df["close"].iloc[step]
                next_price = df["close"].iloc[step + 1]
                reward = 0

                if action == 1 and position == 0:  # خرید
                    position = 1
                    entry_price = current_price
                    reward = -0.001  # هزینه معامله

                elif action == 2 and position == 1:  # فروش
                    pnl = (current_price - entry_price) / entry_price
                    reward = pnl - 0.002  # هزینه معامله
                    position = 0
                    total_reward += pnl

                elif action == 0:  # نگه داشتن
                    if position == 1:
                        unrealized_pnl = (current_price - entry_price) / entry_price
                        reward = unrealized_pnl * 0.1
                    else:
                        reward = 0

                next_state = rl_trader.get_state(df, step + 1)
                done = step == len(df) - 2

                rl_trader.remember(state, action, reward, next_state, done)

                if len(rl_trader.memory) > 32:
                    rl_trader.replay(32)

            if episode % 20 == 0:
                print(
                    f"📈 Episode {episode}: Total Reward: {total_reward:.4f}, Epsilon: {rl_trader.epsilon:.3f}"
                )

        print(f"✅ آموزش RL برای {symbol} کامل شد")

    def get_rl_signal(self, df, symbol):
        """دریافت سیگنال از RL تریدر"""
        if symbol not in self.rl_traders:
            return 0

        rl_trader = self.rl_traders[symbol]
        current_state = rl_trader.get_state(df, len(df) - 1)
        action = rl_trader.act(current_state)

        # تبدیل action به signal
        action_map = {0: 0, 1: 1, 2: -1}
        return action_map[action]
