# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import numpy as np

from typing import Union
from gym.spaces import Discrete

from tensortrade.actions import ActionStrategy, TradeActionUnion, DTypeString
from tensortrade.trades import Trade, TradeType


class TargetStopActionStrategy(ActionStrategy):

    def __init__(self, instrument_symbol: str = 'BTC', position_size: int = 20, profit_target_range: range = range(0, 100, 5),
                 stop_loss_range: range = range(0, 100, 5), timeout_steps: int = np.iinfo.max):
        """
        Arguments:
            instrument_symbol: The exchange symbol of the instrument being traded.
                Defaults to 'BTC'.
            position_size: The number of bins to divide the total balance by for each trade position.
                Defaults to 20 (i.e 1/20, 2/20 ... 20/20).
            profit_target_range: The range of percentages for the profit target of each trade position.
                Defaults to range(0, 100, 5).
            stop_loss_range: The range of percentages for the stop loss of each trade position.
                Defaults to range(0, 100, 5).
            timeout_steps (optional): Number of timesteps allowed per trade before automatically selling at market. 
        """

        super().__init__(action_space=Discrete(position_size), dtype=np.int64)

        self.position_size = position_size
        self.instrument_symbol = instrument_symbol
        self.profit_target_range = profit_target_range
        self.stop_loss_range = stop_loss_range
        self.timeout_steps = timeout_steps

        self.reset()

    @property
    def dtype(self) -> DTypeString:
        """A type or str corresponding to the dtype of the `action_space`."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeString):
        raise ValueError(
            'Cannot change the dtype of a `TargetStopActionStrategy` due to the requirements of `gym.spaces.Discrete` spaces. ')

    def reset(self):
        self.current_step = 0
        self.trading_history = list([])

    def get_trade(self, action: TradeActionUnion) -> Trade:
        """The trade type is determined by `action % len(TradeType)`, and the
        trade amount is determined by the multiplicity of the action.

        For example, TODO: GIVE EXAMPLE INPUT -> OUTPUT
        """
        # TODO: Fix profit target, stop loss, and trade percentages.
        # They are currently calculated incorrectly.

        n_pt_splits = len(self.profit_target_range)
        profit_target_percent = float(action * n_pt_splits / n_pt_splits) + (1 / n_pt_splits)

        n_sl_splits = len(self.stop_loss_range)
        stop_loss_percent = float(action * n_sl_splits / n_sl_splits) + (1 / n_sl_splits)

        n_splits = self.position_size / len(TradeType)
        trade_type = TradeType(action % len(TradeType))
        trade_amount = int(action / len(TradeType)) * float(1 / n_splits) + (1 / n_splits)

        current_price = self._exchange.current_price(symbol=self.instrument_symbol)
        base_precision = self._exchange.base_precision
        instrument_precision = self._exchange.instrument_precision

        amount = self._exchange.instrument_balance(self.instrument_symbol)
        current_price = round(current_price, base_precision)
        current_step = self.current_step

        self.current_step += 1

        for trade, idx in enumerate(self.trading_history):
            profit_target_hit = current_price >= (trade[1] * trade[3])
            stop_loss_hit = current_price <= (trade[1] * trade[4])
            timeout_hit = current_step - trade[0] >= self.timeout_steps

            if profit_target_hit or stop_loss_hit or timeout_hit:
                amount = self._exchange.portfolio.get(self.instrument_symbol, 0)

                if amount >= trade[2]:
                    amount = trade[2]

                del self.trading_history[idx]

                return Trade(self.instrument_symbol, TradeType.MARKET_SELL, amount, current_price)

        if TradeType is TradeType.MARKET_BUY or TradeType is TradeType.LIMIT_BUY:
            amount = round(self._exchange.balance * 0.99 *
                           trade_amount / current_price, instrument_precision)

            self.trading_history.append(
                [current_step, current_price, amount, profit_target_percent, stop_loss_percent])

        return Trade(self.instrument_symbol, trade_type, amount, current_price)
