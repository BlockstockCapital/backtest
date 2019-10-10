import matplotlib.pyplot as plt
import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractStatistics(object):
    """
    Statistics is an abstract class providing an interface for
    all inherited statistic classes (live, historic, custom, etc).
    The goal of a Statistics object is to keep a record of useful
    information about one or many trading strategies as the strategy
    is running. This is done by hooking into the main event loop and
    essentially updating the object according to portfolio performance
    over time.
    Ideally, Statistics should be subclassed according to the strategies
    and timeframes-traded by the user. Different trading strategies
    may require different metrics or frequencies-of-metrics to be updated,
    however the example given is suitable for longer timeframes.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self):
        """
        Update all the statistics according to values of the portfolio
        and open positions. This should be called from within the
        event loop.
        """
        raise NotImplementedError("Should implement update()")

    @abstractmethod
    def get_results(self):
        """
        Return a dict containing all statistics.
        """
        raise NotImplementedError("Should implement get_results()")

    @abstractmethod
    def plot_results(self):
        """
        Plot all statistics collected up until 'now'
        """
        raise NotImplementedError("Should implement plot_results()")


class TearsheetStatistics(AbstractStatistics):
    """
    Displays a Matplotlib-generated 'one-pager' as often
    found in institutional strategy performance reports.
    Includes an equity curve, drawdown curve, monthly
    returns heatmap, yearly returns summary, strategy-
    level statistics and trade-level statistics.
    Also includes an optional annualised rolling Sharpe
    ratio chart.
    """
    def __init__(
        self, config, portfolio_handler,
        title=None, benchmark=None, periods=252,
        rolling_sharpe=False
    ):
        """
        Takes in a portfolio handler.
        """
        self.config = config
        self.portfolio_handler = portfolio_handler
        self.price_handler = portfolio_handler.price_handler
        self.title = '\n'.join(title)
        self.benchmark = benchmark
        self.periods = periods
        self.rolling_sharpe = rolling_sharpe
        self.equity = {}
        self.equity_benchmark = {}
        self.log_scale = False

    def update(self, timestamp, portfolio_handler):
        """
        Update equity curve and benchmark equity curve that must be tracked
        over time.
        """
        self.equity[timestamp] = PriceParser.display(
            self.portfolio_handler.portfolio.equity
        )
        if self.benchmark is not None:
            self.equity_benchmark[timestamp] = PriceParser.display(
                self.price_handler.get_last_close(self.benchmark)
            )

    def get_results(self):
        """
        Return a dict with all important results & stats.
        """
        # Equity
        equity_s = pd.Series(self.equity).sort_index()

        # Returns
        returns_s = equity_s.pct_change().fillna(0.0)

        # Rolling Annualised Sharpe
        rolling = returns_s.rolling(window=self.periods)
        rolling_sharpe_s = np.sqrt(self.periods) * (
            rolling.mean() / rolling.std()
        )

        # Cummulative Returns
        cum_returns_s = np.exp(np.log(1 + returns_s).cumsum())

        # Drawdown, max drawdown, max drawdown duration
        dd_s, max_dd, dd_dur = perf.create_drawdowns(cum_returns_s)

        statistics = {}

        # Equity statistics
        statistics["sharpe"] = perf.create_sharpe_ratio(
            returns_s, self.periods
        )
        statistics["drawdowns"] = dd_s
        # TODO: need to have max_drawdown so it can be printed at end of test
        statistics["max_drawdown"] = max_dd
        statistics["max_drawdown_pct"] = max_dd
        statistics["max_drawdown_duration"] = dd_dur
        statistics["equity"] = equity_s
        statistics["returns"] = returns_s
        statistics["rolling_sharpe"] = rolling_sharpe_s
        statistics["cum_returns"] = cum_returns_s

        positions = self._get_positions()
        if positions is not None:
            statistics["positions"] = positions

        # Benchmark statistics if benchmark ticker specified
        if self.benchmark is not None:
            equity_b = pd.Series(self.equity_benchmark).sort_index()
            returns_b = equity_b.pct_change().fillna(0.0)
            rolling_b = returns_b.rolling(window=self.periods)
            rolling_sharpe_b = np.sqrt(self.periods) * (
                rolling_b.mean() / rolling_b.std()
            )
            cum_returns_b = np.exp(np.log(1 + returns_b).cumsum())
            dd_b, max_dd_b, dd_dur_b = perf.create_drawdowns(cum_returns_b)
            statistics["sharpe_b"] = perf.create_sharpe_ratio(returns_b)
            statistics["drawdowns_b"] = dd_b
            statistics["max_drawdown_pct_b"] = max_dd_b
            statistics["max_drawdown_duration_b"] = dd_dur_b
            statistics["equity_b"] = equity_b
            statistics["returns_b"] = returns_b
            statistics["rolling_sharpe_b"] = rolling_sharpe_b
            statistics["cum_returns_b"] = cum_returns_b

        return statistics


    def plot_results(self, filename=None):
        """
        Plot the Tearsheet
        """
        rc = {
            'lines.linewidth': 1.0,
            'axes.facecolor': '0.995',
            'figure.facecolor': '0.97',
            'font.family': 'serif',
            'font.serif': 'Ubuntu',
            'font.monospace': 'Ubuntu Mono',
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.labelweight': 'bold',
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 10,
            'figure.titlesize': 12
        }
        sns.set_context(rc)
        sns.set_style("whitegrid")
        sns.set_palette("deep", desat=.6)

        if self.rolling_sharpe:
            offset_index = 1
        else:
            offset_index = 0
        vertical_sections = 5 + offset_index
        fig = plt.figure(figsize=(10, vertical_sections * 3.5))
        fig.suptitle(self.title, y=0.94, weight='bold')
        gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.25, hspace=0.5)

        stats = self.get_results()
        ax_equity = plt.subplot(gs[:2, :])
        if self.rolling_sharpe:
            ax_sharpe = plt.subplot(gs[2, :])
        ax_drawdown = plt.subplot(gs[2 + offset_index, :])
        ax_monthly_returns = plt.subplot(gs[3 + offset_index, :2])
        ax_yearly_returns = plt.subplot(gs[3 + offset_index, 2])
        ax_txt_curve = plt.subplot(gs[4 + offset_index, 0])
        ax_txt_trade = plt.subplot(gs[4 + offset_index, 1])
        ax_txt_time = plt.subplot(gs[4 + offset_index, 2])

        self._plot_equity(stats, ax=ax_equity)
        if self.rolling_sharpe:
            self._plot_rolling_sharpe(stats, ax=ax_sharpe)
        self._plot_drawdown(stats, ax=ax_drawdown)
        self._plot_monthly_returns(stats, ax=ax_monthly_returns)
        self._plot_yearly_returns(stats, ax=ax_yearly_returns)
        self._plot_txt_curve(stats, ax=ax_txt_curve)
        self._plot_txt_trade(stats, ax=ax_txt_trade)
        self._plot_txt_time(stats, ax=ax_txt_time)

        # Plot the figure
        plt.show(block=False)

        if filename is not None:
            fig.savefig(filename, dpi=150, bbox_inches='tight')