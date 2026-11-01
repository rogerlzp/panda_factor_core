# panda_factor_core

The factor engine from [PandaFactor](https://github.com/rogerlzp/panda_factor) — extracted as a **standalone, dependency-light library**.

The full platform is a large multi-package application (web UI, MongoDB data hub, LLM service, deployment). This repo is the part that is actually useful on its own: **operators, the `Factor` base class, strategy base classes, and backtest metrics** — with nothing but `numpy` and `pandas`.

```
panda_factor_core/
├── operators.py          # module-level operator functions
├── generate/
│   ├── factor_base.py    # Factor ABC — base class for user-defined factors
│   └── factor_utils.py   # FactorUtils — 103 operator implementations
└── backtest/
    ├── base.py           # backtest engine core
    ├── strategy.py       # BaseStrategy / MultiFactorStrategy / V67Strategy
    └── metrics.py        # performance metrics
```

---

## Why this exists

Quant research code drifts. Operator logic gets copy-pasted across factor files, look-ahead bias creeps in, and two strategies end up reporting Sharpe ratios that were computed differently — so the numbers aren't comparable.

Three layers fix that:

| Layer | What it guarantees |
|---|---|
| **Operators** | One canonical implementation per operator. A fix lands everywhere at once, not just in the file you happened to be editing. |
| **Factor base class** | Implement `calculate(factors)`, inherit all 103 operators as instance methods. New factors are ~10 lines. |
| **Strategy + metrics** | One strategy interface and one metrics implementation, so results are comparable across strategies by construction. |

---

## Install

```bash
pip install -r requirements.txt
```

Or as a package:

```bash
pip install -e .
```

Python 3.9+ (developed on 3.12). No database, no network, no broker dependency.

---

## Quick start

### Write a factor

```python
from generate.factor_base import Factor

class MomentumFactor(Factor):
    def calculate(self, factors):
        close = factors['close']
        mom = close / self.DELAY(close, 20) - 1
        return self.RANK(mom)
```

`calculate()` returns a `pd.Series` indexed by `['symbol', 'date']`.

`RANK` normalises each date's cross-section to **[-0.5, 0.5]** — it groups by `date`, not by time. That's the distinction that keeps look-ahead bias out of cross-sectional factors, and it lives in the base class so nobody re-implements it wrong.

### Use operators directly

```python
from operators import *

ma20  = MA(close, 20)
vol20 = STDDEV(close, 20)
mom   = close / DELAY(close, 20) - 1
```

### Backtest and score

```python
from backtest.strategy import MultiFactorStrategy
from backtest.metrics import calculate_all_metrics

strategy = MultiFactorStrategy(
    name='demo',
    factor_cols=['momentum', 'value', 'quality'],
    weights=[0.4, 0.3, 0.3],
    top_n=30,
)
positions = strategy.select_stocks(df_day, '2026-01-15')

metrics = calculate_all_metrics(pv_df, trades_df, initial_capital=1_000_000)
print(metrics)
```

---

## Operator catalogue

103 operators, implemented in `generate/factor_utils.py` and re-exported at module level by `operators.py`.

| Family | Examples |
|---|---|
| **Time series** | `DELAY`, `DELTA`, `DIFF`, `REF`, `SUM`, `PRODUCT`, `TS_MAX`, `TS_MIN`, `TS_ARGMAX`, `TS_ARGMIN`, `TS_RANK` |
| **Moving average** | `MA`, `EMA`, `SMA`, `WMA`, `DMA`, `DECAY_LINEAR` |
| **Statistical** | `STD`, `STDDEV`, `CORRELATION`, `COVARIANCE`, `AVEDEV`, `SLOPE` |
| **Extremes** | `MAX`, `MIN`, `HHV`, `LLV`, `HHVBARS`, `LLVBARS` |
| **Cross-sectional** | `RANK` (per-date normalisation to [-0.5, 0.5]) |
| **Logical / conditional** | `IF`, `RETURNS` |

---

## Design decisions worth knowing

**Operators are exposed twice, on purpose.** `FactorUtils` holds them as `@staticmethod`; `operators.py` re-exports them as module-level functions; `Factor.__init__` binds them onto instances. One implementation, three access patterns — so a bug fix or an optimisation lands everywhere.

**`RANK` is cross-sectional, not time-series.** It groups by `date` and normalises each date's cross-section to [-0.5, 0.5]. This is the single most common place quant research introduces look-ahead bias, so it's centralised in the base class.

**The strategy layer separates selection from execution.** `select_stocks` decides what to hold; `should_sell` decides when to exit; `get_position_ratio` decides sizing. Keeping them apart is what lets `MultiFactorStrategy` and `V67Strategy` share one backtest engine and one metrics implementation.

**Metrics live in one place.** `calculate_all_metrics` is the only implementation of annual return, max drawdown, Sharpe, Sortino, Calmar, win rate, and profit/loss ratio — so two strategies' numbers mean the same thing.

---

## Metrics

`calculate_all_metrics(pv_df, trades_df, initial_capital)` returns:

| Metric | Definition |
|---|---|
| `annual_return` | CAGR of the portfolio value curve |
| `max_drawdown` | Largest peak-to-trough decline |
| `sharpe_ratio` | Annualised excess return / volatility (default `risk_free_rate=0.03`) |
| `sortino_ratio` | Like Sharpe, but only downside deviation |
| `calmar_ratio` | Annual return / max drawdown |
| `win_rate` | Share of profitable closed trades |
| `profit_loss_ratio` | Average win / average loss |

---

## Related

- **[`panda_factor`](https://github.com/rogerlzp/panda_factor)** — the full platform this was extracted from: web UI, data hub, LLM service, factor analysis, ~5 years of built-in market data, nightly ingestion from Tushare / RiceQuant / Xuntou / Tqsdk.
- **[`dbt_quant_demo`](https://github.com/rogerlzp/dbt_quant_demo)** — the same factor panel rebuilt as a dbt project (staging → intermediate → mart), with semantic metrics and data-quality gates as first-class models.
- **[`mcp_quant_server`](https://github.com/rogerlzp/mcp_quant_server)** — a read-only MCP server that lets an LLM agent query the mart this pipeline builds.

Together: **compute (pandas) → transform (dbt) → serve to agents (MCP)**.

---

## License

MIT — see [LICENSE](LICENSE).
