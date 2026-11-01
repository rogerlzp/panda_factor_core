"""
panda_factor_core 的契约测试。

测三件事：
  ① 算子真的可用（不是空壳）
  ② RANK 是**横截面**算子（按 date 分组，归一到 [-0.5, 0.5]）—— 这是最容易搞错的地方
  ③ 策略与指标接口自洽

⚠️ 本仓库的索引约定是 **(date, symbol)**，不是 (symbol, date)。
   算子实现会把索引名强制改成 ['date','symbol']。

跑法： pip install pytest && pytest tests/ -v
      或  python tests/test_core.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate.factor_base import Factor  # noqa: E402
from operators import DELAY, MA, RANK, STDDEV  # noqa: E402


def _panel(n_dates=6, n_syms=3, seed=0):
    """构造 (date, symbol) 多级索引的行情面板 —— 与仓库约定一致。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-05", periods=n_dates, freq="B")
    symbols = [f"{i:06d}.SZ" for i in range(n_syms)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    return pd.DataFrame({"close": rng.normal(100, 5, len(idx))}, index=idx)


# ─────────────── 算子 ───────────────

def test_operators_exist_and_are_callable():
    for fn in (DELAY, MA, RANK, STDDEV):
        assert callable(fn)


def test_delay_shifts_within_symbol():
    """DELAY 按 symbol 分组位移 —— 必须有 (date, symbol) 索引。"""
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05", "2026-01-06"]), ["A"]],
        names=["date", "symbol"])
    s = pd.Series([1.0, 2.0], index=idx)
    out = DELAY(s, 1)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 1.0


def test_ma_is_rolling_mean():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = MA(s, 3)
    # 前 N-1 个为 NaN，第 3 个起为窗口均值
    assert out.iloc[1] is np.nan or np.isnan(out.iloc[1])
    assert np.isclose(out.iloc[2], 2.0)      # (1+2+3)/3
    assert np.isclose(out.iloc[4], 4.0)      # (3+4+5)/3


def test_stddev_within_symbol():
    """STDDEV 是按 symbol 的滚动标准差，需 (date, symbol) 索引。"""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=10, freq="B"), ["A"]],
        names=["date", "symbol"])
    s = pd.Series(np.arange(10, dtype=float), index=idx)
    out = STDDEV(s, 5)
    assert out.notna().sum() > 0
    assert out.iloc[-1] >= 0


# ─────────────── RANK：横截面语义（最关键） ───────────────

def test_rank_is_cross_sectional_per_date():
    """RANK 必须按 date 分组，不能跨日期归一化。

    构造两个日期，每个日期 3 只标的，数值模式刻意相同。
    正确实现下，两个日期的排名结构必须完全一致；
    若实现是时间序列 rank，第二个日期会得到不同结果。
    """
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    symbols = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    s = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], index=idx)

    out = RANK(s)

    d1 = out.xs(pd.Timestamp("2026-01-05"), level="date")
    d2 = out.xs(pd.Timestamp("2026-01-06"), level="date")

    assert list(d1.values) == list(d2.values), "RANK 不是按 date 分组的横截面归一化"
    # 归一到 [-0.5, 0.5]
    assert np.isclose(d1.min(), -0.5)
    assert np.isclose(d1.max(), 0.5)


def test_rank_ordering_preserved():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05"]), ["A", "B", "C"]], names=["date", "symbol"])
    s = pd.Series([5.0, 1.0, 3.0], index=idx)
    out = RANK(s)
    assert out.loc[(pd.Timestamp("2026-01-05"), "A")] > \
           out.loc[(pd.Timestamp("2026-01-05"), "C")] > \
           out.loc[(pd.Timestamp("2026-01-05"), "B")]


def test_rank_handles_nan():
    """NaN 应被填 0（实现里 result.fillna(0)），不能崩。"""
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05"]), ["A", "B"]], names=["date", "symbol"])
    s = pd.Series([np.nan, 2.0], index=idx)
    out = RANK(s)
    assert len(out) == 2


# ─────────────── Factor ABC ───────────────

def test_factor_subclass_gets_operators_as_methods():
    class Simple(Factor):
        def calculate(self, factors):
            close = factors["close"]
            return self.RANK(close / self.DELAY(close, 1) - 1)

    f = Simple()
    panel = _panel()
    res = f.calculate({"close": panel["close"]})

    assert isinstance(res, pd.Series)
    assert len(res) == len(panel)
    assert res.notna().sum() > 0


def test_factor_requires_calculate():
    try:
        Factor()  # type: ignore[abstract]
    except TypeError:
        return
    raise AssertionError("Factor 应当是抽象类，不能直接实例化")


# ─────────────── 策略与指标 ───────────────

def test_multifactor_strategy_interface():
    from backtest.strategy import MultiFactorStrategy

    s = MultiFactorStrategy(
        factors=["momentum", "value"],
        weights={"momentum": 0.6, "value": 0.4},
        top_n=2,
    )
    for attr in ("select_stocks", "should_sell", "get_position_ratio"):
        assert hasattr(s, attr), f"缺少方法 {attr}"
    assert s.top_n == 2


def test_metrics_keys_present():
    from backtest.metrics import calculate_all_metrics

    # 注意：metrics 期望 date / total_value 是**列**
    pv = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=10, freq="B"),
        "total_value": np.linspace(1_000_000, 1_200_000, 10),
    })
    trades = pd.DataFrame({
        "action": ["buy", "sell"],
        "profit": [0.0, 800.0],
    })
    m = calculate_all_metrics(pv, trades, initial_capital=1_000_000)
    for k in ("initial_capital", "final_value", "total_return", "annual_return",
              "max_drawdown", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
              "win_rate", "profit_loss_ratio", "total_trades"):
        assert k in m, f"指标缺少 {k}"
        assert isinstance(m[k], (int, float, np.floating)), f"{k} 不是数值"


def test_metrics_empty_input_is_safe():
    """空输入必须返回零值字典，不能抛异常 —— 生产里空回测很常见。"""
    from backtest.metrics import calculate_all_metrics

    empty = pd.DataFrame({"date": [], "total_value": []})
    m = calculate_all_metrics(empty, pd.DataFrame({"action": [], "profit": []}), 1_000_000)
    assert m["total_trades"] == 0
    assert m["annual_return"] == 0.0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL  {name}: {e}")
            except Exception as e:
                fails += 1
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print("\nall passed" if not fails else f"\n{fails} failed")
    sys.exit(1 if fails else 0)
