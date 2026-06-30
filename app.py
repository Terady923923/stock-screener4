
import io
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go


st.set_page_config(page_title="株AIスクリーナー V4", layout="wide")

JPX_URLS = [
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls",
    "https://www.jpx.co.jp/english/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_e.xls",
]

FALLBACK_TICKERS = pd.DataFrame(
    [
        {"code": "7203", "name": "トヨタ自動車", "market": "Prime"},
        {"code": "6758", "name": "ソニーグループ", "market": "Prime"},
        {"code": "9432", "name": "日本電信電話", "market": "Prime"},
        {"code": "8306", "name": "三菱UFJフィナンシャル・グループ", "market": "Prime"},
        {"code": "8316", "name": "三井住友フィナンシャルグループ", "market": "Prime"},
        {"code": "8035", "name": "東京エレクトロン", "market": "Prime"},
        {"code": "9984", "name": "ソフトバンクグループ", "market": "Prime"},
        {"code": "6861", "name": "キーエンス", "market": "Prime"},
        {"code": "4063", "name": "信越化学工業", "market": "Prime"},
        {"code": "6098", "name": "リクルートホールディングス", "market": "Prime"},
    ]
)


def normalize_code(x) -> str | None:
    """JPXのコードをYahoo Finance用に壊さず文字列化する。7203.0 -> 7203、130A -> 130A。"""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace(" ", "").replace("　", "")
    if not s:
        return None
    return s


def yahoo_symbol(code: str) -> str:
    return f"{normalize_code(code)}.T"


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_jpx_list() -> tuple[pd.DataFrame, str]:
    errors = []
    for url in JPX_URLS:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            raw = io.BytesIO(r.content)

            # JPXはxls形式のことが多い。xlrdが必要。
            df = pd.read_excel(raw, dtype=str)
            df.columns = [str(c).strip() for c in df.columns]

            def pick_col(candidates):
                for col in df.columns:
                    low = col.lower()
                    for cand in candidates:
                        if cand.lower() in low:
                            return col
                return None

            code_col = pick_col(["コード", "code"])
            name_col = pick_col(["銘柄名", "name", "issue"])
            market_col = pick_col(["市場", "market", "section"])

            if code_col is None:
                raise ValueError(f"コード列を認識できません。columns={list(df.columns)}")

            out = pd.DataFrame()
            out["code"] = df[code_col].map(normalize_code)
            out["name"] = df[name_col].astype(str) if name_col else ""
            out["market"] = df[market_col].astype(str) if market_col else ""
            out = out.dropna(subset=["code"]).drop_duplicates("code")
            out = out[out["code"].str.len() >= 4].reset_index(drop=True)
            return out, f"JPX公式リスト取得成功: {len(out)}銘柄"
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")

    return FALLBACK_TICKERS.copy(), "JPX公式リスト取得失敗。フォールバック銘柄のみ使用。\n" + "\n".join(errors)


def market_match(market: str, selected: list[str]) -> bool:
    if not selected:
        return True
    m = str(market).lower()
    for s in selected:
        ss = s.lower()
        if ss == "prime" and ("prime" in m or "プライム" in m):
            return True
        if ss == "standard" and ("standard" in m or "スタンダード" in m):
            return True
        if ss == "growth" and ("growth" in m or "グロース" in m):
            return True
    return False


@st.cache_data(ttl=30 * 60, show_spinner=False)
def download_prices(symbols: list[str], period: str, interval: str = "1d") -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """yfinanceで複数銘柄をチャンク取得。失敗時は個別取得へフォールバック。"""
    results: dict[str, pd.DataFrame] = {}
    failures: list[dict] = []

    if not symbols:
        return results, failures

    chunk_size = 40

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]

        try:
            data = yf.download(
                tickers=" ".join(chunk),
                period=period,
                interval=interval,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=20,
            )

            if data is None or data.empty:
                raise ValueError("batch download returned empty dataframe")

            for sym in chunk:
                try:
                    if len(chunk) == 1:
                        d = data.copy()
                    else:
                        if sym not in data.columns.get_level_values(0):
                            raise ValueError("symbol missing in batch result")
                        d = data[sym].copy()

                    d = d.dropna(how="all")
                    if "Close" not in d.columns or d["Close"].dropna().empty:
                        raise ValueError("Close is empty")
                    results[sym] = d
                except Exception as e:
                    failures.append({"symbol": sym, "reason": f"batch parse failed: {e}"})

        except Exception as batch_error:
            for sym in chunk:
                try:
                    d = yf.download(
                        tickers=sym,
                        period=period,
                        interval=interval,
                        auto_adjust=False,
                        progress=False,
                        timeout=20,
                    )
                    d = d.dropna(how="all")
                    if d.empty or "Close" not in d.columns or d["Close"].dropna().empty:
                        raise ValueError("empty price data")
                    results[sym] = d
                except Exception as e:
                    failures.append({"symbol": sym, "reason": f"{type(e).__name__}: {e} / batch={batch_error}"})

        time.sleep(0.2)

    return results, failures


def add_indicators(df: pd.DataFrame, short_ma: int, mid_ma: int, long_ma: int, super_long_ma: int) -> pd.DataFrame:
    d = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d[f"MA{short_ma}"] = d["Close"].rolling(short_ma).mean()
    d[f"MA{mid_ma}"] = d["Close"].rolling(mid_ma).mean()
    d[f"MA{long_ma}"] = d["Close"].rolling(long_ma).mean()
    d[f"MA{super_long_ma}"] = d["Close"].rolling(super_long_ma).mean()

    # RSI 14
    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIGNAL"] = d["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger
    d["BB_MID"] = d["Close"].rolling(20).mean()
    d["BB_STD"] = d["Close"].rolling(20).std()
    d["BB_UPPER"] = d["BB_MID"] + 2 * d["BB_STD"]
    d["BB_LOWER"] = d["BB_MID"] - 2 * d["BB_STD"]

    d["VOL_MA20"] = d["Volume"].rolling(20).mean()
    return d


def crossed_within(d: pd.DataFrame, a_col: str, b_col: str, days: int) -> int | None:
    """aがbを上抜けた日が直近days営業日以内なら、何営業日前かを返す。最新日=0。"""
    recent = d[[a_col, b_col]].dropna().tail(days + 1)
    if len(recent) < 2:
        return None
    for i in range(1, len(recent)):
        prev = recent.iloc[i - 1]
        curr = recent.iloc[i]
        if prev[a_col] <= prev[b_col] and curr[a_col] > curr[b_col]:
            return len(recent) - 1 - i
    return None


def macd_gc_within(d: pd.DataFrame, days: int) -> int | None:
    return crossed_within(d, "MACD", "MACD_SIGNAL", days)


def pct_diff(a, b) -> float | None:
    if b is None or pd.isna(b) or b == 0:
        return None
    return (a - b) / b * 100


def check_filters(d: pd.DataFrame, opts: dict) -> tuple[bool, list[str], dict]:
    reasons = []
    meta = {}
    short = opts["short_ma"]
    mid = opts["mid_ma"]
    long = opts["long_ma"]
    super_long = opts["super_long_ma"]

    ma_s = f"MA{short}"
    ma_m = f"MA{mid}"
    ma_l = f"MA{long}"
    ma_sl = f"MA{super_long}"

    d2 = d.dropna(subset=["Close"]).copy()
    if len(d2) < max(short, mid, long, super_long, 30):
        return False, ["必要日数不足"], meta

    latest = d2.iloc[-1]
    prev = d2.iloc[-2] if len(d2) >= 2 else latest

    close = latest["Close"]
    volume = latest.get("Volume", np.nan)

    meta["close"] = close
    meta["volume"] = volume
    meta["ma_short"] = latest.get(ma_s, np.nan)
    meta["ma_mid"] = latest.get(ma_m, np.nan)
    meta["ma_long"] = latest.get(ma_l, np.nan)
    meta["ma_super_long"] = latest.get(ma_sl, np.nan)
    meta["rsi"] = latest.get("RSI14", np.nan)
    meta["macd"] = latest.get("MACD", np.nan)
    meta["macd_signal"] = latest.get("MACD_SIGNAL", np.nan)
    meta["close_vs_short_pct"] = pct_diff(close, meta["ma_short"])

    if opts["min_price"] and close < opts["min_price"]:
        reasons.append("最低株価未満")
    if opts["max_price"] and close > opts["max_price"]:
        reasons.append("最高株価超過")
    if opts["min_volume"] and (pd.isna(volume) or volume < opts["min_volume"]):
        reasons.append("出来高不足")

    if opts["price_position"] == "短期MAより下" and not (close < latest[ma_s]):
        reasons.append("株価が短期MAより下ではない")
    if opts["price_position"] == "短期MAより上" and not (close > latest[ma_s]):
        reasons.append("株価が短期MAより上ではない")
    if opts["price_position"] == "中期MAより上" and not (close > latest[ma_m]):
        reasons.append("株価が中期MAより上ではない")
    if opts["price_position"] == "中期MAより下" and not (close < latest[ma_m]):
        reasons.append("株価が中期MAより下ではない")

    if opts["require_ma_gc"]:
        gc_days = crossed_within(d2, ma_s, ma_m, opts["gc_days"])
        meta["ma_gc_days_ago"] = gc_days
        if gc_days is None:
            reasons.append("短期MAの中期MA上抜けなし")
    else:
        meta["ma_gc_days_ago"] = crossed_within(d2, ma_s, ma_m, opts["gc_days"])

    if opts["require_ma_order"]:
        if not (latest[ma_s] > latest[ma_m] > latest[ma_l]):
            reasons.append("MA順張り配列でない")

    if opts["require_mid_slope_up"]:
        lookback = opts["slope_days"]
        if len(d2.dropna(subset=[ma_m])) <= lookback:
            reasons.append("中期MA傾き判定不可")
        else:
            old = d2.dropna(subset=[ma_m]).iloc[-1 - lookback][ma_m]
            if not latest[ma_m] > old:
                reasons.append("中期MAが上向きでない")

    if opts["require_long_slope_up"]:
        lookback = opts["slope_days"]
        if len(d2.dropna(subset=[ma_l])) <= lookback:
            reasons.append("長期MA傾き判定不可")
        else:
            old = d2.dropna(subset=[ma_l]).iloc[-1 - lookback][ma_l]
            if not latest[ma_l] > old:
                reasons.append("長期MAが上向きでない")

    if opts["use_rsi"]:
        rsi = latest.get("RSI14", np.nan)
        if pd.isna(rsi) or rsi < opts["rsi_min"] or rsi > opts["rsi_max"]:
            reasons.append("RSI範囲外")

    if opts["require_macd_gc"]:
        m_days = macd_gc_within(d2, opts["macd_gc_days"])
        meta["macd_gc_days_ago"] = m_days
        if m_days is None:
            reasons.append("MACD GCなし")
    else:
        meta["macd_gc_days_ago"] = macd_gc_within(d2, opts["macd_gc_days"])

    if opts["require_volume_surge"]:
        vol_ma = latest.get("VOL_MA20", np.nan)
        if pd.isna(vol_ma) or vol_ma == 0 or volume < vol_ma * opts["volume_surge_ratio"]:
            reasons.append("出来高急増なし")
        meta["volume_vs_ma20"] = (volume / vol_ma) if vol_ma and not pd.isna(vol_ma) else np.nan

    if opts["require_bullish"]:
        if not (latest["Close"] > latest["Open"]):
            reasons.append("陽線でない")

    if opts["require_prev_bearish"]:
        if not (prev["Close"] < prev["Open"]):
            reasons.append("前日陰線でない")

    if opts["require_bb_lower_touch"]:
        if not (latest["Low"] <= latest["BB_LOWER"] or latest["Close"] <= latest["BB_LOWER"]):
            reasons.append("BB下限タッチなし")

    return len(reasons) == 0, reasons, meta


def make_chart(df: pd.DataFrame, code: str, name: str, opts: dict):
    short = opts["short_ma"]
    mid = opts["mid_ma"]
    long = opts["long_ma"]
    super_long = opts["super_long_ma"]

    d = df.tail(160).copy()
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=d.index,
            open=d["Open"],
            high=d["High"],
            low=d["Low"],
            close=d["Close"],
            name="株価",
        )
    )
    for n in [short, mid, long, super_long]:
        col = f"MA{n}"
        if col in d.columns:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], mode="lines", name=col))

    fig.update_layout(
        title=f"{code} {name}",
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


st.title("📈 株AIスクリーナー V4")
st.caption("JPX銘柄リスト + Yahoo Finance から株価取得。取得成功数と失敗理由を必ず表示します。")

with st.sidebar:
    st.header("検索対象")
    jpx_df, jpx_msg = load_jpx_list()
    st.info(jpx_msg)

    markets = st.multiselect("市場", ["Prime", "Standard", "Growth"], default=["Prime"])

    max_symbols = st.slider("最大検索銘柄数", 10, 4000, 300, step=10)
    period = st.selectbox("株価取得期間", ["6mo", "1y", "2y", "5y"], index=1)

    manual_codes = st.text_area(
        "テスト用：個別コード指定（任意）",
        value="",
        placeholder="例：7203, 8035, 9984\nここに入力すると、この銘柄だけ検索します",
    )

    st.header("移動平均")
    short_ma = st.number_input("短期MA", min_value=2, max_value=50, value=5)
    mid_ma = st.number_input("中期MA", min_value=5, max_value=100, value=25)
    long_ma = st.number_input("長期MA", min_value=20, max_value=200, value=75)
    super_long_ma = st.number_input("超長期MA", min_value=50, max_value=300, value=200)

    require_ma_gc = st.checkbox("短期MAが中期MAを直近で上抜け", value=True)
    gc_days = st.slider("GC判定日数", 1, 30, 5)

    price_position = st.selectbox(
        "現在株価の位置",
        ["指定なし", "短期MAより下", "短期MAより上", "中期MAより上", "中期MAより下"],
        index=1,
    )

    require_ma_order = st.checkbox("MA順張り配列（短期 > 中期 > 長期）", value=False)
    require_mid_slope_up = st.checkbox("中期MAが上向き", value=False)
    require_long_slope_up = st.checkbox("長期MAが上向き", value=False)
    slope_days = st.slider("傾き判定日数", 1, 20, 5)

    st.header("価格・出来高")
    min_price = st.number_input("最低株価", min_value=0, value=0)
    max_price = st.number_input("最高株価（0なら無制限）", min_value=0, value=0)
    min_volume = st.number_input("最低出来高", min_value=0, value=100000, step=10000)

    require_volume_surge = st.checkbox("出来高急増", value=False)
    volume_surge_ratio = st.slider("出来高急増倍率（20日平均比）", 1.0, 5.0, 1.5, step=0.1)

    st.header("RSI / MACD / ローソク")
    use_rsi = st.checkbox("RSIで絞る", value=False)
    rsi_min, rsi_max = st.slider("RSI範囲", 0, 100, (30, 70))

    require_macd_gc = st.checkbox("MACDが直近でGC", value=False)
    macd_gc_days = st.slider("MACD GC判定日数", 1, 30, 5)

    require_bullish = st.checkbox("最新日が陽線", value=False)
    require_prev_bearish = st.checkbox("前日が陰線", value=False)
    require_bb_lower_touch = st.checkbox("ボリンジャーバンド下限タッチ", value=False)

    debug = st.checkbox("デバッグ表示", value=True)

    run = st.button("🔍 スクリーニング実行", type="primary")
    if run:
        st.session_state["has_run"] = True


opts = {
    "short_ma": int(short_ma),
    "mid_ma": int(mid_ma),
    "long_ma": int(long_ma),
    "super_long_ma": int(super_long_ma),
    "require_ma_gc": require_ma_gc,
    "gc_days": int(gc_days),
    "price_position": price_position,
    "require_ma_order": require_ma_order,
    "require_mid_slope_up": require_mid_slope_up,
    "require_long_slope_up": require_long_slope_up,
    "slope_days": int(slope_days),
    "min_price": float(min_price),
    "max_price": float(max_price) if max_price else 0,
    "min_volume": int(min_volume),
    "require_volume_surge": require_volume_surge,
    "volume_surge_ratio": float(volume_surge_ratio),
    "use_rsi": use_rsi,
    "rsi_min": int(rsi_min),
    "rsi_max": int(rsi_max),
    "require_macd_gc": require_macd_gc,
    "macd_gc_days": int(macd_gc_days),
    "require_bullish": require_bullish,
    "require_prev_bearish": require_prev_bearish,
    "require_bb_lower_touch": require_bb_lower_touch,
}


if "has_run" not in st.session_state:
    st.session_state["has_run"] = False
if not st.session_state["has_run"]:
    st.write("左の条件を設定して **スクリーニング実行** を押してください。")
    st.write("まず動作確認するなら、左の「テスト用：個別コード指定」に `7203,8035,9984` を入れて実行してください。")
    st.stop()
    


# 対象銘柄作成
if manual_codes.strip():
    codes = []
    for part in manual_codes.replace("\n", ",").split(","):
        c = normalize_code(part)
        if c:
            codes.append(c)
    target_df = pd.DataFrame({"code": codes, "name": "", "market": "manual"}).drop_duplicates("code")
else:
    target_df = jpx_df[jpx_df["market"].apply(lambda x: market_match(x, markets))].copy()
    target_df = target_df.head(max_symbols)

target_df["symbol"] = target_df["code"].apply(yahoo_symbol)
symbols = target_df["symbol"].tolist()

st.subheader("取得状況")
status_box = st.empty()

with st.spinner("株価データ取得中..."):
    prices, failures = download_prices(symbols, period=period)

status_box.success(f"対象 {len(symbols)}銘柄 / 株価取得成功 {len(prices)}銘柄 / 取得失敗 {len(failures)}銘柄")

if len(prices) == 0:
    st.error("株価取得成功が0件です。まずテスト用に 7203,8035,9984 を入力して実行してください。")
    if failures:
        st.write("取得失敗例")
        st.dataframe(pd.DataFrame(failures).head(50), use_container_width=True)
    st.stop()

rows = []
exclude_reasons = []
price_store = {}

progress = st.progress(0)
for idx, row in target_df.iterrows():
    sym = row["symbol"]
    if sym not in prices:
        continue

    try:
        d = add_indicators(prices[sym], opts["short_ma"], opts["mid_ma"], opts["long_ma"], opts["super_long_ma"])
        ok, reasons, meta = check_filters(d, opts)
        price_store[row["code"]] = d

        if ok:
            rows.append(
                {
                    "code": row["code"],
                    "name": row.get("name", ""),
                    "market": row.get("market", ""),
                    "close": round(meta.get("close", np.nan), 2),
                    f"MA{opts['short_ma']}": round(meta.get("ma_short", np.nan), 2),
                    f"MA{opts['mid_ma']}": round(meta.get("ma_mid", np.nan), 2),
                    f"MA{opts['long_ma']}": round(meta.get("ma_long", np.nan), 2),
                    "GC何営業日前": meta.get("ma_gc_days_ago"),
                    "短期MA乖離率%": round(meta.get("close_vs_short_pct", np.nan), 2),
                    "出来高": int(meta.get("volume", 0)) if not pd.isna(meta.get("volume", np.nan)) else None,
                    "RSI14": round(meta.get("rsi", np.nan), 1),
                    "MACD_GC何営業日前": meta.get("macd_gc_days_ago"),
                }
            )
        else:
            for r in reasons:
                exclude_reasons.append({"code": row["code"], "name": row.get("name", ""), "reason": r})
    except Exception as e:
        exclude_reasons.append({"code": row["code"], "name": row.get("name", ""), "reason": f"判定エラー: {type(e).__name__}: {e}"})

    if len(target_df) > 0:
        progress.progress(min(1.0, (idx + 1) / len(target_df)))

progress.empty()

result_df = pd.DataFrame(rows)

col1, col2, col3, col4 = st.columns(4)
col1.metric("対象銘柄", len(symbols))
col2.metric("株価取得成功", len(prices))
col3.metric("該当銘柄", len(result_df))
col4.metric("除外理由数", len(exclude_reasons))

if debug:
    with st.expander("デバッグ：株価取得失敗"):
        st.dataframe(pd.DataFrame(failures).head(200), use_container_width=True)

    with st.expander("デバッグ：除外理由サマリー", expanded=True):
        if exclude_reasons:
            ex = pd.DataFrame(exclude_reasons)
            summary_df = ex["reason"].value_counts().rename_axis("reason").reset_index(name="count")
            st.dataframe(summary_df, use_container_width=True)
            st.write("除外理由の例")
            st.dataframe(ex.head(200), use_container_width=True)
        else:
            st.write("除外理由はありません。")

st.subheader("スクリーニング結果")

if result_df.empty:
    st.warning("該当銘柄はありません。条件をゆるめてください。まずは「短期MA GC」をOFF、「株価位置=指定なし」、「最低出来高=0」で確認してください。")
else:
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSVダウンロード", csv, "screening_result.csv", "text/csv")

    selected = st.selectbox(
        "チャート表示",
        result_df["code"].tolist(),
        format_func=lambda c: f"{c} {result_df.loc[result_df['code'] == c, 'name'].iloc[0]}",
    )

    if selected in price_store:
        name = result_df.loc[result_df["code"] == selected, "name"].iloc[0]
        st.plotly_chart(make_chart(price_store[selected], selected, name, opts), use_container_width=True)
