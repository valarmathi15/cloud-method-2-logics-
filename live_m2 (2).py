"""
live_m2.py  -  LIVE PAPER TRADING : Method 2 (Cloud Scale-In, Google-sheet logic)   [single file]
=====================================================================================
BUILDER   : pick folder, exchange, symbols, candle timeframe, capital, qty, Max lots (many) and
            Target % (many)  ->  press Start  ->  ONE FILE is saved for every symbol x lots x target
            (e.g. folder "Nifty test":  Nifty 50 4H 10lots 1%.json,  ... 10lots 2%.json, ...)
MY STRATEGIES : open a folder -> summary table of all its files, all running live side by side;
            or open any file(s) to see that strategy's live position, chart, bar log and trades.

How "live paper" works: Method 2 is fully determined by the candles from the strategy START time,
so every time you open / refresh, the state is rebuilt from Angel One candles (closed candles only).
No background process is needed for the numbers. LIVE PRICE (LTP) comes from the Angel One WebSocket feed and
updates every ~1.5 s on the strategy page; BUY / EXIT decisions still use CLOSED candles only.

Run : python -m streamlit run live_m2.py
Login details (API key, Client ID, MPIN, TOTP secret) are typed on the first screen.
Strategy files are saved in the  strategies/<ClientID>/<folder>/  folder next to this file.
NOTE: on Streamlit Cloud that folder is wiped when the app reboots - run it on your own PC to keep files.
"""
import datetime as dt
import json
import os
import re
import shutil
import threading
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
STRAT_ROOT = os.path.join(HERE, 'strategies')
STATE_FILE = os.path.join(HERE, 'last_signal.json')

# timeframe -> (Angel interval, raw bar length, warm-up calendar days needed for the cloud)
def now_ist():
    """current India time as a naive Timestamp - the server (e.g. Streamlit Cloud) may run in UTC, Angel candles are IST."""
    return pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None)


def ts_ist(epoch):
    return pd.Timestamp(epoch, unit='s', tz='UTC').tz_convert('Asia/Kolkata').tz_localize(None)


TF = {
    '1D':  ('ONE_DAY',        pd.Timedelta(days=1),      260),
    '4H':  ('ONE_HOUR',       pd.Timedelta(hours=1),     160),   # 4H = 4 hourly bars, built per day
    '1H':  ('ONE_HOUR',       pd.Timedelta(hours=1),     100),
    '30m': ('THIRTY_MINUTE',  pd.Timedelta(minutes=30),  25),
    '15m': ('FIFTEEN_MINUTE', pd.Timedelta(minutes=15),  14),
    '10m': ('TEN_MINUTE',     pd.Timedelta(minutes=10),  10),
    '5m':  ('FIVE_MINUTE',    pd.Timedelta(minutes=5),   7),
    '3m':  ('THREE_MINUTE',   pd.Timedelta(minutes=3),   5),
    '1m':  ('ONE_MINUTE',     pd.Timedelta(minutes=1),   5),
}
# max days Angel returns per getCandleData request (kept a bit under the real limit)
INTERVAL_MAX_DAYS = {'ONE_DAY': 1900, 'ONE_HOUR': 380, 'THIRTY_MINUTE': 190, 'FIFTEEN_MINUTE': 190,
                     'TEN_MINUTE': 95, 'FIVE_MINUTE': 95, 'THREE_MINUTE': 55, 'ONE_MINUTE': 28}
LOTS_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TARGET_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SPAN_ONLY = {'SPAN A': 'A', 'SPAN B': 'B', 'SPAN A & B': 'A&B', 'SPAN A OR B': 'AorB'}
CLOUD_EXIT = {'Above cloud': 'CloudAbove', 'Inside cloud': 'CloudInside', 'Below cloud': 'CloudBelow'}
SPAN_OPTIONS = {**SPAN_ONLY, **CLOUD_EXIT}                      # every non-% exit (stored in the file as `span`)
EXIT_OPTIONS = TARGET_OPTIONS + list(SPAN_OPTIONS)               # profit target % + span exits + cloud exits
ENTRY_OPTIONS = {'Above cloud': 'above', 'Inside cloud': 'inside', 'Below cloud': 'below'}
AMOUNT_OPTIONS = [1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000]
QTY_OPTIONS = [1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 500, 1000]
CANDLE_OPTIONS = {'Every candle': 'every', 'Daily': 'daily', 'Weekly': 'weekly'}


def exit_label(x):
    return f'{x}%' if isinstance(x, (int, float)) else x


def size_text(s):
    if s.get('amount'):
        return f"{'equal-weight ' if s.get('sizing') == 'equal' else ''}amount/lot {s['amount']:,.0f}"
    return f"qty/lot {s['qty']:g}"


def entry_text(s):
    return {'below': 'below cloud', 'inside': 'inside cloud'}.get(s.get('entry'), 'above cloud')


def exit_text(s):
    return s.get('exit_label') or f"{s['profit']}%"


def candle_text(s):
    return {v: k for k, v in CANDLE_OPTIONS.items()}.get(s.get('candle', 'every'), 'Every candle')


# =============================================================== ENGINE (M2 only, copied from m.py - same logic)
TENKAN_P, KIJUN_P, SENKOU_B_P, DISPLACEMENT = 9, 26, 52, 26


def add_ichimoku_cloud(df):
    df = df.copy()
    high, low = df['High'], df['Low']
    tenkan = (high.rolling(TENKAN_P).max() + low.rolling(TENKAN_P).min()) / 2
    kijun = (high.rolling(KIJUN_P).max() + low.rolling(KIJUN_P).min()) / 2
    df['SenkouA'] = ((tenkan + kijun) / 2).shift(DISPLACEMENT)
    df['SenkouB'] = ((high.rolling(SENKOU_B_P).max() + low.rolling(SENKOU_B_P).min()) / 2).shift(DISPLACEMENT)
    df['CloudTop'] = df[['SenkouA', 'SenkouB']].max(axis=1, skipna=False)
    df['CloudBottom'] = df[['SenkouA', 'SenkouB']].min(axis=1, skipna=False)
    df['AboveCloud'] = df['Close'] > df['CloudTop']
    return df


SPAN_RULES = {                       # exit rule for the "Span" options (price = bar Close)
    'A':    lambda p, a, b: p < a,
    'B':    lambda p, a, b: p < b,
    'A&B':  lambda p, a, b: p < a and p < b,      # below BOTH spans  (= below the cloud bottom)
    'AorB': lambda p, a, b: p < a or p < b,       # below EITHER span (= below the cloud top)
    'CloudAbove': lambda p, a, b: p > max(a, b),  # closed ABOVE the cloud top
    'CloudBelow': lambda p, a, b: p < min(a, b),  # closed BELOW the cloud bottom
    'CloudInside': lambda p, a, b: min(a, b) <= p <= max(a, b),   # closed INSIDE the cloud
}
SPAN_TEXT = {'A': 'Span A', 'B': 'Span B', 'A&B': 'Span A & B', 'AorB': 'Span A or B',
             'CloudAbove': 'the cloud top', 'CloudBelow': 'the cloud bottom', 'CloudInside': 'the cloud'}


def run_m2(df, profit_target_pct, max_lots, start_date=None, lot_qty=1.0, trace=None, span=None, candle='every', lot_amount=None, skip=None, entry='above'):
    """Method 2 (Google-sheet logic).
    span   : None = exit on Profit target %;  'A' / 'B' / 'A&B' / 'AorB' = exit when Close falls below that span
             (whole position closed; span exits go flat, no repeat exits).
    lot_amount : None = every lot buys `lot_qty` units;  number = every lot invests this AMOUNT (units = amount / lot price).
    entry  : 'above' = buy while Close is ABOVE the cloud top;  'below' = BELOW the cloud bottom;  'inside' = INSIDE the cloud.
    skip   : list of (from, to) timestamps - bars inside are ignored (strategy was paused; no buy / exit then).
    candle : 'every' = buy 1 lot on every candle above the cloud;  'daily' = at most 1 lot per day;
             'weekly' = at most 1 lot per week (first candle of the period that closes above the cloud).
    Returns trades DataFrame; trace (list) gets one dict per bar."""
    full = add_ichimoku_cloud(df)
    if start_date is not None:   # cloud uses full history, simulation starts flat from start_date
        full = full[full['Date'] >= pd.Timestamp(start_date)].reset_index(drop=True)
    rows = full.to_dict('records')
    span_hit = SPAN_RULES.get(span)

    def pkey(d):
        if candle == 'daily':
            return d.date()
        if candle == 'weekly':
            y, w, _ = d.isocalendar()
            return (y, w)
        return None

    exit_reason = (({'CloudAbove': 'Closed above the cloud (Method 2)', 'CloudInside': 'Closed inside the cloud (Method 2)'}.get(span) or f'Closed below {SPAN_TEXT[span]} (Method 2)') if span_hit
                   else f'Profit target {profit_target_pct}% hit (Method 2 - sheet logic)')
    trades = []
    trade_id = 0
    m2_first, m2_lots, m2_avg, m2_prev_exit, m2_seg_start = True, 0, None, False, None
    m2_qty, m2_cost = 0.0, 0.0                     # units held and money spent (avg price = cost / units)
    last_buy_key = None

    for row in rows:
        if pd.isna(row['CloudTop']) or pd.isna(row['CloudBottom']):
            continue
        price, date = row['Close'], row['Date']
        if skip and any(a < date < b for a, b in skip):
            continue
        if entry == 'above':                                                                       # entry condition met?
            above = bool(row['AboveCloud'])
        elif entry == 'below':
            above = bool(price < row['CloudBottom'])
        else:                                                                                      # inside the cloud
            above = bool(row['CloudBottom'] <= price <= row['CloudTop'])
        bought, prev_lots = False, 0
        if m2_first:
            m2_first = False
            m2_lots = 1 if above else 0
            bought = m2_lots == 1
            if m2_lots == 1:
                last_buy_key = pkey(date)
        else:
            prev_lots, bought = m2_lots, False
            can_buy = candle == 'every' or pkey(date) != last_buy_key
            if not above or not can_buy:
                pass
            elif m2_prev_exit:
                m2_lots, bought = 1, True
            elif m2_lots < max_lots:
                m2_lots, bought = m2_lots + 1, True
            if bought:
                last_buy_key = pkey(date)
        if bought:                                         # avg price changes ONLY when a lot is bought
            q = (lot_amount / price) if lot_amount else lot_qty
            if prev_lots == 0:
                m2_qty, m2_cost = q, q * price
            else:
                m2_qty, m2_cost = m2_qty + q, m2_cost + q * price
            m2_avg = m2_cost / m2_qty
        m2_prev_exit = False
        if m2_seg_start is None and m2_lots > 0:
            m2_seg_start = date
        if m2_lots > 0 and m2_avg:
            if span_hit:
                hit = span_hit(price, row['SenkouA'], row['SenkouB'])
            else:
                hit = (price - m2_avg) / m2_avg >= profit_target_pct / 100.0
            if hit:
                trade_id += 1
                trades.append({
                    'Trade ID': trade_id, 'Entry Date (1st lot)': m2_seg_start, 'Exit Date': date,
                    'Holding Days': (date - m2_seg_start).days, 'Lots Filled': m2_lots,
                    'Max Lots Allowed': max_lots, 'Avg Entry Price': round(m2_avg, 4),
                    'Exit Price': round(price, 4),
                    'Profit % (on avg entry)': round((price - m2_avg) / m2_avg * 100, 2),
                    'PnL (pts x qty)': round((price - m2_avg) * m2_qty, 2),
                    'Exit Reason': exit_reason})
                m2_prev_exit = True
                m2_seg_start = None
                m2_lots, m2_avg, m2_qty, m2_cost = 0, None, 0.0, 0.0   # exit = flat; fresh trade on the next close above cloud
        if trace is not None:
            trace.append({'Date': date, 'Close': price, 'CloudTop': row['CloudTop'],
                          'CloudBottom': row['CloudBottom'], 'Above': above, 'Lots': m2_lots, 'Bought': bought, 'Qty': m2_qty,
                          'AvgPrice': None if not m2_avg else round(m2_avg, 4),
                          'Action': 'EXIT' if m2_prev_exit else ''})

    if m2_lots > 0 and m2_avg and not m2_prev_exit and m2_seg_start is not None and rows:
        price, date = full.iloc[-1]['Close'], full.iloc[-1]['Date']
        trade_id += 1
        trades.append({
            'Trade ID': trade_id, 'Entry Date (1st lot)': m2_seg_start, 'Exit Date': date,
            'Holding Days': (date - m2_seg_start).days, 'Lots Filled': m2_lots,
            'Max Lots Allowed': max_lots, 'Avg Entry Price': round(m2_avg, 4),
            'Exit Price': round(price, 4),
            'Profit % (on avg entry)': round((price - m2_avg) / m2_avg * 100, 2),
            'PnL (pts x qty)': round((price - m2_avg) * m2_qty, 2),
            'Exit Reason': 'Open at end of data (marked to last close)'})
    return pd.DataFrame(trades)


# =============================================================== DATA FEED (Angel One SmartAPI)
SCRIP_URL = 'https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json'


def angel_login(client_id, api_key, mpin, totp_secret):
    """Login with Client ID + MPIN + TOTP (the 6-digit code is generated from your TOTP secret)."""
    import pyotp
    from SmartApi import SmartConnect
    smart = SmartConnect(api_key=api_key)
    res = smart.generateSession(client_id, mpin, pyotp.TOTP(totp_secret.replace(' ', '')).now())
    if not res or not res.get('status'):
        raise ValueError((res or {}).get('message', 'login failed'))
    smart._login_data = res.get('data') or {}
    return smart


@st.cache_resource(ttl=86400, show_spinner='Loading Angel One instrument list ...')
def _scrip_master():
    return requests.get(SCRIP_URL, timeout=120).json()


@st.cache_resource(ttl=86400, show_spinner=False)
def _token_index():
    idx = {}
    for r in _scrip_master():
        idx.setdefault((r['exch_seg'], r['symbol'].upper()), r['token'])
    return idx


def _instrument_token(exchange, symbol):
    sym = symbol.strip().upper()
    idx = _token_index()
    for key in (sym, sym + '-EQ'):                                # exact match first
        if (exchange, key) in idx:
            return idx[(exchange, key)]
    raise ValueError(f'{symbol} not found on {exchange}  (equity: SBIN-EQ / index: Nifty 50 / MCX: GOLDM05DEC25FUT)')


def _instrument_token_old(exchange, symbol):
    sym = symbol.strip().upper()
    hits = [r for r in _scrip_master() if r['exch_seg'] == exchange
            and r['symbol'].upper() in (sym, sym + '-EQ')]
    if not hits:
        raise ValueError(f'{symbol} not found on {exchange}  (equity: SBIN-EQ / index: Nifty 50 / MCX: GOLDM05DEC25FUT)')
    hits.sort(key=lambda r: r['symbol'].upper() != sym)          # exact match first
    return hits[0]['token']


# =============================================================== LIVE PRICE (Angel One WebSocket)
EXCH_TYPE = {'NSE': 1, 'NFO': 2, 'BSE': 3, 'BFO': 4, 'MCX': 5}


class TickFeed:
    """One WebSocket connection per login, running in a background thread of the Streamlit server.
    Keeps the latest quote (ltp / day open-high-low / previous close) and a short tick history per instrument."""

    def __init__(self, creds):
        self.creds, self.lock = creds, threading.Lock()
        self.ticks, self.hist, self.wanted = {}, {}, set()
        self.sws, self.thread, self.connected = None, None, False
        self.error, self.last_msg = None, 0.0

    # -- public
    def subscribe(self, exchange, token):
        key = (EXCH_TYPE.get(exchange, 1), str(token))
        with self.lock:
            new = key not in self.wanted
            self.wanted.add(key)
        if new and self.connected and self.sws is not None:
            try:
                self.sws.subscribe(f'm2_{key[0]}_{key[1]}', 2, [{'exchangeType': key[0], 'tokens': [key[1]]}])
            except Exception as e:
                self.error = str(e)
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def get(self, exchange, token):
        return self.ticks.get((EXCH_TYPE.get(exchange, 1), str(token)))

    def history(self, exchange, token):
        with self.lock:
            return list(self.hist.get((EXCH_TYPE.get(exchange, 1), str(token)), []))

    # -- internals
    def _handle(self, msg):
        if not isinstance(msg, dict) or 'last_traded_price' not in msg:
            return
        try:
            key = (int(msg.get('exchange_type', 1)), str(msg.get('token', '')).strip('"\' \x00'))
            f = lambda k: float(msg[k]) / 100.0 if msg.get(k) not in (None, '') else None   # prices come in paise
            now = time.time()
            old = self.ticks.get(key, {})
            tk = {'ltp': f('last_traded_price'), 'open': f('open_price_of_the_day') or old.get('open'),
                  'high': f('high_price_of_the_day') or old.get('high'), 'low': f('low_price_of_the_day') or old.get('low'),
                  'prev_close': f('closed_price') or old.get('prev_close'), 'ts': now}
            if not tk['ltp']:
                return
            self.ticks[key] = tk
            with self.lock:
                self.hist.setdefault(key, __import__('collections').deque(maxlen=20000)).append((now, tk['ltp']))
            self.last_msg = now
        except Exception as e:
            self.error = f'tick parse: {e}'

    def _subscribe_all(self):
        with self.lock:
            wanted = list(self.wanted)
        by_ex = {}
        for et, tok in wanted:
            by_ex.setdefault(et, []).append(tok)
        for et, toks in by_ex.items():
            self.sws.subscribe(f'm2_{et}', 2, [{'exchangeType': et, 'tokens': toks}])

    def _run(self):
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        wait = 3
        while True:
            try:
                smart = angel_login(self.creds['client_id'], self.creds['api_key'], self.creds['mpin'], self.creds['totp'])
                d = smart._login_data
                sws = SmartWebSocketV2(d['jwtToken'], self.creds['api_key'], self.creds['client_id'],
                                       d.get('feedToken') or smart.getfeedToken())

                def on_open(wsapp):
                    self.connected, self.error = True, None
                    try:
                        self._subscribe_all()
                    except Exception as e:
                        self.error = f'subscribe: {e}'

                def on_data(wsapp, message):
                    self._handle(message)

                def on_error(wsapp, error):
                    self.error = f'websocket: {error}'

                def on_close(wsapp, *a):
                    self.connected = False

                sws.on_open, sws.on_data, sws.on_error, sws.on_close = on_open, on_data, on_error, on_close
                self.sws = sws
                wait = 3
                sws.connect()                                   # blocks until the socket closes
            except Exception as e:
                self.error = f'feed: {e}'
                wait = min(wait * 2, 120)
            self.connected = False
            time.sleep(wait)


@st.cache_resource(show_spinner=False)
def _feed_for(client_id, api_key, mpin, totp):
    return TickFeed({'client_id': client_id, 'api_key': api_key, 'mpin': mpin, 'totp': totp})


def get_feed():
    c = st.session_state['creds']
    return _feed_for(c['client_id'], c['api_key'], c['mpin'], c['totp'])


_RATE_LOCK = threading.Lock()
_LAST_CALL = [0.0]
MIN_GAP = 0.7            # seconds between two Angel candle requests (Angel allows ~3/sec; we stay well under)
RECHECK_SECS = 45        # an already-loaded symbol is only topped up with new candles this often


def _is_rate_msg(msg):
    m = str(msg).lower()
    return 'access rate' in m or 'exceeding' in m or 'too many' in m or 'rate limit' in m


def _call_candle_api(smart, params):
    """one getCandleData call: spaced out + retried with back-off when Angel says 'exceeding access rate'."""
    last = 'candle request failed'
    for i in range(5):
        with _RATE_LOCK:
            wait = MIN_GAP - (time.time() - _LAST_CALL[0])
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL[0] = time.time()
        try:
            res = smart.getCandleData(params)
        except Exception as e:
            last = str(e)
            if _is_rate_msg(last):
                time.sleep(2 * (i + 1))
                continue
            raise
        if res and res.get('status'):
            return res
        last = (res or {}).get('message', 'candle request failed')
        if _is_rate_msg(last):
            time.sleep(2 * (i + 1))
            continue
        raise ValueError(last)
    raise ValueError(f'Angel rate limit: {last}')


def _fetch_range(smart, exchange, token, interval, frm, to):
    step = pd.Timedelta(days=INTERVAL_MAX_DAYS.get(interval, 28))      # Angel per-request limits
    rows, cur = [], frm
    while cur < to:
        end = min(cur + step, to)
        res = _call_candle_api(smart, {'exchange': exchange, 'symboltoken': token, 'interval': interval,
                                       'fromdate': cur.strftime('%Y-%m-%d %H:%M'),
                                       'todate': end.strftime('%Y-%m-%d %H:%M')})
        rows += res.get('data') or []
        cur = end + pd.Timedelta(minutes=1)
    if not rows:
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])
    df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for c_ in ('Open', 'High', 'Low', 'Close'):
        df[c_] = pd.to_numeric(df[c_], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    return df[['Date', 'Open', 'High', 'Low', 'Close']]


def fetch_candles(smart, exchange, symbol, tf, start_date):
    """-> raw candles (Date, Open, High, Low, Close), tz-naive IST, oldest first."""
    interval, _, warm_days = TF[tf]
    token = _instrument_token(exchange, symbol)
    df = _fetch_range(smart, exchange, token, interval, pd.Timestamp(start_date) - pd.Timedelta(days=warm_days), now_ist())
    if df.empty:
        raise ValueError('no candles returned')
    return df.drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)


@st.cache_resource
def _candle_store():
    return {'lock': threading.Lock(), 'data': {}}


def load_candles(smart, exchange, symbol, tf, start_day):
    """History is downloaded ONCE per symbol/timeframe; after that only the newest candles are fetched
    (1 small request, at most every RECHECK_SECS) -> no more 'exceeding access rate'."""
    S = _candle_store()
    key = (exchange, symbol, tf, start_day)
    with S['lock']:
        e = S['data'].get(key)
        if e is None:
            e = S['data'][key] = {'df': fetch_candles(smart, exchange, symbol, tf, start_day),
                                  'checked': time.time(), 'error': None}
        elif time.time() - e['checked'] >= RECHECK_SECS:
            try:
                interval = TF[tf][0]
                back = pd.Timedelta(days=5 if interval == 'ONE_DAY' else 2)
                new = _fetch_range(smart, exchange, _instrument_token(exchange, symbol), interval,
                                   e['df']['Date'].iloc[-1] - back, now_ist())
                if not new.empty:
                    df = pd.concat([e['df'], new]).drop_duplicates('Date', keep='last').sort_values('Date')
                    e['df'] = df.reset_index(drop=True)
                e['error'] = None
            except Exception as ex:                         # keep showing the last good data, retry next refresh
                e['error'] = str(ex)
            e['checked'] = time.time()
        return e['df'].copy(), e['checked'], e['error']


# =============================================================== candle preparation
def closed_only(df, delta, close_t, now):
    """drop the candle that has not finished yet (its end = start+length, capped at market close)."""
    sess_end = df['Date'].dt.normalize() + pd.Timedelta(hours=close_t.hour, minutes=close_t.minute)
    end = pd.concat([df['Date'] + delta, sess_end], axis=1).min(axis=1)
    return df[end <= now].reset_index(drop=True)


def to_4h(h, close_t):
    """hourly -> 4H: per day, bars 1-4 = first 4H candle, the rest = second (e.g. NSE 09:15 + 13:15)."""
    h = h.copy()
    h['day'] = h['Date'].dt.normalize()
    h['n'] = h.groupby('day').cumcount() // 4
    out = (h.groupby(['day', 'n'])
             .agg(Date=('Date', 'first'), Open=('Open', 'first'), High=('High', 'max'),
                  Low=('Low', 'min'), Close=('Close', 'last'), cnt=('Close', 'size'), last=('Date', 'last'))
             .reset_index(drop=True))
    day_close = out['Date'].dt.normalize() + pd.Timedelta(hours=close_t.hour, minutes=close_t.minute)
    ok = (out['cnt'] == 4) | ((out['last'] + pd.Timedelta(hours=1)) >= day_close)
    return out[ok][['Date', 'Open', 'High', 'Low', 'Close']].reset_index(drop=True)


def prepare(raw, tf, close_t, now):
    _, delta, _ = TF[tf]
    df = closed_only(raw, delta, close_t, now)
    return to_4h(df, close_t) if tf == '4H' else df


# =============================================================== M2 live state
def compute_m2(df, profit_pct, max_lots, start, qty=1.0, span=None, candle='every', amount=None, skip=None, entry='above'):
    trace = []
    trades = run_m2(df, profit_pct, max_lots, start_date=start, lot_qty=qty, trace=trace, span=span, candle=candle, lot_amount=amount, skip=skip, entry=entry)
    tr = pd.DataFrame(trace)
    if tr.empty:
        return tr, trades
    prev_lots = tr['Lots'].shift(fill_value=0)
    prev_exit = tr['Action'].shift(fill_value='') == 'EXIT'
    buy = tr['Bought'].astype(bool)
    exit_ = tr['Action'] == 'EXIT'
    tr['Signal'] = np.select([buy & exit_, exit_, buy], ['BUY+EXIT', 'EXIT', 'BUY'], 'HOLD')
    for col in ('AvgPrice', 'Qty'):                          # never bought -> column was all None (object) -> .round() crashed
        tr[col] = pd.to_numeric(tr[col], errors='coerce')
    tr['Unrealised %'] = ((tr['Close'] - tr['AvgPrice']) / tr['AvgPrice'] * 100).round(2)
    return tr, trades


def get_smart():
    """Logged-in SmartConnect for THIS browser session (re-logs in automatically if it expired)."""
    c = st.session_state['creds']
    if 'smart' not in st.session_state:
        st.session_state['smart'] = angel_login(c['client_id'], c['api_key'], c['mpin'], c['totp'])
    return st.session_state['smart']


def relogin():
    st.session_state.pop('smart', None)
    return get_smart()


# =============================================================== strategy files (folder / file)
def safe(s):
    return re.sub(r'[^A-Za-z0-9 _.%\-]', '_', str(s)).strip() or 'x'


def user_dir():
    return os.path.join(STRAT_ROOT, safe(st.session_state['creds']['client_id']))


def list_folders():
    d = user_dir()
    return sorted(f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))) if os.path.isdir(d) else []


def delete_folder(folder):
    d = os.path.join(user_dir(), safe(folder))
    if os.path.isdir(d):
        shutil.rmtree(d)


def list_files(folder):
    d = os.path.join(user_dir(), safe(folder))
    return sorted(f for f in os.listdir(d) if f.endswith('.json')) if os.path.isdir(d) else []


def load_strategy(folder, fname):
    with open(os.path.join(user_dir(), safe(folder), fname), encoding='utf-8') as f:
        return json.load(f)


def delete_strategy(folder, fname):
    os.remove(os.path.join(user_dir(), safe(folder), fname))


def create_strategies(folder, prefix, exchange, symbols, tfs, capital, sizes, close, lots_list, exits, candles, start, entries=('above',)):
    """one file per symbol x timeframe x lots x size x exit option x candle mode. Existing file is NOT overwritten (keeps its start time)."""
    d = os.path.join(user_dir(), safe(folder))
    os.makedirs(d, exist_ok=True)
    made, skipped = [], []
    for sym in symbols:
      for tf in tfs:
        for lots in lots_list:
          for kind, val in sizes:
            for ex in exits:
              for ent in entries:
                for cd in candles:
                    cdc = CANDLE_OPTIONS[cd]
                    amt = capital / len(symbols) / lots if kind == 'eq' else val      # equal weightage: capital / symbols / max lots
                    name = (f"{prefix + ' ' if prefix else ''}{sym} {tf} {lots}lots {exit_label(ex)}"
                            + ('' if cdc == 'every' else f' {cdc}')
                            + (f' amt{val:g}' if kind == 'amt' else (' eq' if kind == 'eq' else (f' q{val:g}' if len(sizes) > 1 else '')))
                            + {'below': ' belowcloud', 'inside': ' insidecloud'}.get(ent, ''))
                    path = os.path.join(d, safe(name) + '.json')
                    if os.path.exists(path):
                        skipped.append(name)
                        continue
                    is_span = ex in SPAN_OPTIONS
                    s = {'name': name, 'folder': folder, 'mode': 'paper', 'exchange': exchange, 'symbol': sym,
                         'tf': tf, 'profit': None if is_span else ex, 'span': SPAN_OPTIONS.get(ex),
                         'exit_label': exit_label(ex), 'candle': cdc, 'lots': lots, 'capital': capital,
                         'qty': val if kind == 'qty' else 1.0, 'amount': amt if kind in ('amt', 'eq') else None,
                         'sizing': 'equal' if kind == 'eq' else ('amount' if kind == 'amt' else 'qty'), 'entry': ent,
                         'close': close, 'start': start,
                         'created': now_ist().to_pydatetime().strftime('%Y-%m-%d %H:%M')}
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(s, f, indent=1)
                    made.append(name)
    return made, skipped


# =============================================================== evaluate one saved strategy
def get_candles(s):
    """-> (raw candles, time of last data check, error text or None)"""
    day = pd.Timestamp(s['start']).normalize().strftime('%Y-%m-%d')
    try:
        return load_candles(get_smart(), s['exchange'], s['symbol'], s['tf'], day)
    except Exception as e:
        if _is_rate_msg(e):
            raise
        return load_candles(relogin(), s['exchange'], s['symbol'], s['tf'], day)   # session expired?


def first_bar_after_start(df, tf, close_t, start):
    """Date of the first bar that CLOSES after the start time (None if no such closed bar yet).
    Bars that were already closed at start never trade. (Old code compared the bar OPEN time with the
    start, so a bar that was running at start - e.g. today's daily candle - was wrongly ignored.)"""
    if df.empty:
        return None
    bar = pd.Timedelta(hours=4) if tf == '4H' else TF[tf][1]
    sess_end = df['Date'].dt.normalize() + pd.Timedelta(hours=close_t.hour, minutes=close_t.minute)
    end = pd.concat([df['Date'] + bar, sess_end], axis=1).min(axis=1)
    ok = df[end > pd.Timestamp(start)]
    return None if ok.empty else ok['Date'].iloc[0]


def evaluate(s):
    """-> dict with tr (bar log), trades and the numbers shown in the summary."""
    h, m = [int(x) for x in s['close'].split(':')]
    raw, checked, data_err = get_candles(s)
    now = pd.Timestamp(s['paused_at']) if s.get('paused_at') else now_ist()      # paused = frozen as of that moment
    df = prepare(raw, s['tf'], dt.time(h, m), now)
    skip = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in s.get('gaps', [])]
    eff_start = first_bar_after_start(df, s['tf'], dt.time(h, m), s['start'])
    tr, trades = compute_m2(df, s.get('profit') or 0.0, int(s['lots']), eff_start if eff_start is not None else pd.Timestamp('2100-01-01'),
                            s['qty'], span=s.get('span'), candle=s.get('candle', 'every'), amount=s.get('amount'), skip=skip, entry=s.get('entry', 'above'))
    diag = {'bars': len(df), 'last_bar': None if df.empty else df['Date'].iloc[-1], 'cloud': None}
    if len(df):
        try:
            cl = add_ichimoku_cloud(df).iloc[-1]
            if not (pd.isna(cl['CloudTop']) or pd.isna(cl['CloudBottom'])):
                diag['cloud'] = {'close': float(cl['Close']), 'top': float(cl['CloudTop']), 'bottom': float(cl['CloudBottom']),
                                 'pos': 'Above cloud' if cl['Close'] > cl['CloudTop'] else ('Below cloud' if cl['Close'] < cl['CloudBottom'] else 'Inside cloud')}
        except Exception:
            pass
    out = {'s': s, 'diag': diag, 'tr': tr, 'trades': trades, 'status': 'Waiting for 1st candle', 'last': None,
           'checked': checked, 'data_err': data_err,
           'closed_n': 0, 'closed_pnl': 0.0, 'open_pnl': 0.0, 'invested': 0.0}
    if tr.empty:
        return out
    last = tr.iloc[-1]
    flat = last['Signal'] in ('EXIT', 'BUY+EXIT') or int(last['Lots']) == 0 or pd.isna(last['AvgPrice'])
    openrows = trades[trades['Exit Reason'].str.startswith('Open at end')] if not trades.empty else trades
    closed = trades.drop(openrows.index) if not trades.empty else trades
    out.update(last=last, flat=flat, closed_n=len(closed),
               closed_pnl=float(closed['PnL (pts x qty)'].sum()) if len(closed) else 0.0,
               open_pnl=0.0 if flat or openrows.empty else float(openrows['PnL (pts x qty)'].sum()),
               invested=0.0 if flat else float(last['Qty']) * float(last['AvgPrice']),
               status='Flat' if flat else f"Holding {int(last['Lots'])} lots")
    return out


def money(x):
    return f"{x:,.0f}"


# =============================================================== pages
def checkbox_dropdown(title, options, default, key, fmt=str):
    """Drop-down with a check box beside every option + Select all. Returns the ticked options."""
    base = f'cb_{key}'
    for o in options:
        if f'{base}_{o}' not in st.session_state:
            st.session_state[f'{base}_{o}'] = o in default

    def _toggle_all():
        v = st.session_state[f'{base}__all']
        for o in options:
            st.session_state[f'{base}_{o}'] = v

    chosen = [o for o in options if st.session_state[f'{base}_{o}']]
    st.session_state[f'{base}__all'] = len(chosen) == len(options)
    with st.popover(f'{title} · {len(chosen)} selected', width='stretch'):
        st.checkbox('Select all', key=f'{base}__all', on_change=_toggle_all)
        for o in options:
            st.checkbox(fmt(o), key=f'{base}_{o}')
    return chosen


NIFTY50_STOCKS = ['ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
                  'BAJAJFINSV', 'BEL', 'BHARTIARTL', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'ETERNAL',
                  'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDIGO',
                  'INFY', 'ITC', 'JIOFIN', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC',
                  'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SHRIRAMFIN', 'SUNPHARMA', 'TATACONSUMER',
                  'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'TRENT', 'ULTRACEMCO', 'WIPRO']


@st.cache_data(ttl=86400, show_spinner='Loading instrument list ...')
def instrument_choices(exchange):
    """{'Index': [...], 'Stocks': [...], 'Commodity': [...]} for the drop-downs (built from Angel's instrument list)."""
    try:
        master = _scrip_master()
    except Exception:
        return {}
    rows = [r for r in master if r.get('exch_seg') == exchange]
    out = {}
    idx = sorted({r['symbol'] for r in rows if r.get('instrumenttype') == 'AMXIDX'})
    if idx:
        out['Index'] = idx
    if exchange == 'NSE':
        eq = {r['symbol'] for r in rows if r['symbol'].endswith('-EQ')}
        st_ = [x for x in NIFTY50_STOCKS if x + '-EQ' in eq]
        if st_:
            out['Stocks'] = st_
    if exchange == 'MCX':
        today, best = now_ist().to_pydatetime(), {}
        for r in rows:
            if r.get('instrumenttype') != 'FUTCOM':
                continue
            try:
                exp = dt.datetime.strptime(r['expiry'], '%d%b%Y')
            except Exception:
                continue
            if exp.date() >= today.date() and (r['name'] not in best or exp < best[r['name']][0]):
                best[r['name']] = (exp, r['symbol'])
        if best:
            out['Commodity'] = [v[1] for k, v in sorted(best.items())]      # nearest-expiry future of each commodity
    return out


def lbl(text):                                                # same label look for EVERY field -> rows line up
    st.markdown(f"<div style='font-size:14px;line-height:1.5;padding:0 0 8px 0;opacity:.85'>{text}</div>", unsafe_allow_html=True)


def section_title(title, hint=''):
    st.markdown("<div style='display:flex;align-items:center;gap:10px;margin:2px 0 10px 0'>"
                "<span style='width:5px;height:22px;background:#ff4b4b;border-radius:2px;display:inline-block'></span>"
                f"<b style='font-size:17px'>{title}</b><span style='opacity:.6;font-size:13px'>{hint}</span></div>",
                unsafe_allow_html=True)


def typed_values(txt):
    out = []
    for x in txt.split(','):
        try:
            v = float(x.strip())
        except ValueError:
            continue
        if v > 0 and v not in out:
            out.append(v)
    return out


def page_builder():
    st.markdown('# Strategy Builder')
    with st.container(border=True):
        c1, c2 = st.columns(2)                                # ---- basic details
        with c1:
            lbl('Folder Name')
            folder = st.selectbox('Folder Name', list_folders(), index=None, accept_new_options=True,
                                  placeholder='Select or create a folder', label_visibility='collapsed')
        with c2:
            lbl('File Name')
            prefix = st.text_input('File Name', label_visibility='collapsed', placeholder='e.g. NSE morning')
        c1, c2, c3 = st.columns(3)
        with c1:
            lbl('Exchange')
            exchange = st.selectbox('Exchange', ['NSE', 'MCX', 'NFO', 'BSE'], label_visibility='collapsed')
        with c2:
            lbl('Capital')
            capital = st.number_input('Capital', min_value=0.0, value=1000000.0, step=10000.0, label_visibility='collapsed')
        with c3:
            lbl('Candle timeframe (one file each)')
            tfs = checkbox_dropdown('Select timeframe', list(TF), ['1D'], 'tf')

        with st.container(border=True):                       # ---- ENTRY CONDITION
            section_title('Entry condition', 'Ichimoku cloud, Method 2 - index, stocks, candle, position size (live: starts from now)')
            picked = []
            groups = list(instrument_choices(exchange).items())
            cols = st.columns(len(groups) + 1)
            for col, (label, opts) in zip(cols, groups):
                with col:
                    lbl(f'{label} ({len(opts)})')
                    picked += checkbox_dropdown(f'Select {label.lower()}', opts, [], f'sym_{exchange}_{label}')
            with cols[-1]:
                lbl('Or type names (comma separated)')
                typed_txt = st.text_input('Or type names', label_visibility='collapsed',
                                          placeholder='Nifty 50, RELIANCE, GOLDM05DEC25FUT',
                                          help='Index: Nifty 50  |  Stock: RELIANCE (or RELIANCE-EQ)  |  MCX: exact future symbol. '
                                               'Typed names are added to the ticked ones.')
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                lbl('Max lots (one file each)')
                lots_list = checkbox_dropdown('Select lots', LOTS_OPTIONS, [10], 'lots')
            with c2:
                lbl('Buy on (one file each)')
                candles = checkbox_dropdown('Select candle', list(CANDLE_OPTIONS), ['Every candle'], 'candle')
            with c3:
                lbl('Price vs cloud (one file each)')
                entries = [ENTRY_OPTIONS[x] for x in checkbox_dropdown('Select entry', list(ENTRY_OPTIONS), ['Above cloud'], 'entry')]
            with c4:
                lbl('Market close')
                close_t = st.time_input('Market close', dt.time(23, 30) if exchange == 'MCX' else dt.time(15, 30),
                                        key=f'close_{exchange}', label_visibility='collapsed')
            c = st.columns(3)
            with c[0]:
                lbl('Position - size each lot by')
                modes = checkbox_dropdown('Select position type', ['Quantity', 'Amount', 'Equal weightage'], ['Quantity'], 'mode')
            sizes, blocks = [], []
            if 'Quantity' in modes:
                blocks.append('qty')
            if 'Amount' in modes:
                blocks.append('amt')
            if 'Equal weightage' in modes:
                sizes.append(('eq', 0))
            for i, kind in enumerate(blocks):
                row = c if i == 0 else st.columns(3)
                a_, b_ = (1, 2) if i == 0 else (0, 1)
                if kind == 'qty':
                    with row[a_]:
                        lbl('Quantity per lot')
                        vals = [float(q) for q in checkbox_dropdown('Select quantity', QTY_OPTIONS, [1], 'qty')]
                    with row[b_]:
                        lbl('Or type quantity')
                        vals += typed_values(st.text_input('Or type quantity', label_visibility='collapsed',
                                                           placeholder='3, 15, 40', key='size_txt_qty'))
                else:
                    with row[a_]:
                        lbl('Amount per lot')
                        vals = [float(q) for q in checkbox_dropdown('Select amount', AMOUNT_OPTIONS, [10000], 'amt')]
                    with row[b_]:
                        lbl('Or type amount')
                        vals += typed_values(st.text_input('Or type amount', label_visibility='collapsed',
                                                           placeholder='15000, 75000', key='size_txt_amt',
                                                           help='Money put in every lot (units = amount / price of that lot).'))
                sizes += [(kind, v) for v in dict.fromkeys(vals)]

        with st.container(border=True):                       # ---- EXIT CONDITION
            section_title('Exit condition', 'profit target % on average entry, or Span exit')
            c = st.columns(3)
            with c[0]:
                lbl('Profit target % (one file each)')
                pct_exits = checkbox_dropdown('Select target', TARGET_OPTIONS, [1, 2], 'exits_pct', fmt=exit_label)
            with c[1]:
                lbl('Span exit (one file each)')
                span_exits = checkbox_dropdown('Select span exit', list(SPAN_ONLY), [], 'exits_span')
            with c[2]:
                lbl('Cloud exit (one file each)')
                cloud_exits = checkbox_dropdown('Select cloud exit', list(CLOUD_EXIT), [], 'exits_cloud')
            targets = pct_exits + span_exits + cloud_exits

        start = now_ist().strftime('%Y-%m-%d %H:%M')          # live only: always start from NOW (no past backfill)
        symbols, seen = [], set()
        for x in picked + [t.strip() for t in typed_txt.split(',') if t.strip()]:
            if x.upper() not in seen:
                seen.add(x.upper())
                symbols.append(x)
        n = len(symbols) * len(tfs) * len(lots_list) * len(sizes) * len(targets) * len(candles) * len(entries)
        st.caption(f'{len(symbols)} symbol(s) x {len(tfs)} timeframe(s) x {len(lots_list)} lots x {len(sizes)} size(s) x '
                   f'{len(targets)} exits x {len(candles)} candle mode(s) x {len(entries)} entry side(s) = {n} file(s) will be saved.  Start = {start}  '
                   f'(live - first candle that closes after this)')
        st.caption('Live trading (real orders) is not included - run Paper first and check the numbers.')
        if st.button('▶ Start', type='primary'):
            fname = (folder or '').strip()
            if not fname:
                st.error('Give a folder name.')
            elif n == 0:
                st.error('Pick (or type) at least one symbol, timeframe, Max lots, position size, entry side, candle option and exit option.')
            elif n > 150:
                st.error('Too many files (max 150 at a time).')
            else:
                try:
                    for sym in symbols:
                        _instrument_token(exchange, sym)      # fail early if a symbol is wrong
                except Exception as e:
                    st.error(str(e))
                    return
                made, skipped = create_strategies(fname, (prefix or '').strip(), exchange, symbols, tfs, capital, sizes,
                                                  close_t.strftime('%H:%M'), lots_list, targets, candles, start, entries)
                if made:
                    st.success(f'Saved {len(made)} file(s) in folder "{fname}". Open "My Strategies" to watch them live.')
                if skipped:
                    st.warning(f'{len(skipped)} file(s) already existed and were kept as they are (their start time is unchanged).')


def save_strategy(folder, fname, s):
    with open(os.path.join(user_dir(), safe(folder), fname), 'w', encoding='utf-8') as f:
        json.dump(s, f, indent=1)


def exit_strategy(folder, fname, ev):
    """Stop a strategy: open position is squared off at the last closed price and the numbers are frozen."""
    s = load_strategy(folder, fname)
    s['stopped'] = True
    s['final'] = {'closed_n': ev['closed_n'] + (1 if ev['last'] is not None and not ev['flat'] else 0) if ev else 0,
                  'realized': (ev['closed_pnl'] + ev['open_pnl']) if ev else 0.0,
                  'at': now_ist().strftime('%Y-%m-%d %H:%M')}
    save_strategy(folder, fname, s)


def log_event(s, text):
    s.setdefault('events', []).append({'time': now_ist().strftime('%Y-%m-%d %H:%M:%S'), 'event': text})


def strat_state(s):
    return 'EXITED' if s.get('stopped') else ('PAUSED' if s.get('paused_at') else 'RUNNING')


def pause_strategy(folder, fname):
    s = load_strategy(folder, fname)
    s['paused_at'] = now_ist().strftime('%Y-%m-%d %H:%M:%S')
    log_event(s, 'Stopped (paused) - no buy / exit until Resume')
    save_strategy(folder, fname, s)


def resume_strategy(folder, fname):
    s = load_strategy(folder, fname)
    delta = TF[s['tf']][1]
    a = pd.Timestamp(s['paused_at']) - delta                       # bars that closed during the pause are ignored
    s.setdefault('gaps', []).append([a.strftime('%Y-%m-%d %H:%M:%S'), now_ist().strftime('%Y-%m-%d %H:%M:%S')])
    s['paused_at'] = None
    log_event(s, 'Resumed - candles that closed while paused are skipped')
    save_strategy(folder, fname, s)


def restart_strategy(folder, fname):
    """Start again after Exit: fresh from NOW (old trades are cleared, event log is kept)."""
    s = load_strategy(folder, fname)
    for k in ('stopped', 'final', 'paused_at'):
        s.pop(k, None)
    s['gaps'] = []
    s['start'] = now_ist().strftime('%Y-%m-%d %H:%M')
    log_event(s, f"Started again from {s['start']}")
    save_strategy(folder, fname, s)


def session_times(s, base):
    """(open time, [candle close times]) of one trading day."""
    ch, cm = [int(x) for x in s['close'].split(':')]
    oh, om = (9, 0) if s['exchange'] == 'MCX' else (9, 15)
    o, c = base + pd.Timedelta(hours=oh, minutes=om), base + pd.Timedelta(hours=ch, minutes=cm)
    if s['tf'] == '1D':
        return o, [c]
    if s['tf'] == '4H':
        return o, [o + pd.Timedelta(hours=4), c]
    delta, times, x = TF[s['tf']][1], [], o + TF[s['tf']][1]
    while x < c:
        times.append(x)
        x += delta
    return o, times + [c]


def next_candle_close(s, now):
    for i in range(8):
        base = pd.Timestamp(now).normalize() + pd.Timedelta(days=i)
        if base.weekday() >= 5:
            continue
        for x in session_times(s, base)[1]:
            if x > now:
                return x
    return None


def current_bar_start(s, now):
    """start of the candle that is running right now (None when the market is closed)."""
    base = pd.Timestamp(now).normalize()
    if base.weekday() >= 5:
        return None
    o, ends = session_times(s, base)
    if now < o or now >= ends[-1]:
        return None
    prev = [x for x in ends if x <= now]
    return prev[-1] if prev else o


def quote_panel(fol, fname):
    """LIVE price strip - re-runs every ~1.5 s on its own (does not touch the candle maths)."""
    try:
        s = load_strategy(fol, fname)
        feed = get_feed()
        token = _instrument_token(s['exchange'], s['symbol'])
        feed.subscribe(s['exchange'], token)
    except Exception as e:
        st.caption(f'Live price not available: {e}')
        return
    tk = feed.get(s['exchange'], token)
    if tk is None:
        st.caption('Live price: ' + (f'error - {feed.error}' if feed.error else
                   ('connecting ...' if not feed.connected else 'connected, waiting for the first tick (market closed?)')))
        return
    now = now_ist()
    age = time.time() - tk['ts']
    ltp, prev = tk['ltp'], tk.get('prev_close')
    chg = (ltp - prev) if prev else None
    c = st.columns(6)
    c[0].metric('LTP' + (' 🟢' if age < 10 else ' 🟡'), f'{ltp:,.2f}',
                None if chg is None else f'{chg:+,.2f} ({chg / prev * 100:+.2f}%)')
    c[1].metric('Day open', '-' if not tk.get('open') else f"{tk['open']:,.2f}")
    c[2].metric('Day high', '-' if not tk.get('high') else f"{tk['high']:,.2f}")
    c[3].metric('Day low', '-' if not tk.get('low') else f"{tk['low']:,.2f}")
    start = current_bar_start(s, now)
    if start is not None:
        pts = [p for ts_, p in feed.history(s['exchange'], token) if ts_ist(ts_) >= start]
        if pts:
            c[4].metric(f"Running {s['tf']} candle", f'{pts[-1]:,.2f}', f'O {pts[0]:,.2f} | H {max(pts):,.2f} | L {min(pts):,.2f}', delta_color='off')
    pos = st.session_state.get(f'pos_{fol}_{fname}')
    if pos and pos.get('lots', 0) > 0 and pos.get('avg'):
        pnl = (ltp - pos['avg']) * pos['units']
        c[5].metric('Open P&L @ LTP', f'{pnl:,.2f}', f"{(ltp / pos['avg'] - 1) * 100:+.2f}%")
    st.caption(f"tick {ts_ist(tk['ts']):%H:%M:%S}  |  entries / exits use CLOSED candles only; this strip is just the live price")


def csv_button(label, df, key):
    st.download_button(label, df.to_csv(index=False).encode('utf-8'), file_name=f'{key}.csv', mime='text/csv', key=f'dl_{key}')


def page_file(topic, refresh, fol, fname):
    """One strategy on its own page: controls, settings, key numbers and separate tabs."""
    if st.button('← Back to My Strategies'):
        st.session_state.pop('open_file', None)
        st.rerun()
    try:
        s0 = load_strategy(fol, fname)
    except FileNotFoundError:
        st.session_state.pop('open_file', None)
        st.rerun()
        return
    st.markdown(f"# {fol} / {s0['name']}")
    if not s0.get('stopped'):
        st.fragment(run_every=1.5)(quote_panel)(fol, fname)

    def body():
        s = load_strategy(fol, fname)
        state = strat_state(s)
        b = st.columns([1, 1, 1, 1, 2.6])                              # ---- controls
        with b[0].popover('Start', width='stretch', disabled=state != 'EXITED'):
            st.write('Start again from now (old trades are cleared)')
            if st.button('✔ Yes', key=f'rs_{fname}', type='primary'):
                restart_strategy(fol, fname)
                st.rerun()
        if b[1].button('Resume', width='stretch', disabled=state != 'PAUSED'):
            resume_strategy(fol, fname)
            st.rerun()
        if b[2].button('Stop', width='stretch', disabled=state != 'RUNNING'):
            pause_strategy(fol, fname)
            st.rerun()
        with b[3].popover('Exit', width='stretch', disabled=state == 'EXITED'):
            st.write('Close the open position at the last closed price and stop for good')
            if st.button('✔ Yes', key=f'ex_{fname}', type='primary'):
                try:
                    ev_ = evaluate(s)
                except Exception:
                    ev_ = None
                exit_strategy(fol, fname, ev_)
                st.rerun()
        b[4].markdown(f'Status: **{state}**')

        size = size_text(s)
        st.info(f"**{s['exchange']}** | {s['symbol']} | {s['tf']} | capital {s['capital']:,.0f} | max lots {s['lots']} | {size} | "
                f"entry {entry_text(s)} | exit {exit_text(s)} | buy {candle_text(s).lower()} | started {s['start']}")

        with st.expander('Edit settings (running-ah irundhaalum maathalaam)'):
            e1, e2, e3 = st.columns(3)
            ex_opts = EXIT_OPTIONS
            cur_ex = s.get('span') and next(k for k, v in SPAN_OPTIONS.items() if v == s['span']) or s.get('profit')
            new_ex = e1.selectbox('Exit', ex_opts, index=ex_opts.index(cur_ex) if cur_ex in ex_opts else 0,
                                  format_func=exit_label, key=f'ed_ex_{fname}')
            new_lots = e2.number_input('Max lots', 1, 1000, int(s['lots']), key=f'ed_lots_{fname}')
            cds = list(CANDLE_OPTIONS)
            cur_cd = next((k for k, v in CANDLE_OPTIONS.items() if v == s.get('candle', 'every')), cds[0])
            new_cd = e3.selectbox('Buy on', cds, index=cds.index(cur_cd), key=f'ed_cd_{fname}')
            e1, e2, e3 = st.columns(3)
            by = e1.selectbox('Size each lot by', ['Quantity', 'Amount'], index=1 if s.get('amount') else 0, key=f'ed_by_{fname}')
            val = e2.number_input('Quantity per lot' if by == 'Quantity' else 'Amount per lot', min_value=0.0,
                                  value=float(s['amount'] if s.get('amount') else s['qty']), key=f'ed_val_{fname}_{by}')
            new_cap = e3.number_input('Capital', min_value=0.0, value=float(s['capital']), step=10000.0, key=f'ed_cap_{fname}')
            new_entry = st.selectbox('Entry', list(ENTRY_OPTIONS), index=list(ENTRY_OPTIONS.values()).index(s.get('entry', 'above')), key=f'ed_ent_{fname}')
            st.caption('Numbers are re-calculated from the start time with the new settings.')
            if st.button('Save settings', key=f'ed_save_{fname}'):
                s['lots'], s['capital'], s['candle'] = int(new_lots), new_cap, CANDLE_OPTIONS[new_cd]
                s['entry'] = ENTRY_OPTIONS[new_entry]
                s['profit'] = None if new_ex in SPAN_OPTIONS else new_ex
                s['span'] = SPAN_OPTIONS.get(new_ex)
                s['exit_label'] = exit_label(new_ex)
                s['qty'], s['amount'] = (val, None) if by == 'Quantity' else (1.0, val)
                log_event(s, f"Settings changed: exit {exit_label(new_ex)}, lots {new_lots}, {by.lower()} {val:g}, buy {new_cd}")
                save_strategy(fol, fname, s)
                st.rerun()

        now = now_ist()
        if state == 'EXITED':
            f = s.get('final', {})
            st.markdown(f"**EXITED** | {s['exchange']} | exited {f.get('at', '')} | timeframe: {s['tf']}")
            m = st.columns(4)
            m[0].metric('Open lots', f"0 / {s['lots']}")
            m[1].metric('Closed trades', f.get('closed_n', 0))
            m[2].metric('Realized P&L', f"{f.get('realized', 0.0):,.2f}")
            m[3].metric('Unrealized P&L', '0.00')
            ev = None
        else:
            try:
                ev = evaluate(s)
            except Exception as e:
                st.error(f'Could not load data: {e}')
                return
            open_txt = '09:00' if s['exchange'] == 'MCX' else '09:15'
            is_open = now.weekday() < 5 and open_txt <= now.strftime('%H:%M') <= s.get('close', '15:30')
            st.markdown(f"**{state}** | {s['exchange']} market **{'OPEN' if is_open else 'CLOSED'}** | now {now:%H:%M:%S} IST | "
                        f"prices updated {ts_ist(ev['checked']).strftime('%H:%M:%S')} | timeframe: {s['tf']} | {size}")
            nxt = next_candle_close(s, now)
            if state == 'PAUSED':
                st.caption(f"Engine: paused since {s['paused_at']} - press Resume to continue")
            elif ev.get('data_err'):
                st.warning('Angel rate limit / data problem - showing the last good candles, retrying on the next refresh: ' + ev['data_err'][:120])
            else:
                st.caption(f"Engine: Waiting for the next candle ({nxt:%H:%M}{'' if nxt.date() == now.date() else nxt.strftime(' %d %b')})" if nxt is not None else 'Engine: -')
            last, tr, trades = ev['last'], ev['tr'], ev['trades']
            holding_ = last is not None and not ev['flat']
            st.session_state[f'pos_{fol}_{fname}'] = ({'lots': int(last['Lots']), 'avg': float(last['AvgPrice']), 'units': float(last['Qty'])}
                                                       if holding_ else {'lots': 0})
            closed = trades[~trades['Exit Reason'].str.startswith('Open at end')] if not trades.empty else trades
            win = f"{(closed['PnL (pts x qty)'] > 0).mean() * 100:.0f}%" if len(closed) else '-'
            m = st.columns(5)
            m[0].metric('Open lots', f"{0 if last is None or ev['flat'] else int(last['Lots'])} / {s['lots']}")
            m[1].metric('Closed trades', ev['closed_n'])
            m[2].metric('Realized P&L', f"{ev['closed_pnl']:,.2f}")
            m[3].metric('Unrealized P&L', f"{ev['open_pnl']:,.2f}")
            m[4].metric('Win rate (closed)', win)

        # ---- topics as tabs: click to view
        tabs = st.tabs(['1. Monitoring', '2. Signals', '3. Live position', '4. Closed trades', '5. Logs'])
        key = f'{fol}_{fname}'.replace(' ', '_')
        with tabs[0]:
            if ev is None:
                st.info('Exited - monitoring stopped.')
            elif ev['last'] is None:
                dg = ev['diag']
                nxt = next_candle_close(s, now)
                st.info(f"Waiting for the first candle that CLOSES after the start time ({s['start']}). "
                        f"Next candle closes at {nxt:%H:%M}{'' if nxt is not None and nxt.date() == now.date() else ('' if nxt is None else nxt.strftime(' %d %b'))}. "
                        'If that candle closes ABOVE the cloud, lot 1 is bought.')
                if dg['cloud'] is None:
                    st.warning(f"Cloud not ready: {dg['bars']} closed bars loaded, need 78+ (Angel returned too little history).")
                else:
                    cl_ = dg['cloud']
                    st.dataframe(pd.DataFrame([{'symbol': s['symbol'], 'latest_closed_bar': str(dg['last_bar'])[:16], 'close': round(cl_['close'], 2),
                                                'cloud_top': round(cl_['top'], 2), 'cloud_bottom': round(cl_['bottom'], 2),
                                                'price_vs_cloud': cl_['pos'], 'bars_loaded': dg['bars']}]), width='stretch', hide_index=True)
                    st.caption('Latest closed bar is before the start time, so it is only shown here - it does not trade.')
            else:
                last = ev['last']
                prev = ev['tr']['Close'].iloc[-2] if len(ev['tr']) > 1 else None
                pos = 'Above cloud' if last['Close'] > last['CloudTop'] else ('Below cloud' if last['Close'] < last['CloudBottom'] else 'Inside cloud')
                st.caption('last closed bar = latest finished candle | change_% = vs previous bar close | cloud = Ichimoku (9/26/52, shift 26)')
                st.dataframe(pd.DataFrame([{'symbol': s['symbol'], 'last_bar': str(last['Date'])[:16], 'close': round(float(last['Close']), 2),
                                            'change_%': round((last['Close'] / prev - 1) * 100, 2) if prev else None,
                                            'cloud_top': round(float(last['CloudTop']), 2), 'cloud_bottom': round(float(last['CloudBottom']), 2),
                                            'price_vs_cloud': pos, 'signal': last['Signal'],
                                            'position': ev['status']}]), width='stretch', hide_index=True)
                st.line_chart(ev['tr'].tail(150).set_index('Date')[['Close', 'CloudTop', 'CloudBottom']])
        with tabs[1]:
            if ev is None or ev['tr'].empty or (ev['tr']['Signal'] == 'HOLD').all():
                st.info('No signal yet.')
            else:
                sg = ev['tr'][ev['tr']['Signal'] != 'HOLD'][['Date', 'Signal', 'Close', 'Lots', 'AvgPrice', 'Unrealised %']]
                sg = sg.sort_values('Date', ascending=False)
                st.dataframe(sg, width='stretch', hide_index=True)
                csv_button('Download signals CSV', sg, f'signals_{key}')
        with tabs[2]:
            if ev is None or ev['last'] is None or ev['flat']:
                st.write('Open position illa.')
            else:
                last = ev['last']
                units = float(last['Qty'])
                st.dataframe(pd.DataFrame([{'symbol': s['symbol'], 'lots': int(last['Lots']), 'avg_entry': round(float(last['AvgPrice']), 2),
                                            'units': round(units, 4), 'invested': round(units * float(last['AvgPrice']), 2),
                                            'last_close': round(float(last['Close']), 2), 'unrealised_%': last['Unrealised %'],
                                            'unrealised_pnl': round(ev['open_pnl'], 2)}]), width='stretch', hide_index=True)
        with tabs[3]:
            if ev is None:
                st.write(f"{s.get('final', {}).get('closed_n', 0)} trade(s) closed before exit (details are not kept after Exit).")
            elif closed.empty:
                st.write('No closed trades yet.')
            else:
                ct = closed.sort_values('Exit Date', ascending=False)
                st.dataframe(ct, width='stretch', hide_index=True)
                csv_button('Download closed trades CSV', ct, f'closed_{key}')
        with tabs[4]:
            evs_ = pd.DataFrame(s.get('events', []))
            st.markdown('**Control events**')
            st.dataframe(evs_.iloc[::-1] if not evs_.empty else pd.DataFrame([{'time': s.get('created', ''), 'event': 'Created'}]),
                         width='stretch', hide_index=True)
            if ev is not None and not ev['tr'].empty:
                st.markdown('**Bar log (latest first)**')
                bl = ev['tr'].sort_values('Date', ascending=False).head(300)
                st.dataframe(bl, width='stretch', hide_index=True)
                csv_button('Download bar log CSV', bl, f'barlog_{key}')

    st.fragment(run_every=int(refresh))(body)()


def make_folder():
    name = st.session_state.get('new_folder_name', '').strip()
    if name:
        os.makedirs(os.path.join(user_dir(), safe(name)), exist_ok=True)
    st.session_state['new_folder_name'] = ''


def pnl_txt(x, pct=None):
    col = 'green' if x > 0 else ('red' if x < 0 else 'gray')
    return f":{col}[{x:,.2f}]" + ('' if pct is None else f" ({pct:+.2f}%)")


def row_stats(s, ev, err):
    if s.get('stopped'):
        f = s.get('final', {})
        return {'state': 'Stopped', 'sub': f"exited {f.get('at', '')}", 'open_n': 0, 'closed_n': f.get('closed_n', 0),
                'realized': f.get('realized', 0.0), 'unreal': 0.0}
    if ev is None:
        return {'state': 'Error', 'sub': (err or '')[:70], 'open_n': 0, 'closed_n': 0, 'realized': 0.0, 'unreal': 0.0}
    last = ev['last']
    holding = last is not None and not ev['flat']
    return {'state': 'Paused' if s.get('paused_at') else 'Running', 'sub': 'Waiting for 1st candle' if last is None else ev['status'],
            'open_n': 1 if holding else 0, 'closed_n': ev['closed_n'], 'realized': ev['closed_pnl'], 'unreal': ev['open_pnl']}


def page_view(topic, refresh):
    of = st.session_state.get('open_file')
    if of:
        if os.path.exists(os.path.join(user_dir(), safe(of[0]), of[1])):
            page_file(topic, refresh, of[0], of[1])
            return
        st.session_state.pop('open_file', None)
    st.markdown('# My Strategies')

    def body():
        c1, c2 = st.columns(2)
        query = c1.text_input('Search', placeholder='🔍 Search folder / strategy...', label_visibility='collapsed').strip().lower()
        with c2.container(border=True):
            n1, n2 = st.columns([4, 1.3])
            n1.text_input('New folder', key='new_folder_name', placeholder='Enter new folder name', label_visibility='collapsed')
            n2.button('📁 + Create', type='primary', width='stretch', on_click=make_folder)
        status_ph = st.empty()

        folders = list_folders()
        data = {}
        for fol in folders:                                        # load every strategy file
            data[fol] = [(f, load_strategy(fol, f)) for f in list_files(fol)]
        shown = [fol for fol in folders if not query or query in fol.lower()
                 or any(query in s['name'].lower() for _, s in data[fol])]
        if not folders:
            st.info('No folders yet. Create one above or in Strategy Builder.')
            return

        evs, errs = {}, {}
        for fol in shown:                                          # evaluate (candles are cached, only newest ones refetched)
            for f, s in data[fol]:
                if s.get('stopped'):
                    continue
                try:
                    evs[(fol, f)] = evaluate(s)
                except Exception as e:
                    errs[(fol, f)] = str(e)
        # ---- one-line live status
        now = now_ist().to_pydatetime()
        any_s = next((s for fol in shown for _, s in data[fol] if not s.get('stopped')), None)
        if any_s is None:
            status_ph.caption(f'Refreshed {now:%H:%M:%S}  |  no running strategy')
        else:
            open_txt = '09:00' if any_s['exchange'] == 'MCX' else '09:15'
            is_open = now.weekday() < 5 and open_txt <= now.strftime('%H:%M') <= any_s.get('close', '15:30')
            stale = [ev for ev in evs.values() if ev.get('data_err')]
            checks = [ev['checked'] for ev in evs.values() if ev.get('checked')]
            msg = (f"🟢 **LIVE**  |  market {'open' if is_open else 'closed'}  |  refreshed {now:%H:%M:%S} (every {int(refresh)}s)  |  "
                   f"Angel data checked {ts_ist(max(checks)).strftime('%H:%M:%S') if checks else '-'}  |  {len(evs)} running file(s) OK")
            if errs or stale:
                msg += f"  |  ⚠️ {len(errs)} error(s), {len(stale)} using last good data (Angel rate limit?)"
            status_ph.markdown(msg)

        for fol in shown:
            files = data[fol]
            running = sum(1 for _, s in files if not s.get('stopped'))
            with st.expander(f'📁 {fol} · Running {running} · Total {len(files)}'):
                if not files:
                    st.caption('Indha folder-la file illa. Strategy Builder-la file create pannunga.')
                    with st.popover('🗑️'):
                        if st.button('✔ Yes', key=f'delfol_{fol}', type='primary'):
                            delete_folder(fol)
                            st.rerun()
                    continue
                stats = {f: row_stats(s, evs.get((fol, f)), errs.get((fol, f))) for f, s in files}
                o = sum(x['open_n'] for x in stats.values())
                cl = sum(x['closed_n'] for x in stats.values())
                real = sum(x['realized'] for x in stats.values())
                unreal = sum(x['unreal'] for x in stats.values())
                cap = sum(float(s.get('capital') or 0) for _, s in files)
                h = st.columns([6, 1.4, 0.7])
                h[0].markdown('**Folder P&L**')
                h[0].markdown(f"Open **{o}** · Closed **{cl}** · Realized {pnl_txt(real)}  \n"
                              f"Unrealized {pnl_txt(unreal)} · **Total** {pnl_txt(real + unreal, (real + unreal) / cap * 100 if cap else None)}")
                with h[1].popover('Exit folder', width='stretch'):
                    st.write(f'{fol} - stop all running ({running})')
                    if st.button('✔ Yes', key=f'exitfol_{fol}', type='primary'):
                        for f, s in files:
                            if not s.get('stopped'):
                                exit_strategy(fol, f, evs.get((fol, f)))
                        st.rerun()
                with h[2].popover('🗑️', width='stretch'):
                    st.write(f'{fol} ({len(files)})')
                    if st.button('✔ Yes', key=f'delfol_{fol}', type='primary'):
                        delete_folder(fol)
                        st.rerun()

                acts = []                                          # latest BUY / EXIT signals since the files started
                for f, s in files:
                    ev = evs.get((fol, f))
                    if ev is not None and not ev['tr'].empty:
                        for _, r in ev['tr'][ev['tr']['Signal'] != 'HOLD'].iterrows():
                            acts.append({'Bar time': str(r['Date'])[:16], 'Strategy': s['name'], 'Signal': r['Signal'],
                                         'Price': round(float(r['Close']), 2), 'Lots after': int(r['Lots'])})
                if acts:
                    st.caption(f'Latest signals ({len(acts)} since start)')
                    st.dataframe(pd.DataFrame(acts).sort_values('Bar time', ascending=False).head(8), width='stretch', hide_index=True)
                else:
                    st.caption('No signal yet - a strategy acts only when a candle CLOSES after its start time.')
                st.divider()

                w = [3, 1.6, 2, 3.4, 0.7, 0.9, 0.7]
                hd = st.columns(w)
                for col, name in zip(hd, ['Name & details', 'Instrument', 'Status', 'P&L', 'Open', 'Exit', 'Del']):
                    col.markdown(f'<span style="opacity:.6;font-weight:600;font-size:14px">{name}</span>', unsafe_allow_html=True)
                for f, s in files:
                    x = stats[f]
                    total = x['realized'] + x['unreal']
                    pct = total / float(s['capital']) * 100 if s.get('capital') else None
                    r = st.columns(w)
                    if r[0].button(f"📄 {s['name']}", key=f'open_{fol}_{f}', type='tertiary'):
                        st.session_state['open_file'] = (fol, f)
                        st.rerun()
                    size = size_text(s)
                    r[0].caption(f"{s['exchange']} · {s['tf']} · {s['lots']} lots · entry {entry_text(s)} · exit {exit_text(s)} · {candle_text(s).lower()} · {size} · capital {s['capital']:g} · started {s['start']}")
                    r[1].write(s['symbol'])
                    if not s.get('stopped'):
                        try:
                            tok_ = _instrument_token(s['exchange'], s['symbol'])
                            get_feed().subscribe(s['exchange'], tok_)
                            tk_ = get_feed().get(s['exchange'], tok_)
                            r[1].caption('LTP ' + (f"{tk_['ltp']:,.2f}" if tk_ else '...'))
                        except Exception:
                            pass
                    dot = {'Running': '🟢', 'Paused': '🟡', 'Stopped': '⚫', 'Error': '🔴'}[x['state']]
                    r[2].write(f"{dot} {x['state']}")
                    r[2].caption(x['sub'])
                    r[3].markdown(f"Open **{x['open_n']}** · Closed **{x['closed_n']}** · Realized {pnl_txt(x['realized'])}  \n"
                                  f"Unrealized {pnl_txt(x['unreal'])} · **Total** {pnl_txt(total, pct)}")
                    if r[4].button('📈', key=f'btn_{fol}_{f}', help='Open'):
                        st.session_state['open_file'] = (fol, f)
                        st.rerun()
                    with r[5].popover('Exit', width='stretch', disabled=s.get('stopped', False)):
                        st.write(s['name'])
                        if st.button('✔ Yes', key=f'exit_{fol}_{f}', type='primary'):
                            exit_strategy(fol, f, evs.get((fol, f)))
                            st.rerun()
                    with r[6].popover('🗑️', width='stretch'):
                        st.write(s['name'])
                        if st.button('✔ Yes', key=f'del_{fol}_{f}', type='primary'):
                            delete_strategy(fol, f)
                            st.rerun()
                    st.divider()

    st.fragment(run_every=int(refresh))(body)()


# =============================================================== app
GLOBAL_CSS = """<style>
div[data-testid='stVerticalBlock']{gap:.6rem}
/* check-box drop-downs look exactly like the other inputs (same grey, same height) */
div[data-testid='stPopover'] button{background:rgba(151,166,195,.15);border:1px solid transparent;min-height:2.5rem;border-radius:.5rem}
div[data-testid='stPopover'] button:hover{border-color:rgba(255,75,75,.6)}
</style>"""


def main():
    st.set_page_config(page_title='Live Paper Trading', layout='wide')
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ---- login screen (nothing is stored on the server)
    if 'creds' not in st.session_state:
        st.title('Live Paper Trading - Method 2')
        st.subheader('Enter your Angel One details')
        api_key = st.text_input('API key', type='password')
        client_id = st.text_input('Client ID')
        mpin = st.text_input('MPIN', type='password')
        totp = st.text_input('TOTP secret key', type='password',
                             help='The long text shown when you enable TOTP (not the 6-digit code)')
        if st.button('Login') and api_key and client_id and mpin and totp:
            creds = {'api_key': api_key.strip(), 'client_id': client_id.strip().upper(),
                     'mpin': mpin.strip(), 'totp': totp.strip()}
            try:
                st.session_state['creds'] = creds
                get_smart()
                st.rerun()
            except Exception as e:
                st.session_state.pop('creds', None)
                st.session_state.pop('smart', None)
                st.error(f'Login failed: {e}')
        return

    page = st.session_state.setdefault('page', 'strategies')      # ---- top navigation
    n1, n2, _ = st.columns([1.3, 1.5, 4])
    if n1.button('📁 My Strategies', type='primary' if page == 'strategies' else 'secondary', width='stretch'):
        st.session_state['page'] = 'strategies'
        st.session_state.pop('open_file', None)
        st.rerun()
    if n2.button('🛠️ Strategy Builder', type='primary' if page == 'builder' else 'secondary', width='stretch'):
        st.session_state['page'] = 'builder'
        st.rerun()
    st.divider()

    if page == 'builder':
        page_builder()
    else:
        page_view('', st.session_state.get('refresh_secs', 60))

    st.write('')
    with st.expander('⚙️ Account'):
        st.write(f"Client ID: {st.session_state['creds']['client_id']}")
        try:
            fd = get_feed()
            st.caption('Live price feed: ' + ('connected' if fd.connected else 'not connected') +
                       (f' | {fd.error}' if fd.error else '') + f' | {len(fd.wanted)} instrument(s) subscribed')
        except Exception:
            pass
        st.number_input('Refresh every (seconds)', 15, 3600, 60, 15, key='refresh_secs')
        if st.button('Logout / change details'):
            for k_ in ('creds', 'smart'):
                st.session_state.pop(k_, None)
            st.rerun()


if __name__ == '__main__':
    main()
