"""
NOVA SIGNALS BOT - Top 100 crypto, 5m/15m/1h/4h, 9 strategies, chart + SL/TP on Telegram
Install : pip install requests pandas numpy matplotlib
Run     : export TG_TOKEN="..." ; export TG_CHAT_ID="..." ; python nova_signals_bot.py
Data    : Binance Futures public API (no key needed)
NOT financial advice. Paper-trade first.
"""
import os, io, time, threading, json, datetime
from concurrent.futures import ThreadPoolExecutor
import requests, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TG_TOKEN = os.getenv("TG_TOKEN", "PASTE_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "PASTE_CHAT_ID")
TFS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
TOP_N = 100
# ---- accuracy filters (research based) ----
MIN_GROUPS = {"5m": 4, "15m": 3, "1h": 3, "4h": 3}   # kitne ALAG type ke groups agree karein
HTF = {"5m": "1h", "15m": "1h", "1h": "4h", "4h": "1d"}   # bade timeframe ka trend signal ke saath hona chahiye
ADX_MIN = 20            # bade timeframe par trend itna strong ho
MIN_RVOL = 1.0          # signal candle ka volume 20-candle average se kam na ho
MIN_RISK_PCT = 0.30     # SL distance (%) isse chhota ho to fees profit kha jaati hain
COOLDOWN_CANDLES = 6    # same coin+tf+side ka signal itni candles tak dobara nahi
USE_BTC_FILTER = True   # alt coin ka LONG tab nahi jab BTC bearish ho (aur ulta)
SL_ATR, TP1_R, TP2_R = 1.5, 1.5, 3.0
GROUPS = {"Structure": ["Trend HH/HL-LH/LL", "Breakout+Retest"], "Smart Money": ["Liquidity Sweep", "ICT FVG"],
