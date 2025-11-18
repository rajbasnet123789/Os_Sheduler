# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import psutil
import pandas as pd
import joblib
import asyncio
import os
import time
import numpy as np
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scheduler-backend-linux")

app = FastAPI(title="AI Scheduler Predictor (Linux)")

# CORS - allow everything for local/dev (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL / SCALER FILES (must exist)
MODEL_FILE = "ai_scheduler_model.pkl"
SCALER_FILE = "ai_scheduler_scaler.pkl"

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    raise RuntimeError(f"Model or scaler file not found. Expected {MODEL_FILE} and {SCALER_FILE}")

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# FEATURES expected by your trained scaler/model
TRAIN_FEATURES = [
    "arrival_time",
    "burst_time",
    "priority",
    "io_wait_time",
    "context_switches",
    "cpu_utilization",
    "waiting_time",
    "turnaround_time",
]

# state to compute deltas across calls
_prev_proc_cpu_seconds: Dict[int, float] = {}   # pid -> total cpu seconds (user+system)
_prev_ctx_switches: Dict[int, int] = {}        # pid -> total ctx switches

# clock ticks per second (used to convert /proc stat utime/stime)
CLK_TCK = os.sysconf(os.sysconf_names['SC_CLK_TCK'])


# -------------------------
# Utility: safe conversion
# -------------------------
def safe(x, default=0.0):
    try:
        if x is None:
            return float(default)
        if isinstance(x, (float, int, np.float64)):
            if np.isnan(x) or np.isinf(x):
                return float(default)
            return float(x)
        return float(x)
    except Exception:
        return float(default)


# -------------------------
# Read /proc/<pid>/stat CPU times (utime + stime)
# returns cpu_seconds (float)
# -------------------------
def read_proc_stat_cpu_seconds(pid: int) -> float:
    stat_path = f"/proc/{pid}/stat"
    try:
        with open(stat_path, "r") as f:
            data = f.read().split()
        # per proc(5) stat format: utime is at index 13, stime at 14 (0-based)
        utime = int(data[13])
        stime = int(data[14])
        total_ticks = utime + stime
        cpu_seconds = total_ticks / CLK_TCK
        return cpu_seconds
    except Exception:
        # fallback to psutil if /proc unreadable
        try:
            p = psutil.Process(pid)
            t = p.cpu_times()
            return safe(t.user + t.system)
        except Exception:
            return 0.0


# -------------------------
# Read /proc/<pid>/io: read_bytes + write_bytes
# -------------------------
def read_proc_io_bytes(pid: int) -> float:
    io_path = f"/proc/{pid}/io"
    try:
        with open(io_path, "r") as f:
            lines = f.readlines()
        read_b = 0
        write_b = 0
        for line in lines:
            if line.startswith("read_bytes:"):
                read_b = int(line.split()[1])
            elif line.startswith("write_bytes:"):
                write_b = int(line.split()[1])
        return float(read_b + write_b)
    except Exception:
        # fallback to psutil if possible
        try:
            p = psutil.Process(pid)
            io = p.io_counters()
            return float((io.read_bytes if io else 0) + (io.write_bytes if io else 0))
        except Exception:
            return 0.0


# -------------------------
# Read /proc/<pid>/status context switches
# returns total voluntary + nonvoluntary (int)
# -------------------------
def read_proc_ctx_switches(pid: int) -> int:
    status_path = f"/proc/{pid}/status"
    try:
        vol = 0
        nonvol = 0
        with open(status_path, "r") as f:
            for line in f:
                if line.startswith("voluntary_ctxt_switches"):
                    vol = int(line.split()[1])
                elif line.startswith("nonvoluntary_ctxt_switches"):
                    nonvol = int(line.split()[1])
        return int(vol + nonvol)
    except Exception:
        # fallback to psutil: Process.num_ctx_switches() returns a namedtuple
        try:
            p = psutil.Process(pid)
            ctx = p.num_ctx_switches()
            return int((ctx.voluntary if ctx else 0) + (ctx.involuntary if ctx else 0))
        except Exception:
            return 0


# -------------------------
# Prime cpu_percent for all processes so first reading isn't zero
# -------------------------
def prime_cpu_percent():
    for p in psutil.process_iter(attrs=[]):
        try:
            # first call returns a meaningless 0.0, so we call to initialize
            p.cpu_percent(None)
        except Exception:
            continue
    # small sleep to allow kernel counters to advance
    time.sleep(0.12)


# -------------------------
# Build feature DataFrame (Linux-focused)
# -------------------------
def build_feature_frame() -> pd.DataFrame:
    rows = []
    now = time.time()

    # prime CPU percent readings
    prime_cpu_percent()

    # iterate processes; use psutil to list process ids and some attrs
    for proc in psutil.process_iter(attrs=["pid", "name", "create_time", "nice"]):
        try:
            info = proc.info
            pid = int(info["pid"])
            name = info.get("name") or f"pid_{pid}"
            nice = safe(info.get("nice", 0))

            # arrival_time: process create time (epoch seconds)
            arrival_time = safe(info.get("create_time", 0.0))

            # total CPU seconds (user+system) from /proc (best accuracy)
            total_cpu_seconds = safe(read_proc_stat_cpu_seconds(pid))

            # burst_time: delta since last measurement (seconds)
            prev_total = safe(_prev_proc_cpu_seconds.get(pid, total_cpu_seconds))
            burst_delta = safe(total_cpu_seconds - prev_total)
            # keep prev stored as the latest total for next call
            _prev_proc_cpu_seconds[pid] = total_cpu_seconds

            # context switches: delta since last measurement
            total_ctx = safe(read_proc_ctx_switches(pid))
            prev_ctx = safe(_prev_ctx_switches.get(pid, total_ctx))
            ctx_delta = int(safe(total_ctx - prev_ctx))
            _prev_ctx_switches[pid] = int(total_ctx)

            # io wait time or bytes (we use bytes here; model expects a numeric io_wait_time)
            io_bytes = safe(read_proc_io_bytes(pid))

            # cpu utilization: use psutil (already primed)
            try:
                cpu_util = safe(proc.cpu_percent(None))
            except Exception:
                cpu_util = 0.0

            # turnaround_time: wall-clock time since process creation
            turnaround_time = safe(now - arrival_time)

            # waiting_time: simple estimate: time not on CPU since creation
            # = turnaround_time - total_cpu_seconds (>=0)
            waiting_time = safe(turnaround_time - total_cpu_seconds)
            if waiting_time < 0:
                waiting_time = 0.0

            rows.append({
                "process_name": name,
                "pid": pid,
                "arrival_time": arrival_time,
                "burst_time": burst_delta,
                "priority": nice,
                "io_wait_time": io_bytes,
                "context_switches": ctx_delta,
                "cpu_utilization": cpu_util,
                "waiting_time": waiting_time,
                "turnaround_time": turnaround_time,
            })

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # process disappeared or not accessible - skip
            continue
        except Exception as e:
            logger.debug("Error reading pid %s: %s", proc.pid if 'proc' in locals() else "?", e)
            continue

    df = pd.DataFrame(rows)
    return df


# -------------------------
# Prepare features exactly matching TRAIN_FEATURES
# -------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # ensure columns exist, order them exactly as expected
    for feat in TRAIN_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    df = df[TRAIN_FEATURES].copy()
    df = df.fillna(0.0)
    df = df.replace([np.inf, -np.inf], 0.0)
    return df


# -------------------------
# Endpoints
# -------------------------
@app.get("/predict")
def predict():
    df = build_feature_frame()
    if df.empty:
        return {"error": "No processes found"}

    # keep process names for response pairing
    names = df["process_name"].tolist()
    features = prepare_features(df)

    # transform & predict
    scaled = scaler.transform(features)   # must match training feature names
    preds = model.predict(scaled)

    return [
        {"process_name": n, "predicted_scheduler": str(p)}
        for n, p in zip(names, preds)
    ]


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            df = build_feature_frame()
            if df.empty:
                await websocket.send_json({"data": []})
                await asyncio.sleep(1)
                continue

            names = df["process_name"].tolist()
            features = prepare_features(df)
            scaled = scaler.transform(features)
            preds = model.predict(scaled)

            results = [
                {"process_name": n, "predicted_scheduler": str(p)}
                for n, p in zip(names, preds)
            ]

            await websocket.send_json({"data": results})
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        await websocket.close()
