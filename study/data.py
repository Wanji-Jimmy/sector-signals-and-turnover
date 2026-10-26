"""Read a checked historical snapshot; never fill absent prices."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_data(directory=ROOT / "data", end=None):
    directory = Path(directory)
    protocol = json.loads((ROOT / "study" / "protocol.json").read_text())
    end = end or protocol["evaluation_end"]
    manifest = json.loads((directory / "manifest.json").read_text())
    sources = {row["symbol"]: row for row in manifest["symbols"]}
    symbols = protocol["symbols"] + [protocol["market"]]
    series = {}
    for symbol in symbols:
        source = sources[symbol]
        content = (directory / source["file"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(f"Snapshot changed: {symbol}; audit the revision first")
        data = json.loads(content)["chart"]["result"][0]
        if data["meta"]["currency"] != "USD":
            raise ValueError(f"Unexpected currency: {symbol}")
        dates = [datetime.fromtimestamp(t, ZoneInfo("America/New_York"))
                 .date().isoformat() for t in data["timestamp"]]
        quote = data["indicators"]["quote"][0]
        arrays = {name: np.asarray(quote[name], dtype=float)
                  for name in ("open", "close", "high", "low")}
        arrays["adjusted_close"] = np.asarray(
            data["indicators"]["adjclose"][0]["adjclose"], dtype=float)
        if len(set(dates)) != len(dates) or dates != sorted(dates):
            raise ValueError(f"Unsorted or duplicate dates: {symbol}")
        for name, values in arrays.items():
            if len(values) != len(dates) or not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"Invalid {name}: {symbol}")
        adjusted_open = arrays["open"] * arrays["adjusted_close"] / arrays["close"]
        series[symbol] = dict(zip(dates, zip(adjusted_open, arrays["adjusted_close"])))
    first = max(protocol["data_start"], max(min(v) for v in series.values()))
    dates = [d for d in series[protocol["market"]]
             if first <= d <= end]
    for symbol, values in series.items():
        selected = [d for d in values if first <= d <= end]
        if selected != dates:
            raise ValueError(f"Missing or extra sessions relative to SPY: {symbol}")
    prices = np.asarray([[series[s][d] for s in symbols] for d in dates])
    provenance = {
        "provider": manifest["provider"], "download_audit_utc": manifest["audit_utc"],
        "sessions": len(dates), "first_date": dates[0], "last_date": dates[-1],
        "symbols": symbols, "raw_sha256": {s: sources[s]["sha256"] for s in symbols},
        "protocol_sha256": hashlib.sha256((ROOT / "study" / "protocol.json").read_bytes()).hexdigest(),
        "calendar_check": "All symbols match SPY; no independent exchange-calendar audit",
        "price_convention": "Open times adjusted-close/close; a total-return proxy, not an executable price",
    }
    return np.asarray(dates), symbols, prices[:, :, 0], prices[:, :, 1], protocol, provenance


if __name__ == "__main__":
    *_, provenance = load_data()
    print(json.dumps(provenance, indent=2))
