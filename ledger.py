"""
ledger.py : Phase 1 stock ledger for the Malaria Early Warning product.

There is NO Streamlit in this file, so it can be tested on its own (test_ledger.py).

Three rules this code enforces:
  1. The ledger is APPEND-ONLY. Rows are never edited or deleted.
     A mistake is fixed by a 'correction' row that reverses the wrong row.
  2. A balance can never go below zero.
  3. Days of stock left = current balance / REAL average daily use (no placeholders).
"""
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from math import ceil

import pandas as pd

DB_PATH = "stock_ledger.db"   # separate file, so your old malaria_logs.db is never touched
RED_DAYS = 7                  # under 7 days of stock  -> red
AMBER_DAYS = 14               # 7 to under 14 days     -> amber, 14+ -> green
ALLOW_RESET = True            # set to False before a real pilot

COMMODITIES = [
    ("ACT", "ACT (Artemether-Lumefantrine)", "courses"),
    ("RDT", "RDT test kits", "kits"),
    ("IVART", "IV Artesunate", "vials"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS commodities (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    unit TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facilities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    ward       TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stock_ledger (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_date     TEXT    NOT NULL,                  -- the day the stock moved (YYYY-MM-DD)
    facility_id    INTEGER NOT NULL REFERENCES facilities(id),
    commodity_code TEXT    NOT NULL REFERENCES commodities(code),
    entry_type     TEXT    NOT NULL CHECK (entry_type IN ('opening','received','used','correction')),
    qty_in         INTEGER NOT NULL DEFAULT 0 CHECK (qty_in  >= 0),
    qty_out        INTEGER NOT NULL DEFAULT 0 CHECK (qty_out >= 0),
    period_days    INTEGER NOT NULL DEFAULT 1 CHECK (period_days >= 1),  -- 'used 50 over 3 days' = 3
    balance_after  INTEGER NOT NULL CHECK (balance_after >= 0),          -- never below zero
    entered_by     TEXT    NOT NULL,
    note           TEXT,
    corrects_id    INTEGER REFERENCES stock_ledger(id),  -- set only on correction rows
    entered_at     TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP          -- UTC
);

-- Make the table immutable at database level: any UPDATE or DELETE is rejected.
CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON stock_ledger
BEGIN SELECT RAISE(ABORT, 'stock_ledger is append-only: use a correction entry'); END;

CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON stock_ledger
BEGIN SELECT RAISE(ABORT, 'stock_ledger is append-only: use a correction entry'); END;

CREATE INDEX IF NOT EXISTS idx_ledger_pair ON stock_ledger (facility_id, commodity_code);
"""


# ---------------------------------------------------------------- database basics
def get_conn():
    # isolation_level=None means WE decide when a transaction starts and ends (see transaction()).
    conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def transaction(conn):
    """Everything inside the 'with' block succeeds together or not at all."""
    conn.execute("BEGIN IMMEDIATE")   # locks the DB so two people can't both spend the last stock
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.executemany("INSERT OR IGNORE INTO commodities (code, name, unit) VALUES (?, ?, ?)", COMMODITIES)
    conn.close()


def reset_database():
    """DEV ONLY: wipes the ledger. Blocked when ALLOW_RESET is False."""
    if not ALLOW_RESET:
        raise PermissionError("Reset is disabled (ALLOW_RESET is False).")
    conn = get_conn()
    conn.executescript("DROP TABLE IF EXISTS stock_ledger; DROP TABLE IF EXISTS facilities; DROP TABLE IF EXISTS commodities;")
    conn.close()
    init_db()


def _read(sql, params=()):
    conn = get_conn()
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()


def list_facilities():
    return _read("SELECT id, name, ward FROM facilities ORDER BY name")


def list_commodities():
    return _read("SELECT code, name, unit FROM commodities ORDER BY name")


def add_facility(name, ward=""):
    name = (name or "").strip()
    if not name:
        raise ValueError("Facility / village name is required.")
    conn = get_conn()
    try:
        cur = conn.execute("INSERT INTO facilities (name, ward) VALUES (?, ?)", (name, (ward or "").strip()))
        return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError(f"'{name}' already exists.")
    finally:
        conn.close()


# ---------------------------------------------------------------- writing to the ledger
def _balance(conn, facility_id, code):
    row = conn.execute(
        "SELECT COALESCE(SUM(qty_in) - SUM(qty_out), 0) AS b FROM stock_ledger "
        "WHERE facility_id = ? AND commodity_code = ?", (facility_id, code)).fetchone()
    return int(row["b"])


def record_entry(entry_date, facility_id, commodity_code, entry_type, quantity,
                 entered_by, period_days=1, note=""):
    """Add one row. entry_type is 'opening' (starting stock), 'received' or 'used'.
    Returns the new balance. Raises ValueError with a plain-language message if not allowed."""
    if entry_type not in ("opening", "received", "used"):
        raise ValueError("Entry type must be opening, received or used.")
    quantity = int(quantity)
    if quantity < 1:
        raise ValueError("Quantity must be at least 1.")
    entered_by = (entered_by or "").strip()
    if not entered_by:
        raise ValueError("Enter your name in the sidebar so the entry can be traced to you.")
    if isinstance(entry_date, str):
        entry_date = date.fromisoformat(entry_date)
    if entry_date > date.today():
        raise ValueError("The date cannot be in the future.")
    period_days = int(period_days) if entry_type == "used" else 1
    if period_days < 1:
        raise ValueError("'Covers how many days' must be at least 1.")
    facility_id = int(facility_id)

    conn = get_conn()
    try:
        with transaction(conn):
            if entry_type == "opening":
                already = conn.execute(
                    "SELECT 1 FROM stock_ledger l WHERE facility_id = ? AND commodity_code = ? "
                    "AND entry_type = 'opening' AND NOT EXISTS "
                    "(SELECT 1 FROM stock_ledger x WHERE x.corrects_id = l.id)",
                    (facility_id, commodity_code)).fetchone()
                if already:
                    raise ValueError("Starting stock is already set here. Use 'Stock received' for deliveries, "
                                     "or void the old starting-stock entry first.")
            balance = _balance(conn, facility_id, commodity_code)
            qty_in, qty_out = (quantity, 0) if entry_type in ("opening", "received") else (0, quantity)
            new_balance = balance + qty_in - qty_out
            if new_balance < 0:
                raise ValueError(f"Not enough stock: the balance is {balance} and you tried to use {quantity}.")
            conn.execute(
                "INSERT INTO stock_ledger (entry_date, facility_id, commodity_code, entry_type, qty_in, qty_out, "
                "period_days, balance_after, entered_by, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry_date.isoformat(), facility_id, commodity_code, entry_type, qty_in, qty_out,
                 period_days, new_balance, entered_by, (note or "").strip()))
            return new_balance
    finally:
        conn.close()


def void_entry(entry_id, reason, entered_by):
    """'Edit/void' = add a correction row that reverses the wrong row. Nothing is deleted."""
    reason = (reason or "").strip()
    entered_by = (entered_by or "").strip()
    if not reason:
        raise ValueError("Give a reason for voiding this entry.")
    if not entered_by:
        raise ValueError("Enter your name in the sidebar so the correction can be traced to you.")
    conn = get_conn()
    try:
        with transaction(conn):
            row = conn.execute("SELECT * FROM stock_ledger WHERE id = ?", (int(entry_id),)).fetchone()
            if row is None:
                raise ValueError("Entry not found.")
            if row["entry_type"] == "correction":
                raise ValueError("A correction cannot be voided. Enter a new entry instead.")
            if conn.execute("SELECT 1 FROM stock_ledger WHERE corrects_id = ?", (row["id"],)).fetchone():
                raise ValueError("This entry has already been voided.")
            balance = _balance(conn, row["facility_id"], row["commodity_code"])
            new_balance = balance + row["qty_out"] - row["qty_in"]   # undo the original movement
            if new_balance < 0:
                raise ValueError(f"Cannot void: that stock has already been used (balance would be {new_balance}).")
            conn.execute(
                "INSERT INTO stock_ledger (entry_date, facility_id, commodity_code, entry_type, qty_in, qty_out, "
                "period_days, balance_after, entered_by, note, corrects_id) "
                "VALUES (?, ?, ?, 'correction', ?, ?, 1, ?, ?, ?, ?)",
                (date.today().isoformat(), row["facility_id"], row["commodity_code"],
                 row["qty_out"], row["qty_in"], new_balance, entered_by,
                 f"VOID of #{row['id']}: {reason}", row["id"]))
            return new_balance
    finally:
        conn.close()


# ---------------------------------------------------------------- reading: usage, days left, status
def _avg_daily_use(conn, facility_id, code, window_days, today):
    """Average units used per day over the last `window_days`, from REAL usage entries.
    A 'used 50 over 3 days' entry is spread evenly across those 3 days.
    Voided entries are ignored. Returns (average or None, days_of_data)."""
    rows = conn.execute(
        "SELECT entry_date, qty_out, period_days FROM stock_ledger l "
        "WHERE facility_id = ? AND commodity_code = ? AND entry_type = 'used' "
        "AND NOT EXISTS (SELECT 1 FROM stock_ledger x WHERE x.corrects_id = l.id)",
        (facility_id, code)).fetchall()
    if not rows:
        return None, 0
    win_start = today - timedelta(days=window_days - 1)
    used, first_start = 0.0, None
    for r in rows:
        end = date.fromisoformat(r["entry_date"])
        start = end - timedelta(days=r["period_days"] - 1)
        first_start = start if first_start is None else min(first_start, start)
        ov_start, ov_end = max(start, win_start), min(end, today)
        if ov_end >= ov_start:   # only the part of the entry that falls inside the window counts
            used += r["qty_out"] * ((ov_end - ov_start).days + 1) / r["period_days"]
    # A new facility with 3 days of history is averaged over 3 days, not 14.
    eff_days = (today - max(win_start, first_start)).days + 1
    return used / eff_days, eff_days


def status_for(balance, avg):
    """Returns (status label, days left). Colours react only to real numbers."""
    if balance <= 0:
        return "🔴 Stock-out", 0.0
    if not avg:
        return "⚪ No usage data", None
    days = balance / avg
    if days < RED_DAYS:
        return "🔴 Red", days
    if days < AMBER_DAYS:
        return "🟠 Amber", days
    return "🟢 Green", days


def _derived(balance, avg, target_days, today):
    status, days = status_for(balance, avg)
    stockout = (today + timedelta(days=int(days))).isoformat() if days is not None else ""
    order = max(0, ceil(avg * target_days - balance)) if avg else 0
    return status, (round(days, 1) if days is not None else None), stockout, order


def _pair_rows(window_days, today):
    conn = get_conn()
    try:
        pairs = conn.execute(
            "SELECT l.facility_id, f.name AS facility, l.commodity_code AS code, c.name AS commodity, "
            "COALESCE(SUM(l.qty_in) - SUM(l.qty_out), 0) AS balance "
            "FROM stock_ledger l JOIN facilities f ON f.id = l.facility_id "
            "JOIN commodities c ON c.code = l.commodity_code "
            "GROUP BY l.facility_id, l.commodity_code ORDER BY f.name, c.name").fetchall()
        out = []
        for p in pairs:
            avg, eff = _avg_daily_use(conn, p["facility_id"], p["code"], window_days, today)
            out.append({"facility": p["facility"], "code": p["code"], "commodity": p["commodity"],
                        "balance": int(p["balance"]), "avg": avg, "eff_days": eff})
        return out
    finally:
        conn.close()


def stock_summary(window_days=14, target_days=30, today=None, uplift=0.0):
    """One row per facility and commodity.
    uplift is a what-if demand surge: 0.5 means 'demand rises by 50%'. Default 0 = real usage only."""
    today = today or date.today()
    rows = []
    for r in _pair_rows(window_days, today):
        avg = r["avg"] * (1 + uplift) if r["avg"] else r["avg"]
        status, days, stockout, order = _derived(r["balance"], avg, target_days, today)
        rows.append({"Facility": r["facility"], "Commodity": r["commodity"], "Balance": r["balance"],
                     "Avg daily use": round(avg, 1) if avg else None, "Days of data": r["eff_days"],
                     "Days left": days, "Stock-out date": stockout, "Suggested order": order, "Status": status})
    return pd.DataFrame(rows, columns=["Facility", "Commodity", "Balance", "Avg daily use", "Days of data",
                                       "Days left", "Stock-out date", "Suggested order", "Status"])


def commodity_totals(window_days=14, target_days=30, today=None, uplift=0.0):
    """One row per commodity, added up across all facilities (uplift = optional what-if surge)."""
    today = today or date.today()
    totals = {}
    for r in _pair_rows(window_days, today):
        t = totals.setdefault(r["commodity"], {"balance": 0, "avg": 0.0})
        t["balance"] += r["balance"]
        t["avg"] += (r["avg"] or 0.0) * (1 + uplift)
    rows = []
    for name, t in totals.items():
        status, days, stockout, order = _derived(t["balance"], t["avg"], target_days, today)
        rows.append({"Commodity": name, "Balance": t["balance"], "Avg daily use": round(t["avg"], 1),
                     "Days left": days, "Stock-out date": stockout, "Suggested order": order, "Status": status})
    return pd.DataFrame(rows, columns=["Commodity", "Balance", "Avg daily use", "Days left",
                                       "Stock-out date", "Suggested order", "Status"])


def history(facility_id=None, commodity_code=None, limit=500):
    """The full audit trail, newest first."""
    where, params = [], []
    if facility_id is not None:
        where.append("l.facility_id = ?")
        params.append(int(facility_id))
    if commodity_code:
        where.append("l.commodity_code = ?")
        params.append(commodity_code)
    sql = (
        "SELECT l.id, l.entry_date AS date, f.name AS facility, c.name AS commodity, l.entry_type AS type, "
        "l.qty_in AS received, l.qty_out AS used, l.period_days, l.balance_after AS balance, "
        "l.entered_by, l.note, "
        "CASE WHEN l.entry_type = 'correction' THEN 'Correction of #' || l.corrects_id "
        "     WHEN EXISTS (SELECT 1 FROM stock_ledger x WHERE x.corrects_id = l.id) THEN 'Voided' "
        "     ELSE 'Active' END AS status, l.entered_at "
        "FROM stock_ledger l JOIN facilities f ON f.id = l.facility_id "
        "JOIN commodities c ON c.code = l.commodity_code "
        + ("WHERE " + " AND ".join(where) + " " if where else "")
        + "ORDER BY l.id DESC LIMIT ?")
    return _read(sql, params + [int(limit)])


# ---------------------------------------------------------------- demo data (testing only)
DEMO = [  # facility, commodity, starting stock, typical daily use
    ("DEMO Village A", "ACT", 190, 10), ("DEMO Village A", "RDT", 600, 25),
    ("DEMO Village B", "ACT", 500, 8), ("DEMO Village B", "RDT", 900, 20),
    ("DEMO Village C", "ACT", 240, 9), ("DEMO Village C", "IVART", 60, 2),
]


def load_demo_data():
    """Fills the ledger with clearly-labelled DEMO entries so you can see red/amber/green react."""
    if list_facilities()["name"].str.startswith("DEMO").any():
        raise ValueError("Demo data is already loaded.")
    today, ids = date.today(), {}
    for name, code, opening, base in DEMO:
        if name not in ids:
            ids[name] = add_facility(name, "DEMO ward")
        record_entry(today - timedelta(days=14), ids[name], code, "opening", opening, "demo-loader", note="DEMO DATA")
        for i in range(14):
            qty = max(1, base + (i % 3 - 1))
            record_entry(today - timedelta(days=13 - i), ids[name], code, "used", qty, "demo-loader", note="DEMO DATA")
