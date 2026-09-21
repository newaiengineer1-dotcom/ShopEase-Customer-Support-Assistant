"""Support tools over a mock back-end. Identity (customer_id) is injected by the app; the LLM can never choose it."""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .security import redact_pii

ORDER_RE = re.compile(r"SE-\d{4,8}")
SKU_RE = re.compile(r"SKU-[A-Z0-9-]{2,24}")
REASONS = frozenset({"damaged", "wrong_item", "not_as_described", "no_longer_needed", "other"})
DEFECT_REASONS = frozenset({"damaged", "wrong_item", "not_as_described"})

# args: {name: (rule, required)}   rule = compiled regex | frozenset (enum) | None (free text)
TOOL_SPECS = {
    "list_orders": {"write": False, "args": {}, "doc": "list the signed-in customer's recent orders"},
    "get_order_status": {"write": False, "args": {"order_id": (ORDER_RE, True)},
                         "doc": "status, carrier, tracking and delivery estimate of one order"},
    "check_return_eligibility": {"write": False, "args": {"order_id": (ORDER_RE, True), "sku": (SKU_RE, True), "reason": (REASONS, False)},
                                 "doc": "is an item returnable, with deadline and item total; use when the customer asks WHETHER they can return it "
                                        "(reason: damaged|wrong_item|not_as_described|no_longer_needed|other)"},
    "create_return_request": {"write": True, "args": {"order_id": (ORDER_RE, True), "sku": (SKU_RE, True), "reason": (REASONS, True)},
                              "doc": "start a return; the system checks eligibility first and runs it only after the customer presses Confirm"},
    "escalate_to_human": {"write": True, "args": {"summary": (None, True)},
                          "doc": "open a ticket for a human agent (runs only after the customer presses Confirm)"},
}


def validate_args(spec_args, args):
    """Whitelist validation: unknown keys (e.g. a customer_id invented by the LLM) are dropped."""
    clean = {}
    for key, (rule, required) in spec_args.items():
        val = args.get(key)
        if val is None or str(val).strip() == "":
            if required:
                raise ValueError(f"missing '{key}'")
            continue
        val = str(val).strip()
        if isinstance(rule, re.Pattern):
            val = val.upper()
            if not rule.fullmatch(val):
                raise ValueError(f"invalid '{key}'")
        elif isinstance(rule, frozenset):
            val = val.lower()
            if val not in rule:
                raise ValueError(f"invalid '{key}'")
        else:
            val = val[:300]
        clean[key] = val
    return clean


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ToolResult:
    name: str
    args: dict
    ok: bool
    data: dict = field(default_factory=dict)
    error: str | None = None
    label: str = ""
    ts: str = field(default_factory=_now)

    def evidence(self):
        return f"{self.name}({json.dumps(self.args, sort_keys=True)}) -> {json.dumps(self.data, default=str)}"

    def brief(self):
        return {"tool": self.name, "args": self.args, "ok": self.ok, "error": self.error}


class SupportTools:
    def __init__(self, cfg, today=None, storage_dir=None):
        p, data = cfg.paths, Path(cfg.paths.data_dir)
        self._customers = {c["customer_id"]: c for c in json.loads((data / "customers.json").read_text(encoding="utf-8"))}
        self._orders = {o["order_id"]: o for o in json.loads((data / "orders.json").read_text(encoding="utf-8"))}
        self.rules = json.loads(Path(p.rules).read_text(encoding="utf-8"))
        self._today = today
        self._store = Path(storage_dir or p.storage_dir)
        self._store.mkdir(parents=True, exist_ok=True)
        self._returned = set()
        f = self._store / "returns.jsonl"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._returned.add((r["order_id"], r["sku"]))

    @property
    def today(self):
        if self._today:
            return self._today
        env = os.getenv("APP_TODAY")
        return date.fromisoformat(env) if env else date.today()

    def customers(self):
        return [{"customer_id": c["customer_id"], "name": c["name"], "email": c["email"]} for c in self._customers.values()]

    # ---- data access (dates are computed relative to "today" so the demo never goes stale)
    def _materialize(self, o):
        t, d = self.today, dict(o)
        d["placed_date"] = (t - timedelta(days=o["placed_days_ago"])).isoformat()
        d["delivered_date"] = (t - timedelta(days=o["delivered_days_ago"])).isoformat() if o.get("delivered_days_ago") is not None else None
        d["estimated_delivery"] = (t + timedelta(days=o["eta_in_days"])).isoformat() if o.get("eta_in_days") is not None else None
        d["total"] = round(sum(i["unit_price"] * i["qty"] for i in o["items"]), 2)
        return d

    def _order(self, cid, order_id):
        o = self._orders.get(order_id)
        return self._materialize(o) if o and o["customer_id"] == cid else None   # same answer for "missing" and "someone else's"

    def _own_orders(self, cid):
        rows = (self._materialize(o) for o in self._orders.values() if o["customer_id"] == cid)
        return sorted(rows, key=lambda o: o["placed_date"], reverse=True)

    def customer_context(self, cid):
        """Compact context for the planner: the signed-in customer's own order ids only (data minimisation)."""
        if not cid:
            return "signed_in=false (order tools unavailable; only general policy questions can be answered)"
        lines = [f"signed_in=true; today={self.today.isoformat()}", "Orders, newest first:"]
        for o in self._own_orders(cid):
            items = "; ".join(f"{i['sku']} {i['name']}" for i in o["items"])
            lines.append(f"- {o['order_id']} placed {o['placed_date']}, status {o['status']}, items: {items}")
        return "\n".join(lines)

    # ---- single entry point used by the agent
    def run(self, name, customer_id, args, confirmed=False):
        spec = TOOL_SPECS.get(name)
        if spec is None:
            return ToolResult(name, {}, False, error="unknown_tool")
        try:
            clean = validate_args(spec["args"], args if isinstance(args, dict) else {})
        except ValueError as e:
            return ToolResult(name, {}, False, error=f"invalid_arguments: {e}")
        if not customer_id:
            return ToolResult(name, clean, False, error="not_signed_in")
        if spec["write"] and not confirmed:
            return ToolResult(name, clean, False, error="confirmation_required")
        data = getattr(self, "_" + name)(customer_id, **clean)
        if not isinstance(data, dict) or len(json.dumps(data, default=str)) > 6000:      # output validation
            return ToolResult(name, clean, False, error="invalid_tool_output")
        ids = ([data["order_id"]] if "order_id" in data else []) + [o["order_id"] for o in data.get("orders", [])]
        if any(self._orders.get(i, {}).get("customer_id") != customer_id for i in ids):  # defence in depth
            return ToolResult(name, clean, False, error="ownership_violation")
        err = data.get("error")
        return ToolResult(name, clean, err is None, data, err)

    # ---- tools
    def _list_orders(self, cid):
        return {"as_of": self.today.isoformat(), "orders": [
            {"order_id": o["order_id"], "status": o["status"], "placed_date": o["placed_date"], "order_total": o["total"],
             "items": [f"{i['name']} (x{i['qty']})" for i in o["items"]]} for o in self._own_orders(cid)]}

    def _get_order_status(self, cid, order_id):
        o = self._order(cid, order_id)
        if not o:
            return {"error": "order_not_found", "message": "Could not find this order on your account."}
        return {"as_of": self.today.isoformat(), "order_id": o["order_id"], "status": o["status"], "placed_date": o["placed_date"],
                "estimated_delivery": o["estimated_delivery"], "delivered_date": o["delivered_date"], "carrier": o["carrier"],
                "tracking_number": o["tracking_number"], "order_total": o["total"],
                "items": [{"sku": i["sku"], "name": i["name"], "quantity": i["qty"]} for i in o["items"]]}

    def _check_return_eligibility(self, cid, order_id, sku, reason="no_longer_needed"):
        o = self._order(cid, order_id)
        if not o:
            return {"error": "order_not_found", "message": "Could not find this order on your account."}
        item = next((i for i in o["items"] if i["sku"] == sku), None)
        if not item:
            return {"error": "item_not_in_order", "message": "This item is not part of that order."}
        r = self.rules
        out = {"as_of": self.today.isoformat(), "order_id": order_id, "sku": sku, "item": item["name"], "reason": reason,
               "item_total": round(item["unit_price"] * item["qty"], 2)}
        if o["status"] != "delivered":
            return {**out, "eligible": False, "reason_code": "not_delivered", "order_status": o["status"]}
        if item["category"] in r["non_returnable_categories"]:
            return {**out, "eligible": False, "reason_code": "non_returnable_category", "category": item["category"]}
        if (order_id, sku) in self._returned:
            return {**out, "eligible": False, "reason_code": "already_requested"}
        window = r["defect_window_days"] if reason in DEFECT_REASONS else \
            r["return_window_days"].get(item["category"], r["return_window_days"]["default"])
        delivered = date.fromisoformat(o["delivered_date"])
        used = (self.today - delivered).days
        deadline = delivered + timedelta(days=window)
        ok = used <= window
        return {**out, "eligible": ok, "reason_code": "within_window" if ok else "window_expired", "window_days": window,
                "delivered_date": o["delivered_date"], "days_since_delivery": used, "return_deadline": deadline.isoformat(),
                "days_remaining": max((deadline - self.today).days, 0)}

    def _create_return_request(self, cid, order_id, sku, reason):
        el = self._check_return_eligibility(cid, order_id, sku, reason)      # re-checked at execution time
        if el.get("error"):
            return el
        if not el["eligible"]:
            return {"error": "not_eligible", "message": "This item is not eligible for return.", "reason_code": el["reason_code"]}
        rma = "RMA-" + uuid.uuid4().hex[:8].upper()
        self._append("returns.jsonl", {"rma_id": rma, "customer_id": cid, "order_id": order_id, "sku": sku, "reason": reason, "ts": _now()})
        self._returned.add((order_id, sku))
        return {"rma_id": rma, "order_id": order_id, "sku": sku, "item": el["item"], "reason": reason, "status": "created"}

    def _escalate_to_human(self, cid, summary):
        tid = "TCK-" + uuid.uuid4().hex[:8].upper()
        self._append("tickets.jsonl", {"ticket_id": tid, "customer_id": cid, "summary": redact_pii(summary), "ts": _now()})
        return {"ticket_id": tid, "status": "open", "response_time_hours": self.rules["human_support_response_hours"]}

    def _append(self, name, rec):
        with open(self._store / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
