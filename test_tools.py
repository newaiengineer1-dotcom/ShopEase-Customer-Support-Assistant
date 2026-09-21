def test_owner_only(tools):
    assert tools.run("get_order_status", "C-1001", {"order_id": "SE-1002"}).ok
    r = tools.run("get_order_status", "C-1001", {"order_id": "SE-2001"})      # belongs to C-1002
    assert not r.ok and r.error == "order_not_found"


def test_llm_supplied_customer_id_is_ignored(tools):
    r = tools.run("list_orders", "C-1001", {"customer_id": "C-1002"})
    assert {o["order_id"] for o in r.data["orders"]} == {"SE-1001", "SE-1002", "SE-1003"}


def test_bad_arguments_and_unknown_tools_rejected(tools):
    assert tools.run("get_order_status", "C-1001", {"order_id": "1002; DROP TABLE"}).error.startswith("invalid_arguments")
    assert tools.run("delete_everything", "C-1001", {}).error == "unknown_tool"


def test_guest_cannot_use_tools(tools):
    assert tools.run("list_orders", None, {}).error == "not_signed_in"


def test_write_tools_need_confirmation_and_are_idempotent(tools):
    args = {"order_id": "SE-1001", "sku": "SKU-EARBUDS-01", "reason": "damaged"}
    assert tools.run("create_return_request", "C-1001", args).error == "confirmation_required"
    ok = tools.run("create_return_request", "C-1001", args, confirmed=True)
    assert ok.ok and ok.data["rma_id"].startswith("RMA-")
    again = tools.run("check_return_eligibility", "C-1001", {"order_id": "SE-1001", "sku": "SKU-EARBUDS-01"})
    assert again.data["reason_code"] == "already_requested"


def test_return_rules(tools):
    def elig(cid, oid, sku, reason=None):
        a = {"order_id": oid, "sku": sku, **({"reason": reason} if reason else {})}
        return tools.run("check_return_eligibility", cid, a).data

    assert elig("C-1001", "SE-1001", "SKU-EARBUDS-01")["eligible"]                           # electronics, 5 of 15 days
    assert elig("C-1003", "SE-3001", "SKU-TABLET-05")["reason_code"] == "window_expired"     # electronics, 20 of 15 days
    assert elig("C-1003", "SE-3001", "SKU-TABLET-05", "damaged")["eligible"]                 # defect window is 30 days
    assert elig("C-1002", "SE-2001", "SKU-JACKET-11")["reason_code"] == "window_expired"     # 40 of 30 days
    assert elig("C-1002", "SE-2002", "SKU-GC-25")["reason_code"] == "non_returnable_category"
    assert elig("C-1002", "SE-2002", "SKU-LAMP-09")["eligible"]
    assert elig("C-1001", "SE-1002", "SKU-MUG-02")["reason_code"] == "not_delivered"


def test_rules_match_the_knowledge_base_text(cfg, tools):
    text = (cfg.paths.kb_dir / "returns_and_refunds_policy.md").read_text(encoding="utf-8")
    assert f"{tools.rules['return_window_days']['default']} days" in text
    assert f"{tools.rules['return_window_days']['electronics']} days" in text
