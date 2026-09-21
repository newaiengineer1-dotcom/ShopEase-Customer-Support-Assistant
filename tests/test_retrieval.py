from src.ingest import load_units


def test_index_is_consistent(kb):
    assert kb.problems == [] and kb.info["documents"] >= 9 and kb.info["chunks"] >= 20


def test_txt_headings_are_detected(cfg):
    sections = [s for s, _, _ in load_units(cfg.paths.kb_dir / "payments_and_billing.txt")]
    assert "Accepted Payment Methods" in sections


def test_in_scope_question_finds_returns_policy(kb):
    r = kb.search("How many days do I have to return an item?")
    assert r.sufficient and any(h.chunk["document_id"] == "KB-RET-001" for h in r.hits)


def test_out_of_scope_is_insufficient(kb):
    assert not kb.search("quantum chromodynamics lattice simulation").sufficient


def test_customer_never_sees_internal_documents(kb):
    r = kb.search("supervisor approval limit for refunds", role="customer")
    assert all(h.chunk["access_level"] == "public" for h in r.hits)
    assert r.stats["blocked_permission"] > 0


def test_staff_can_see_internal_documents(kb):
    r = kb.search("supervisor approval limit for refunds", role="staff")
    assert any(h.chunk["document_id"] == "KB-INT-001" for h in r.hits)


def test_superseded_documents_are_never_retrieved(kb):
    r = kb.search("how many days to return an item", role="staff")
    assert all(h.chunk["status"] == "active" for h in r.hits)
    assert all(h.chunk["document_id"] != "KB-RET-000" for h in r.hits)
