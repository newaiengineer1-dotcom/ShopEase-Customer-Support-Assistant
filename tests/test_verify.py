from src.retrieval import Hit
from src.tools import ToolResult
from src.verify import verify_answer

CHUNK = {"text": "Most items can be returned within 30 days of delivery for a refund to the original payment method. "
                 "Electronics can be returned within 15 days."}
HITS = [Hit("S1", CHUNK, 0.9, 1.0, 0.03)]
TOOL = ToolResult("get_order_status", {"order_id": "SE-1002"}, True,
                  {"order_id": "SE-1002", "status": "shipped", "estimated_delivery": "2026-09-23"}, label="T1")


def check(answer, results=()):
    return verify_answer(answer, HITS, list(results))


def test_supported_answer_passes():
    r = check("Most items can be returned within 30 days of delivery [S1]. Would you like help starting a return?")
    assert r["passed"], r["issues"]


def test_invented_number_fails():
    assert not check("Most items can be returned within 45 days of delivery [S1].")["passed"]


def test_uncited_statement_fails():
    assert not check("Returns are always free for everyone.")["passed"]


def test_unknown_label_fails():
    assert not check("Most items can be returned within 30 days of delivery [S9].")["passed"]


def test_completed_action_claim_fails():
    assert not check("I have created your return request [S1].")["passed"]


def test_tool_evidence_supports_order_facts():
    r = check("Your order SE-1002 has shipped and should arrive on 2026-09-23 [T1].", [TOOL])
    assert r["passed"], r["issues"]
