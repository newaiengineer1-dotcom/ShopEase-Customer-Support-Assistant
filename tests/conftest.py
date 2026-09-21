import pytest

from src.agent import SupportAgent
from src.config import load_settings
from src.embeddings import get_embedder
from src.ingest import build_index
from src.retrieval import KnowledgeBase
from src.tools import SupportTools
from tests.helpers import TODAY


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    c = load_settings()
    tmp = tmp_path_factory.mktemp("storage")
    c.paths.storage_dir, c.paths.index_dir = tmp, tmp / "index"
    c.embedding.provider = "hash"              # offline and deterministic: no model download in tests
    c.retrieval.similarity_threshold = 0.15    # calibrated for the hash embedder only
    c.verify.llm_verifier = False
    return c


@pytest.fixture(scope="session")
def kb(cfg):
    emb = get_embedder(cfg)
    build_index(cfg, emb)
    return KnowledgeBase(cfg, emb)


@pytest.fixture()
def tools(cfg, tmp_path):
    return SupportTools(cfg, today=TODAY, storage_dir=tmp_path)


@pytest.fixture()
def make_agent(cfg, kb, tools):
    return lambda llm: SupportAgent(cfg, kb, llm, tools)
