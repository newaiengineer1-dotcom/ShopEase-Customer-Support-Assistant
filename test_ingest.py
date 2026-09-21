import pytest

from src.ingest import load_units, window


def test_window_splits_and_respects_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = window(text, size=40, overlap=10)
    assert len(chunks) > 1 and all(len(c.split()) <= 60 for c in chunks)


def test_pdf_loader_keeps_page_numbers(tmp_path):
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, "Gift card balances never expire. Gift cards cannot be returned.")
    p = tmp_path / "t.pdf"
    pdf.output(str(p))
    units = list(load_units(p))
    assert units and units[0][1] == 1 and "never expire" in units[0][2]


def test_docx_loader_uses_headings(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Seller items", level=2)
    d.add_paragraph("Sellers ship some items directly.")
    p = tmp_path / "t.docx"
    d.save(str(p))
    assert list(load_units(p))[0][0] == "Seller items"
