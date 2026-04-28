"""Estrae il testo da GRUCNN.pdf in posturalInstability/GRUCNN_text.txt

Uso: esegui dallo workspace root.
"""
from pathlib import Path
from PyPDF2 import PdfReader


def extract_pdf_text(pdf_path: Path, out_path: Path) -> None:
    reader = PdfReader(str(pdf_path))
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")

    out_path.write_text("\n\n".join(texts), encoding="utf-8")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    pdf = repo_root / "GRUCNN.pdf"
    out = repo_root / "GRUCNN_text.txt"
    if not pdf.exists():
        print(f"PDF non trovato: {pdf}")
    else:
        extract_pdf_text(pdf, out)
        print(f"Testo estratto in: {out}")
