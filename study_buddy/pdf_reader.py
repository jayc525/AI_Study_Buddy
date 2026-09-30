from io import BytesIO

from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a Streamlit uploaded PDF file."""
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages)
