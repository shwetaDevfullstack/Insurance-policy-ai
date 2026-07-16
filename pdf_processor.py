import fitz


def extract_pdf_text(uploaded_file):
    text = ""
    page_texts = []

    pdf_bytes = uploaded_file.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    for page_number, page in enumerate(doc):

        page_content = page.get_text()

        page_texts.append({
            "page": page_number + 1,
            "text": page_content
        })

        text += page_content + "\n"

    return {
        "pages": len(doc),
        "text": text,
        "page_texts": page_texts
    }