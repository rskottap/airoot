__all__ = [
    'pdf_to_pages',
    'pdf_to_text',
    'is_pdf',
]
    
def is_pdf(bytes):
    from kern import infer_type
    return infer_type(bytes) == 'pdf'

def pdf_to_pages(file):
    import pypdf
    import assure
    # assure file is seekable to support stdin.
    file = assure.seekable(file)
    reader = pypdf.PdfReader(file)
    pages = [page.extract_text() for page in reader.pages]
    return pages

def pdf_to_text(file):
    pages = pdf_to_pages(file)
    return '\n\n'.join(pages)
