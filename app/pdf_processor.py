import os
from PyPDF2 import PdfReader

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """
        Extracts all text content from a given PDF document.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        return text