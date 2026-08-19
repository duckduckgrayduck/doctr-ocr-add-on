"""
This is an Add-On that uses docTR https://github.com/mindee/doctr to OCR documents for DocumentCloud
"""

import sys
import time

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from documentcloud.addon import AddOn
from documentcloud.exceptions import APIError


def _build_page(page):
    """Convert a docTR page export into a DocumentCloud page dict."""
    text = ""
    positions = []
    for block in page["blocks"]:
        for line in block["lines"]:
            line_text = ""
            for word in line["words"]:
                value = word["value"]
                line_text += value + " "
                if not value:
                    continue
                (x1, y1), (x2, y2) = word["geometry"]
                positions.append(
                    {
                        "text": value,
                        "x1": float(x1),
                        "x2": float(x2),
                        "y1": float(y1),
                        "y2": float(y2),
                    }
                )
            text += line_text.strip() + "\n"
        text += "\n"
    return {
        "page_number": page["page_idx"],
        "text": text,
        "ocr": "doctr",
        "positions": positions,
    }


class Doctr(AddOn):
    """Class definition"""

    def _upload_pages(self, document, pages):
        """PATCH pages to the API in chunks and wait for processing."""
        page_chunk_size = 20  # Max allowed by the API
        for i in range(0, len(pages), page_chunk_size):
            chunk = pages[i : i + page_chunk_size]
            resp = self.client.patch(f"documents/{document.id}/", json={"pages": chunk})
            resp.raise_for_status()
            while True:
                document_ref = self.client.documents.get(document.id)
                time.sleep(15)
                if document_ref.status == "success":
                    break

    def _tag_document(self, document, max_retries=5, retry_delay=60):
        """Tag the document with the OCR engine, retrying on API errors."""
        retries = 0
        while retries < max_retries:
            try:
                print("Tagging document...")
                self.client.patch(
                    f"documents/{document.id}/",
                    json={"data": {"ocr_engine": ["azure"]}},
                )
                print("Finished tagging document")
                return
            except APIError as exc:
                print(f"Error tagging document. {exc}. Retrying...")
                retries += 1
                time.sleep(retry_delay)
        print(f"Failed to tag document after {max_retries} attempts.")
        self.set_message(
            "Failed to set the OCR tag for this document. "
            "Email info@documentcloud.org to debug."
        )
        sys.exit(1)

    def main(self):
        """The main add-on functionality goes here."""
        self.client.session.headers.update({"User-Agent": "docTR OCR Add-On"})
        to_tag = self.data.get("to_tag", False)

        if self.get_document_count() is None:
            self.set_message("Please select at least one document.")
            return

        model = ocr_predictor(
            "db_resnet50",
            "crnn_vgg16_bn",
            pretrained=True,
            assume_straight_pages=False,
            export_as_straight_boxes=True,
        )

        for document in self.get_documents():
            pdf_name = f"{document.id}.pdf"
            with open(pdf_name, "wb") as pdf:
                pdf.write(document.pdf)
            doc = DocumentFile.from_pdf(pdf_name)
            result = model(doc)
            json_export = result.export()
            pages = [_build_page(p) for p in json_export["pages"]]
            self._upload_pages(document, pages)
            if to_tag:
                self._tag_document(document)


if __name__ == "__main__":
    Doctr().main()
