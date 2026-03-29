"""
PDF Text Extractor Tool
Extracts text content from all PDF files in a directory using docling
"""

from pathlib import Path
from docling.document_converter import DocumentConverter


def extract_text_from_pdfs(directory_path: str) -> dict:
    """
    Extract text content from all PDF files in a directory using docling.

    Args:
        directory_path: Path to the directory containing PDF files to extract text from

    Returns:
        A dictionary containing:
            - documents: List of extracted document results, each with:
                - file_name: Name of the source PDF file
                - text: The extracted text content
                - page_count: Number of pages processed
                - success: Boolean indicating if extraction was successful
                - error: Error message if extraction failed
            - total_files: Total number of PDF files found
            - successful: Number of successfully processed files
            - failed: Number of failed extractions
    """
    try:
        

        dir_path = Path(directory_path)

        # Validate directory exists
        if not dir_path.exists():
            return {
                "documents": [],
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "error": f"Directory not found: {directory_path}"
            }

        if not dir_path.is_dir():
            return {
                "documents": [],
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "error": f"Path is not a directory: {directory_path}"
            }

        # Find all PDF files in the directory
        pdf_files = list(dir_path.glob("*.pdf"))

        if not pdf_files:
            return {
                "documents": [],
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "error": f"No PDF files found in directory: {directory_path}"
            }

        # Initialize the document converter once for all documents
        converter = DocumentConverter()

        documents = []
        successful_count = 0
        failed_count = 0

        for pdf_path in pdf_files:
            try:
                # Convert the PDF document
                result = converter.convert(str(pdf_path))

                # Extract the main text content as markdown
                text_content = result.document.export_to_markdown()

                # Get page count
                page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 0

                documents.append({
                    "file_name": pdf_path.name,
                    "text": text_content,
                    "page_count": page_count,
                    "success": True,
                    "error": None
                })
                successful_count += 1

            except Exception as e:
                documents.append({
                    "file_name": pdf_path.name,
                    "text": "",
                    "page_count": 0,
                    "success": False,
                    "error": f"Error processing file: {str(e)}"
                })
                failed_count += 1

        return {
            "documents": documents,
            "total_files": len(pdf_files),
            "successful": successful_count,
            "failed": failed_count
        }

    except ImportError:
        return {
            "documents": [],
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "error": "docling is not installed. Install with: pip install docling"
        }
    except Exception as e:
        return {
            "documents": [],
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "error": f"Error initializing document converter: {str(e)}"
        }
