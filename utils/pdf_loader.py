"""
PDF text extraction utility for financial reports.
Extracts text content from uploaded PDF files.
"""

import os
import PyPDF2
from typing import Optional


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
        
    Raises:
        Exception: If PDF cannot be read or processed
    """
    try:
        text_content = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    text_content.append(text)
        
        # Combine all pages
        full_text = ' '.join(text_content)
        
        # Basic cleaning: remove excessive whitespace
        full_text = ' '.join(full_text.split())
        
        if not full_text.strip():
            raise ValueError("No text could be extracted from the PDF")
            
        return full_text
        
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def validate_pdf(pdf_path: str) -> bool:
    """
    Validate if the file is a readable PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if valid PDF, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            return False
            
        # Check file extension
        if not pdf_path.lower().endswith('.pdf'):
            return False
            
        # Try to read with PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Check if PDF has pages
            if len(reader.pages) == 0:
                return False
        return True
    except Exception as e:
        print(f"PDF validation error: {e}")
        return False
