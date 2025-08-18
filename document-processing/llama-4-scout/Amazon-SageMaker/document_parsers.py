import pdfplumber
from docx import Document
import io
from typing import Dict

def parse_document(content: bytes, filename: str, content_type: str) -> Dict[str, str]:
    """
    Parse different document types and extract text content
    """
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf') or 'pdf' in content_type:
        return parse_pdf(content)
    elif filename_lower.endswith('.docx') or 'word' in content_type:
        return parse_docx(content)
    elif filename_lower.endswith(('.txt', '.md')) or 'text' in content_type:
        return parse_text(content)
    else:
        # Fallback to text parsing
        return parse_text(content)

def parse_pdf(content: bytes) -> Dict[str, str]:
    """Extract text from PDF content"""
    try:
        pdf_file = io.BytesIO(content)
        
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        return {
            "text": text.strip(),
            "document_type": "PDF"
        }
    except Exception as e:
        raise Exception(f"Failed to parse PDF: {str(e)}")

def parse_docx(content: bytes) -> Dict[str, str]:
    """Extract text from DOCX content"""
    try:
        doc_file = io.BytesIO(content)
        doc = Document(doc_file)
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return {
            "text": text.strip(),
            "document_type": "DOCX"
        }
    except Exception as e:
        raise Exception(f"Failed to parse DOCX: {str(e)}")

def parse_text(content: bytes) -> Dict[str, str]:
    """Extract text from plain text files"""
    try:
        # Try UTF-8 first, fallback to latin-1
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')
        
        return {
            "text": text.strip(),
            "document_type": "TEXT"
        }
    except Exception as e:
        raise Exception(f"Failed to parse text file: {str(e)}")

def extract_api_info(text: str) -> Dict[str, str]:
    """
    Extract API-specific information from documentation
    This is a simple implementation - could be enhanced with more sophisticated parsing
    """
    api_info = {
        "base_url": "",
        "endpoints": [],
        "authentication": "",
        "examples": []
    }
    
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Look for base URLs
        if 'base url' in line_lower or 'api endpoint' in line_lower:
            if 'http' in line:
                # Extract URL from line
                words = line.split()
                for word in words:
                    if word.startswith('http'):
                        api_info["base_url"] = word
                        break
        
        # Look for authentication info
        if 'auth' in line_lower or 'token' in line_lower or 'key' in line_lower:
            api_info["authentication"] = line.strip()
        
        # Look for code examples
        if 'curl' in line_lower or 'example' in line_lower:
            api_info["examples"].append(line.strip())
    
    return api_info

