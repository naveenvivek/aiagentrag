"""
Document processing utilities for various file formats.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import requests

from src.utils.logging import get_logger

logger = get_logger("documents.processor")


class DocumentProcessor:
    """Handles processing of various document formats."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.supported_formats = {'.txt', '.pdf', '.docx', '.html', '.md'}
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a file and extract its content.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary containing file metadata and content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")
        
        logger.info(f"Processing file: {file_path}")
        
        content = ""
        if extension == '.txt' or extension == '.md':
            content = self._process_text_file(file_path)
        elif extension == '.pdf':
            content = self._process_pdf_file(file_path)
        elif extension == '.docx':
            content = self._process_docx_file(file_path)
        elif extension == '.html':
            content = self._process_html_file(file_path)
        
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'extension': extension,
            'content': content,
            'size': file_path.stat().st_size,
            'modified_time': file_path.stat().st_mtime
        }
    
    def _process_text_file(self, file_path: Path) -> str:
        """Process a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def _process_pdf_file(self, file_path: Path) -> str:
        """Process a PDF file."""
        content = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        content += page.extract_text() + "\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {e}")
        except Exception as e:
            logger.error(f"Failed to process PDF file {file_path}: {e}")
            raise
        
        return content
    
    def _process_docx_file(self, file_path: Path) -> str:
        """Process a DOCX file."""
        try:
            doc = Document(file_path)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content
        except Exception as e:
            logger.error(f"Failed to process DOCX file {file_path}: {e}")
            raise
    
    def _process_html_file(self, file_path: Path) -> str:
        """Process an HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"Failed to process HTML file {file_path}: {e}")
            raise
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to the directory to process
            
        Returns:
            List of processed document dictionaries
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        documents = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self.process_file(file_path)
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}")
        
        logger.info(f"Processed {len(documents)} documents from {directory_path}")
        return documents
    
    def process_url(self, url: str) -> Dict[str, Any]:
        """
        Process content from a URL.
        
        Args:
            url: URL to process
            
        Returns:
            Dictionary containing URL metadata and content
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.get_text()
            
            return {
                'url': url,
                'title': soup.title.string if soup.title else 'No title',
                'content': content,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', '')
            }
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {e}")
            raise
