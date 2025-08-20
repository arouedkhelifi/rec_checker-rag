"""
Large File Processing for RAG System

Handles chunking of large codebases while preserving context and structure.
"""

import os
import zipfile
import tempfile
import logging
from typing import List, Dict, Tuple, Any
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class FileProcessor:
    
    def __init__(self, max_chunk_size: int = 50000, overlap: int = 5000):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        # Supported archive extensions
        self.archive_extensions = {
            '.zip', '.rar', '.7z', '.tar', '.tar.gz', '.tgz', 
            '.tar.bz2', '.tbz2', '.tar.xz', '.txz'
        }
    
    def is_archive(self, file_path: str) -> bool:
        """Check if file is a supported archive."""
        return Path(file_path).suffix.lower() in self.archive_extensions
    
    def process_large_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a large file or archive into manageable chunks."""
        try:
            if self.is_archive(file_path):
                return self._process_archive_file(file_path)
            else:
                return self._process_single_file(file_path)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def _process_archive_file(self, archive_path: str) -> List[Dict[str, Any]]:
        """Extract and process archive file contents."""
        chunks = []
        file_ext = Path(archive_path).suffix.lower()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract based on archive type
                if file_ext == '.zip':
                    self._extract_zip(archive_path, temp_dir)
                elif file_ext == '.rar':
                    self._extract_rar(archive_path, temp_dir)
                elif file_ext == '.7z':
                    self._extract_7z(archive_path, temp_dir)
                elif file_ext in ['.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz']:
                    self._extract_tar(archive_path, temp_dir)
                else:
                    # Fallback to patoolib for other formats
                    self._extract_generic(archive_path, temp_dir)
                
                # Find and process code files
                code_files = self._find_code_files(temp_dir)
                logger.info(f"Found {len(code_files)} code files in archive")
                
                for file_path in code_files:
                    try:
                        file_chunks = self._process_single_file(file_path)
                        chunks.extend(file_chunks)
                        
                        # Limit total chunks
                        if len(chunks) >= 50:
                            logger.warning(f"Reached chunk limit, stopping at {len(chunks)} chunks")
                            break
                            
                    except Exception as e:
                        logger.warning(f"Failed to process {file_path}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to extract archive {archive_path}: {e}")
                raise Exception(f"Failed to extract archive: {e}")
        
        return chunks
    
    def _extract_zip(self, archive_path: str, extract_dir: str):
        """Extract ZIP file."""
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Test ZIP integrity
            bad_file = zip_ref.testzip()
            if bad_file:
                raise zipfile.BadZipFile(f"Corrupted file in ZIP: {bad_file}")
            zip_ref.extractall(extract_dir)
            logger.debug(f"Extracted ZIP to {extract_dir}")
    
    def _extract_rar(self, archive_path: str, extract_dir: str):
        """Extract RAR file."""
        try:
            with rarfile.RarFile(archive_path) as rar_ref:
                rar_ref.extractall(extract_dir)
                logger.debug(f"Extracted RAR to {extract_dir}")
        except rarfile.RarCannotExec:
            # Fallback if unrar tool not available
            raise Exception("RAR extraction requires 'unrar' tool to be installed")
    
    def _extract_7z(self, archive_path: str, extract_dir: str):
        """Extract 7Z file."""
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extractall(path=extract_dir)
            logger.debug(f"Extracted 7Z to {extract_dir}")
    
    def _extract_tar(self, archive_path: str, extract_dir: str):
        """Extract TAR file (including compressed variants)."""
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            # Security check for path traversal
            def is_safe_path(path, base_path):
                return os.path.commonpath([base_path, os.path.abspath(os.path.join(base_path, path))]) == base_path
            
            members = tar_ref.getmembers()
            safe_members = [m for m in members if is_safe_path(m.name, extract_dir)]
            
            tar_ref.extractall(extract_dir, members=safe_members)
            logger.debug(f"Extracted TAR to {extract_dir}")
    
    def _extract_generic(self, archive_path: str, extract_dir: str):
        """Extract using patoolib (supports many formats)."""
        try:
            patoolib.extract_archive(archive_path, outdir=extract_dir)
            logger.debug(f"Extracted archive using patoolib to {extract_dir}")
        except Exception as e:
            raise Exception(f"Failed to extract with patoolib: {e}")
        
    def _process_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single code file into chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return []
        
        if len(content) <= self.max_chunk_size:
            return [{
                'content': content,
                'filename': os.path.basename(file_path),
                'filepath': file_path,
                'chunk_index': 0,
                'total_chunks': 1,
                'language': self._detect_language(file_path),
                'size': len(content)
            }]
        
        # Split into intelligent chunks
        chunks = self._intelligent_chunk_split(content, file_path)
        return chunks
    
    def _intelligent_chunk_split(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Split code into intelligent chunks preserving structure."""
        chunks = []
        filename = os.path.basename(file_path)
        language = self._detect_language(file_path)
        
        # Try to split by functions/classes first
        structural_chunks = self._split_by_structure(content, language)
        
        if structural_chunks:
            # Use structural splitting
            for i, chunk_content in enumerate(structural_chunks):
                if len(chunk_content.strip()) > 100:  # Skip tiny chunks
                    chunks.append({
                        'content': chunk_content,
                        'filename': filename,
                        'filepath': file_path,
                        'chunk_index': i,
                        'total_chunks': len(structural_chunks),
                        'language': language,
                        'size': len(chunk_content),
                        'type': 'structural'
                    })
        else:
            # Fallback to simple chunking
            simple_chunks = self._simple_chunk_split(content)
            for i, chunk_content in enumerate(simple_chunks):
                chunks.append({
                    'content': chunk_content,
                    'filename': filename,
                    'filepath': file_path,
                    'chunk_index': i,
                    'total_chunks': len(simple_chunks),
                    'language': language,
                    'size': len(chunk_content),
                    'type': 'simple'
                })
        
        return chunks
    
    def _split_by_structure(self, content: str, language: str) -> List[str]:
        """Split code by structural elements (functions, classes, etc.)."""
        chunks = []
        
        if language == 'python':
            # Split by class and function definitions
            patterns = [
                r'^class\s+\w+.*?(?=^class\s+|\Z)',  # Classes
                r'^def\s+\w+.*?(?=^def\s+|^class\s+|\Z)'  # Functions
            ]
        elif language in ['javascript', 'typescript']:
            patterns = [
                r'^class\s+\w+.*?(?=^class\s+|\Z)',
                r'^function\s+\w+.*?(?=^function\s+|^class\s+|\Z)',
                r'^const\s+\w+\s*=\s*.*?(?=^const\s+|^function\s+|^class\s+|\Z)'
            ]
        elif language == 'java':
            patterns = [
                r'^public\s+class\s+\w+.*?(?=^public\s+class\s+|\Z)',
                r'^public\s+\w+\s+\w+\s*\(.*?(?=^public\s+|\Z)'
            ]
        else:
            return []  # No structural splitting for unknown languages
        
        current_pos = 0
        lines = content.split('\n')
        current_chunk = []
        
        for line_num, line in enumerate(lines):
            current_chunk.append(line)
            
            # Check if we hit a structural boundary
            for pattern in patterns:
                if re.match(pattern, line.strip(), re.MULTILINE):
                    if len(current_chunk) > 1:  # Don't create chunk with just the new function
                        chunk_content = '\n'.join(current_chunk[:-1])
                        if len(chunk_content.strip()) > 200:
                            chunks.append(chunk_content)
                        current_chunk = [line]  # Start new chunk with current line
                    break
            
            # Also split if chunk gets too large
            if len('\n'.join(current_chunk)) > self.max_chunk_size:
                chunk_content = '\n'.join(current_chunk)
                chunks.append(chunk_content)
                current_chunk = []
        
        # Add remaining content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            if len(chunk_content.strip()) > 200:
                chunks.append(chunk_content)
        
        return chunks
    
    def _simple_chunk_split(self, content: str) -> List[str]:
        """Simple overlapping chunk split."""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.max_chunk_size
            
            if end >= len(content):
                chunks.append(content[start:])
                break
            
            # Try to break at a good point (line break)
            break_point = content.rfind('\n', start, end)
            if break_point == -1 or break_point <= start:
                break_point = end
            
            chunks.append(content[start:break_point])
            start = break_point - self.overlap  # Overlap
            
            if start < 0:
                start = break_point
        
        return chunks
    
    def _find_code_files(self, directory: str) -> List[str]:
        """Find all code files in directory."""
        code_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt',
            '.scala', '.r', '.m', '.sh', '.bat', '.ps1'
        }
        
        code_files = []
        for root, dirs, files in os.walk(directory):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in {
                '.git', '.svn', '.hg', '__pycache__', 'node_modules',
                '.vscode', '.idea', 'build', 'dist', 'target'
            }]
            
            for file in files:
                if Path(file).suffix.lower() in code_extensions:
                    code_files.append(os.path.join(root, file))
        
        return code_files[:100]  # Limit to first 100 files
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        return language_map.get(ext, 'unknown')

# Global instance
file_processor = FileProcessor()