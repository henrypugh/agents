"""
Filesystem tools for the MCP server.

This module contains tools for interacting with the local filesystem.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from traceloop.sdk.decorators import tool
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("FilesystemTools")

def register_filesystem_tools(mcp: FastMCP) -> None:
    """
    Register all filesystem tools with the MCP server.
    
    Parameters:
    -----------
    mcp : FastMCP
        The MCP server instance
    """
    
    @mcp.tool()
    def get_working_directory() -> Dict[str, Any]:
        """
        Get the current working directory and environment information.
        
        This helps locate where the server is running from and what paths are available.
        
        Returns:
        --------
        Dict
            Dictionary containing working directory and environment information
        """
        try:
            # Get the current working directory
            cwd = os.getcwd()
            logger.info(f"Current working directory: {cwd}")
            
            # Get path to the script directory
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # List top-level directories in cwd
            top_dirs = []
            top_files = []
            
            try:
                for item in os.listdir(cwd):
                    item_path = os.path.join(cwd, item)
                    if os.path.isdir(item_path):
                        top_dirs.append(item)
                    elif os.path.isfile(item_path):
                        top_files.append(item)
            except Exception as e:
                logger.warning(f"Error listing directory contents: {str(e)}")
            
            # Find common code directories
            code_dirs = []
            for common_dir in ['src', 'server', 'app', 'lib', 'pkg', 'scripts']:
                if os.path.isdir(os.path.join(cwd, common_dir)):
                    code_dirs.append(common_dir)
            
            return {
                "working_directory": cwd,
                "script_directory": script_dir,
                "top_level_directories": top_dirs,
                "top_level_files": top_files,
                "common_code_directories": code_dirs,
                "path_separator": os.path.sep,
                "absolute_path_example": os.path.abspath("example.txt"),
                "python_path": sys.path,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error getting working directory: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @mcp.tool()
    def list_directory(path: str = ".") -> Dict[str, Any]:
        """
        List the contents of a directory
        
        Parameters:
        -----------
        path : str
            Path to the directory to list (default: current directory)
            
        Returns:
        --------
        Dict
            Dictionary containing directory information including files and subdirectories
        """
        try:
            # Convert to absolute path for clarity
            abs_path = os.path.abspath(path)
            logger.info(f"Listing directory: {abs_path}")
            
            if not os.path.exists(abs_path):
                return {
                    "error": f"Path does not exist: {abs_path}",
                    "status": "error"
                }
                
            if not os.path.isdir(abs_path):
                return {
                    "error": f"Path is not a directory: {abs_path}",
                    "status": "error"
                }
            
            # Get directory contents
            contents = os.listdir(abs_path)
            
            # Separate files and directories
            files = []
            directories = []
            
            for item in contents:
                item_path = os.path.join(abs_path, item)
                if os.path.isdir(item_path):
                    directories.append(item)
                else:
                    files.append(item)
            
            return {
                "path": abs_path,
                "files": files,
                "directories": directories,
                "count": {
                    "files": len(files),
                    "directories": len(directories),
                    "total": len(contents)
                },
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error listing directory {path}: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @mcp.tool()
    def read_file(file_path: str) -> Dict[str, Any]:
        """
        Read the contents of a file
        
        Parameters:
        -----------
        file_path : str
            Path to the file to read
            
        Returns:
        --------
        Dict
            Dictionary containing file contents and metadata
        """
        try:
            # Convert to absolute path for clarity
            abs_path = os.path.abspath(file_path)
            logger.info(f"Reading file: {abs_path}")
            
            if not os.path.exists(abs_path):
                return {
                    "error": f"File does not exist: {abs_path}",
                    "status": "error"
                }
                
            if not os.path.isfile(abs_path):
                return {
                    "error": f"Path is not a file: {abs_path}",
                    "status": "error"
                }
                
            # Get file extension
            _, ext = os.path.splitext(abs_path)
            
            # Read file contents
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get file metadata
            stat = os.stat(abs_path)
            
            return {
                "path": abs_path,
                "content": content,
                "extension": ext,
                "size": stat.st_size,
                "status": "success"
            }
        except UnicodeDecodeError:
            # Handle binary files
            logger.warning(f"File appears to be binary: {abs_path}")
            return {
                "path": abs_path,
                "error": "Cannot read binary file",
                "is_binary": True,
                "status": "error"
            }
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }
    @mcp.tool()
    def search_files(
        directory: str = ".",
        pattern: str = "*",
        text_search: Optional[str] = None,
        max_depth: int = 5,
        max_files: int = 100,
        include_content: bool = False,
        code_files_only: bool = False
    ) -> Dict[str, Any]:
        """
        Search for files matching a pattern and optionally containing specific text
        
        Parameters:
        -----------
        directory : str
            Directory to search in (default: current directory)
        pattern : str
            File pattern to match (e.g., "*.py" for Python files)
        text_search : Optional[str]
            Text to search for within files (optional)
        max_depth : int
            Maximum directory depth to search (default: 5)
        max_files : int
            Maximum number of files to return (default: 100)
        include_content : bool
            Whether to include matching file contents (default: False)
        code_files_only : bool
            Whether to only search common code file types (default: False)
            
        Returns:
        --------
        Dict
            Dictionary containing search results
        """
        try:
            # Convert to absolute path for clarity
            abs_path = os.path.abspath(directory)
            logger.info(f"Searching in directory: {abs_path} with pattern: {pattern}")
            
            if not os.path.exists(abs_path):
                return {
                    "error": f"Directory does not exist: {abs_path}",
                    "status": "error"
                }
                
            if not os.path.isdir(abs_path):
                return {
                    "error": f"Path is not a directory: {abs_path}",
                    "status": "error"
                }
            
            # Define common code file extensions if code_files_only is True
            code_extensions = []
            if code_files_only:
                code_extensions = [
                    # Python
                    ".py", ".pyi", ".pyx", ".pyd", ".ipynb",
                    # JavaScript/TypeScript
                    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
                    # Web
                    ".html", ".htm", ".css", ".scss", ".sass", ".less",
                    # Configuration
                    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".config",
                    # Documentation
                    ".md", ".rst", ".txt", ".mdx",
                    # Shell
                    ".sh", ".bash", ".zsh", ".fish",
                    # Others
                    ".xml", ".csv", ".env", ".example", ".template",
                    # Special files
                    "Dockerfile", "Makefile", "Jenkinsfile", ".gitignore", 
                    "requirements.txt", "setup.py", "pyproject.toml"
                ]
                logger.info(f"Filtering for code files with extensions: {code_extensions}")
            
            # Search for files
            matched_files = []
            files_processed = 0
            
            base_path = Path(abs_path)
            
            # Get all files matching pattern up to max_depth
            for file_path in base_path.glob(f"{'*/' * max_depth}{pattern}"):
                if files_processed >= max_files:
                    break
                    
                # Skip directories, only process files
                if not file_path.is_file():
                    continue
                    
                # Skip non-code files if code_files_only is True
                if code_files_only:
                    file_ext = file_path.suffix.lower()
                    file_name = file_path.name
                    if file_ext not in code_extensions and file_name not in code_extensions:
                        continue
                    
                # Check if we need to search file contents
                file_matches = True
                file_content = None
                matching_lines = []
                
                if text_search:
                    file_matches = False
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            line_number = 0
                            for line in f:
                                line_number += 1
                                if text_search.lower() in line.lower():
                                    file_matches = True
                                    matching_lines.append({
                                        "line": line_number,
                                        "content": line.strip()
                                    })
                            
                            # Optionally get whole file content
                            if file_matches and include_content:
                                f.seek(0)
                                file_content = f.read()
                    except UnicodeDecodeError:
                        # Skip binary files
                        continue
                    except Exception as e:
                        # Skip files with read errors
                        logger.warning(f"Error reading file {file_path}: {str(e)}")
                        continue
                
                # Add file if it matches all criteria
                if file_matches:
                    file_entry = {
                        "path": str(file_path),
                        "relative_path": str(file_path.relative_to(base_path)),
                        "size": file_path.stat().st_size
                    }
                    
                    if matching_lines:
                        file_entry["matching_lines"] = matching_lines
                        
                    if include_content and file_content:
                        file_entry["content"] = file_content
                        
                    matched_files.append(file_entry)
                    files_processed += 1
            
            return {
                "directory": abs_path,
                "pattern": pattern,
                "text_search": text_search,
                "code_files_only": code_files_only,
                "matched_files": matched_files,
                "count": len(matched_files),
                "max_reached": files_processed >= max_files,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error searching files: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @mcp.tool()
    def analyze_project_structure(
        root_dir: str = ".", 
        max_depth: int = 3,
        include_files: bool = True,
        ignore_dirs: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the structure of a project directory
        
        Parameters:
        -----------
        root_dir : str
            Root directory of the project to analyze (default: current directory)
        max_depth : int
            Maximum directory depth to traverse (default: 3)
        include_files : bool
            Whether to include files in the analysis (default: True)
        ignore_dirs : List[str]
            List of directory names to ignore (default: ['.git', '__pycache__', '.venv', 'venv', 'node_modules'])
            
        Returns:
        --------
        Dict
            Dictionary containing project structure information
        """
        try:
            # Set default ignore directories
            if ignore_dirs is None:
                ignore_dirs = ['.git', '__pycache__', '.venv', 'venv', 'node_modules', 
                               '.idea', '.vscode', '__pycache__', '.pytest_cache']
            
            # Convert to absolute path
            abs_path = os.path.abspath(root_dir)
            logger.info(f"Analyzing project structure at: {abs_path}")
            
            if not os.path.exists(abs_path):
                return {
                    "error": f"Directory does not exist: {abs_path}",
                    "status": "error"
                }
                
            if not os.path.isdir(abs_path):
                return {
                    "error": f"Path is not a directory: {abs_path}",
                    "status": "error"
                }
            
            # Analyze directory structure
            structure = {}
            file_count = 0
            dir_count = 0
            
            # Helper function for recursive directory analysis
            def analyze_dir(dir_path, current_depth, parent_dict):
                nonlocal file_count, dir_count
                
                if current_depth > max_depth:
                    return
                
                # List directory contents
                try:
                    items = os.listdir(dir_path)
                except PermissionError:
                    parent_dict["__error__"] = "Permission denied"
                    return
                except Exception as e:
                    parent_dict["__error__"] = str(e)
                    return
                
                # Process items
                for item in sorted(items):
                    item_path = os.path.join(dir_path, item)
                    
                    # Skip ignored directories
                    if os.path.isdir(item_path) and item in ignore_dirs:
                        continue
                    
                    # Process directories
                    if os.path.isdir(item_path):
                        dir_count += 1
                        parent_dict[item] = {}
                        analyze_dir(item_path, current_depth + 1, parent_dict[item])
                    
                    # Process files if include_files is True
                    elif include_files and os.path.isfile(item_path):
                        file_count += 1
                        parent_dict[item] = "file"
            
            # Start analysis
            analyze_dir(abs_path, 1, structure)
            
            # Detect project type
            project_type = "unknown"
            project_indicators = {
                "python": ["setup.py", "pyproject.toml", "requirements.txt", "poetry.lock", "Pipfile"],
                "node": ["package.json", "node_modules", "yarn.lock", "package-lock.json"],
                "web": ["index.html", "webpack.config.js"],
                "golang": ["go.mod", "go.sum"],
                "rust": ["Cargo.toml", "Cargo.lock"],
                "java": ["pom.xml", "build.gradle", "gradlew"],
                "docker": ["Dockerfile", "docker-compose.yml"]
            }
            
            detected_types = []
            root_files = os.listdir(abs_path)
            for ptype, indicators in project_indicators.items():
                for indicator in indicators:
                    if indicator in root_files:
                        detected_types.append(ptype)
                        break
            
            if detected_types:
                project_type = ", ".join(detected_types)
            
            # Look for key project components
            has_readme = any(f.lower() == "readme.md" for f in root_files)
            has_license = any(f.lower() == "license" or f.lower() == "license.md" for f in root_files)
            has_tests = "tests" in root_files or "test" in root_files
            has_docs = "docs" in root_files or "documentation" in root_files
            
            return {
                "root_directory": abs_path,
                "structure": structure,
                "file_count": file_count,
                "directory_count": dir_count,
                "project_type": project_type,
                "has_readme": has_readme,
                "has_license": has_license,
                "has_tests": has_tests,
                "has_docs": has_docs,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing project structure: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @mcp.tool()
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a file
        
        Parameters:
        -----------
        file_path : str
            Path to the file
            
        Returns:
        --------
        Dict
            Dictionary containing file metadata
        """
        try:
            # Convert to absolute path for clarity
            abs_path = os.path.abspath(file_path)
            logger.info(f"Getting info for file: {abs_path}")
            
            if not os.path.exists(abs_path):
                return {
                    "error": f"File does not exist: {abs_path}",
                    "status": "error"
                }
                
            # Get file information
            stat = os.stat(abs_path)
            path_obj = Path(abs_path)
            
            return {
                "path": abs_path,
                "name": path_obj.name,
                "extension": path_obj.suffix,
                "directory": str(path_obj.parent),
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "accessed": stat.st_atime,
                "is_file": os.path.isfile(abs_path),
                "is_directory": os.path.isdir(abs_path),
                "is_symlink": os.path.islink(abs_path),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }