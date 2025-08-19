import zipfile
import os
import tempfile

def prepare_state_from_input(uploaded_path: str, code_language: str, target_language="english") -> dict:
    """
    Prepare the RAG agent state from the uploaded input file.

    Args:
        uploaded_path (str): Path to the uploaded file (can be a ZIP archive or a single code file).
        code_language (str): The programming language specified or detected.
        target_language (str, optional): The language for output recommendations. Defaults to "english".

    Returns:
        dict: State dictionary containing language info and a list of code files with their content.
              Format: { "code_language": ..., "target_language": ..., "files": [(filename, code_str), ...] }
    """

    # Initialize the state with language info
    state = {
        "code_language": code_language,
        "target_language": target_language
    }

    # Handle ZIP archive input
    if uploaded_path.endswith(".zip"):
        files = []
        # Create a temporary directory to extract the ZIP contents
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract all files inside the ZIP archive to the temp directory
            with zipfile.ZipFile(uploaded_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            # Walk through the extracted directory tree
            for root, dirs, filelist in os.walk(tmpdir):
                for file in filelist:
                    # Only process supported source code files by extension
                    if file.endswith((
                        ".py", ".js", ".java", ".cpp", ".c", ".ts", ".rb", ".php", ".go",
                        ".rs", ".cs", ".swift", ".kt", ".html", ".css", ".sql"
                    )):
                        filepath = os.path.join(root, file)
                        try:
                            # Read the code content with UTF-8 encoding
                            with open(filepath, 'r', encoding='utf-8') as f:
                                code = f.read()
                                # Only add files with non-empty code content
                                if code.strip():
                                    # Save relative path inside ZIP for display purposes
                                    relpath = os.path.relpath(filepath, tmpdir)
                                    files.append((relpath, code))
                        except Exception:
                            # Ignore files that cannot be read due to encoding or other issues
                            pass

        # If no supported code files found, add a placeholder message
        if not files:
            state["files"] = [("No supported code files found in the ZIP.", "")]
        else:
            state["files"] = files

    else:
        # Handle single code file input (not a ZIP)
        with open(uploaded_path, 'r', encoding='utf-8') as f:
            code = f.read()
            # Store the filename and code content in state
            state["files"] = [(os.path.basename(uploaded_path), code)]

    return state
