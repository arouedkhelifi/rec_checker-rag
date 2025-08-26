
# Language detection patterns (keeping existing patterns)
LANGUAGE_PATTERNS = {
    "python": [r'\.py$', r'import\s+', r'from\s+\w+\s+import', r'def\s+\w+\s*\(', r'class\s+\w+\s*:'],
    "javascript": [r'\.js$', r'const\s+', r'let\s+', r'var\s+', r'function\s+', r'import\s+.*from', r'export\s+'],
    "java": [r'\.java$', r'public\s+class', r'private\s+', r'protected\s+', r'import\s+java\.', r'package\s+'],
    "typescript": [r'\.ts$', r'interface\s+', r'type\s+', r'namespace\s+', r'enum\s+'],
    "go": [r'\.go$', r'package\s+main', r'func\s+', r'import\s+\(', r'type\s+\w+\s+struct'],
    "rust": [r'\.rs$', r'fn\s+main', r'let\s+mut', r'use\s+std', r'impl\s+', r'pub\s+fn'],
    "c#": [r'\.cs$', r'namespace\s+', r'using\s+System', r'public\s+class', r'private\s+void'],
    "php": [r'\.php$', r'\<\?php', r'function\s+', r'namespace\s+', r'use\s+'],
    "ruby": [r'\.rb$', r'require\s+', r'def\s+', r'class\s+', r'module\s+'],
    "swift": [r'\.swift$', r'import\s+Foundation', r'func\s+', r'class\s+', r'struct\s+'],
    "kotlin": [r'\.kt$', r'fun\s+main', r'val\s+', r'var\s+', r'class\s+', r'package\s+'],
    "c++": [r'\.cpp$', r'#include', r'using\s+namespace', r'int\s+main', r'class\s+\w+\s*\{'],
    "c": [r'\.c$', r'#include', r'int\s+main', r'void\s+\w+\s*\(', r'struct\s+\w+\s*\{'],
    "html": [r'\.html$', r'\<\!DOCTYPE', r'\<html', r'\<head', r'\<body'],
    "css": [r'\.css$', r'\w+\s*\{', r'@media', r'@import', r'@keyframes'],
    "sql": [r'\.sql$', r'SELECT', r'INSERT', r'UPDATE', r'DELETE', r'CREATE\s+TABLE']
}

# Comment patterns for metrics (keeping existing patterns)
COMMENT_PATTERNS = {
    "python": r'(#[^\n]*|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
    "javascript": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "typescript": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "java": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "go": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "c#": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "php": r'(//[^\n]*|/\*[\s\S]*?\*/|#[^\n]*)',
    "ruby": r'(#[^\n]*|=begin[\s\S]*?=end)',
    "c++": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "c": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "swift": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "kotlin": r'(//[^\n]*|/\*[\s\S]*?\*/)'
}

# Complexity patterns for metrics (keeping existing patterns)
COMPLEXITY_PATTERNS = {
    "python": r'(if|for|while|def|class|with|try|except)',
    "javascript": r'(if|for|while|function|class|try|catch|switch)',
    "typescript": r'(if|for|while|function|class|try|catch|switch|interface)',
    "java": r'(if|for|while|switch|case|try|catch|class|interface)',
    "go": r'(if|for|switch|func|struct|interface)',
    "c#": r'(if|for|while|switch|case|try|catch|class|interface)',
    "php": r'(if|for|while|switch|case|try|catch|class|interface|function)',
    "ruby": r'(if|for|while|case|begin|rescue|class|module|def)',
    "c++": r'(if|for|while|switch|case|try|catch|class|struct)',
    "c": r'(if|for|while|switch|case)',
    "swift": r'(if|for|while|switch|case|try|catch|class|struct|enum)',
    "kotlin": r'(if|for|while|when|try|catch|class|interface|fun)'
}