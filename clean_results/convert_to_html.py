import sys
import subprocess

try:
    import markdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"])
    import markdown

css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #2c3e50;
        max-width: 960px;
        margin: 0 auto;
        padding: 3rem 2rem;
        background-color: #f4f7f6;
    }
    .container {
        background-color: #ffffff;
        padding: 3rem 4rem;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    h1, h2, h3, h4 { 
        color: #1a252f; 
        margin-top: 1.8em; 
        font-weight: 700;
    }
    h1 { 
        border-bottom: 3px solid #3498db; 
        padding-bottom: 0.4em; 
        font-size: 2.2rem;
        margin-top: 0;
    }
    h2 { 
        border-bottom: 2px solid #ecf0f1; 
        padding-bottom: 0.3em; 
        font-size: 1.6rem;
    }
    h3 {
        font-size: 1.3rem;
        color: #2980b9;
    }
    code {
        background-color: #f0f4f8;
        padding: 0.2em 0.4em;
        border-radius: 4px;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 0.9em;
        color: #e74c3c;
    }
    pre {
        background-color: #282c34;
        padding: 1.5em;
        border-radius: 8px;
        overflow-x: auto;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    pre code {
        background-color: transparent;
        padding: 0;
        color: #abb2bf;
        font-size: 0.9em;
        line-height: 1.5;
        border-radius: 0;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 2em 0;
        font-size: 0.95em;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-radius: 8px;
        overflow: hidden;
    }
    th, td {
        border-bottom: 1px solid #e1e4e8;
        padding: 15px 20px;
        text-align: left;
    }
    th {
        background-color: #f8f9fa;
        font-weight: 600;
        color: #2c3e50;
        text-transform: uppercase;
        font-size: 0.85em;
        letter-spacing: 0.05em;
    }
    tr:last-child th, tr:last-child td { border-bottom: none; }
    tr:hover { background-color: #fbfdff; }
    blockquote {
        margin: 2em 0;
        padding: 1em 1.5em;
        color: #5c6c7d;
        border-left: 5px solid #3498db;
        background-color: #ebf5fb;
        border-radius: 0 6px 6px 0;
        font-style: italic;
    }
    a {
        color: #3498db;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    ul, ol {
        margin: 1em 0;
        padding-left: 2em;
    }
    li { margin-bottom: 0.5em; }
    
    @media print {
        .no-print { display: none !important; }
        body { background-color: #ffffff; padding: 0; }
        .container { box-shadow: none; padding: 1rem; border-radius: 0; }
    }
    .print-button {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background-color: #3498db;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 50px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
        transition: all 0.3s ease;
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .print-button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.6);
    }
    .print-button svg {
        width: 20px;
        height: 20px;
    }
</style>
"""

with open('d:\\Projects\\ECG\\clean_results\\detailed_model_analysis.md', 'r', encoding='utf-8') as f:
    text = f.read()

html_body = markdown.markdown(text, extensions=['tables', 'fenced_code'])

html_full = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Detailed Model Analysis: SNN, QNN, and Hybrid Architectures</title>
    {css}
</head>
<body>
    <button class="print-button no-print" onclick="window.print()">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"></path></svg>
        Download PDF
    </button>
    <div class="container">
        {html_body}
    </div>
</body>
</html>"""

with open('d:\\Projects\\ECG\\clean_results\\detailed_model_analysis.html', 'w', encoding='utf-8') as f:
    f.write(html_full)

print("HTML generation successful! Saved to detailed_model_analysis.html")
