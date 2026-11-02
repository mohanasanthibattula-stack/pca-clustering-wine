"""Execute notebook and export HTML."""
import sys
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

def main():
    proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    nb_path = os.path.join(proj_dir, "notebooks", "pca_clustering.ipynb")
    html_path = os.path.join(proj_dir, "reports", "pca_clustering.html")
    
    print(f"Loading notebook from {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    print("Executing notebook with ExecutePreprocessor (working_dir = notebooks)...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    notebook_dir = os.path.join(proj_dir, "notebooks")
    ep.preprocess(nb, {'metadata': {'path': notebook_dir}})
    
    print("Saving executed notebook...")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print(f"Exporting HTML to {html_path}...")
    html_exporter = HTMLExporter()
    body, _ = html_exporter.from_notebook_node(nb)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(body)
        
    print("Notebook executed and HTML exported successfully!")

if __name__ == "__main__":
    main()
