import json
import re
import sys

def clean_notebook(filepath):
    print(f"Cleaning {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Prepare Executive Summary cell
    exec_summary_source = [
        "# Executive Summary\n",
        "\n",
        "**Project Goal:** Analyze customer purchasing behavior on Instacart to identify trends in reordering and peak shopping hours.\n",
        "\n",
        "**Key Findings:**\n",
        "- Cleaned and merged multiple relational datasets containing over 4 million rows.\n",
        "- Identified key business metrics such as the most frequently reordered products.\n",
        "- Determined optimal restock times based on shopping hour patterns.\n",
        "\n",
        "**Tech Stack:** Python, Pandas, Matplotlib.\n"
    ]
    
    exec_summary_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": exec_summary_source
    }

    cleaned_cells = []
    
    # Flags to identify if the executive summary requires insertion
    # (Checking if it's already there)
    has_exec_summary = False

    for cell in nb.get('cells', []):
        if not cell.get('source'):
            cleaned_cells.append(cell)
            continue
            
        source_text = "".join(cell['source'])
        
        # Check if the executive summary is already present
        if "# Executive Summary" in source_text:
            has_exec_summary = True
            
        # 2. Skip Reviewer Comments
        if "Reviewer's comment" in source_text or "Reviewers comment" in source_text or '<div class="alert' in source_text:
            print("Removing reviewer comment cell.")
            continue
            
        # 3. Clean course instructions from remaining text
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if "# In this cell, type" not in line and "# In this cell, run" not in line:
                    new_source.append(line)
                else:
                    print("Removing instruction line from code cell.")
            cell['source'] = new_source
            
        elif cell['cell_type'] == 'markdown':
            # Specific course instruction replacements
            if "In the cells below, display the datasets" in source_text:
                print("Rewriting dataset display instruction.")
                cell['source'] = ["Initial Data Exploration"]
            elif "Repeat the use of .info() on the remaining datasets" in source_text:
                print("Rewriting info repeat instruction.")
                cell['source'] = ["Continuing Data Exploration for additional datasets."]
            elif "Repeat this process for each dataset" in source_text:
                print("Removing repeat instructions.")
                continue

        cleaned_cells.append(cell)

    if not has_exec_summary:
        print("Inserting Executive Summary at the top.")
        cleaned_cells.insert(0, exec_summary_cell)
        
    nb['cells'] = cleaned_cells
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Successfully cleaned '{filepath}'.")

if __name__ == "__main__":
    notebook_path = '/home/haven/MyGitRepo/TSanders/portfolio/EDA.ipynb'
    clean_notebook(notebook_path)
