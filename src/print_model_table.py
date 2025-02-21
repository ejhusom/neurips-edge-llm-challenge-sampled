import json

# Load the JSON file
with open("ollama_models.json", "r") as f:
    data = json.load(f)

# Start LaTeX table
latex_table = """
\\begin{table}[h]
    \centering
    \\begin{tabular}{|l|c|c|c|}
        \hline
        Model Name & Parameter Size & Size (MB) & Quantization Level \\\\
        \hline
"""

# Populate table with model data
for model in data["models"]:
    name = model["name"].replace("_", "\\_")
    size_bytes = model["size"]
    size_mb = size_bytes / (1024 * 1024)  # Convert bytes to MB
    param_size = model["details"]["parameter_size"]
    quantization = model["details"]["quantization_level"].replace("_", "\\_")
    
    latex_table += f"        \\texttt{{{name}}} & {param_size} & {size_mb:.2f} MB & \\texttt{{{quantization}}} \\\\\n"

# Close LaTeX table
latex_table += """
        \hline
    \end{tabular}
    \caption{Model information extracted from JSON file}
    \label{tab:model_info}
\end{table}
"""

# Save to file
with open("models_table.tex", "w") as f:
    f.write(latex_table)

print("LaTeX table generated and saved as models_table.tex.")
