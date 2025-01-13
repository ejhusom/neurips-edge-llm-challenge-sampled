import csv
import sys
import html

def generate_html_report(csv_file, html_file):
    with open(csv_file, 'r') as infile, open(html_file, 'w') as outfile:
        reader = csv.reader(infile)
        headers = next(reader)

        column_count = len(headers)  # Determine the number of columns dynamically

        outfile.write('<html>\n<head>\n<title>Comparison Results</title>\n')
        outfile.write('<style>\n')
        outfile.write('table { width: 100%; border-collapse: collapse; table-layout: fixed; }\n')
        outfile.write('th, td { border: 1px solid black; padding: 8px; text-align: left; overflow-wrap: break-word; word-wrap: break-word; word-break: break-word; }\n')
        outfile.write('th { background-color: #f2f2f2; }\n')
        outfile.write('col:nth-child(1), col:nth-child(2) { width: 10%; } /* Minimal space for the first two columns */\n')

        # Dynamically set widths for the remaining columns
        if column_count > 2:
            remaining_width = (100 - 20) / (column_count - 2)  # Distribute 80% among remaining columns
            for i in range(3, column_count + 1):
                outfile.write(f'col:nth-child({i}) {{ width: {remaining_width:.2f}%; }}\n')

        outfile.write('</style>\n')
        outfile.write('</head>\n<body>\n')
        outfile.write('<h1>Comparison Results</h1>\n')
        outfile.write('<div style="overflow-x: auto;">\n')  # Allow horizontal scrolling if needed
        outfile.write('<table>\n')
        outfile.write('<colgroup>\n')
        outfile.write('<col span="1">\n<col span="1">\n')  # Minimal width for the first two columns
        outfile.write(f'<col span="{column_count - 2}">\n')  # Remaining columns dynamic
        outfile.write('</colgroup>\n')
        outfile.write('<tr>' + ''.join(f'<th>{html.escape(header)}</th>' for header in headers) + '</tr>\n')

        for row in reader:
            outfile.write('<tr>')
            for cell in row:
                # Escape the cell content to ensure special characters are displayed as text
                escaped_cell = html.escape(cell)
                # Replace newlines with <br> to preserve line breaks
                escaped_cell = escaped_cell.replace('\n', '<br>')
                outfile.write(f'<td>{escaped_cell}</td>')
            outfile.write('</tr>\n')

        outfile.write('</table>\n')
        outfile.write('</div>\n')
        outfile.write('</body>\n</html>')

# def generate_interactive_html_report(csv_file, html_file):
#     with open(csv_file, 'r') as infile, open(html_file, 'w') as outfile:
#         reader = csv.reader(infile)
#         headers = next(reader)

#         column_count = len(headers)  # Determine the number of columns dynamically

#         # Read the rows into memory to identify unique values for the first two columns
#         rows = list(reader)
#         column1_values = sorted(set(row[0] for row in rows))  # Unique values for column 1
#         column2_values = sorted(set(row[1] for row in rows))  # Unique values for column 2

#         outfile.write('<html>\n<head>\n<title>Interactive Comparison Results</title>\n')
#         outfile.write('<style>\n')
#         outfile.write('table { width: 100%; border-collapse: collapse; table-layout: fixed; }\n')
#         outfile.write('th, td { border: 1px solid black; padding: 8px; text-align: left; overflow-wrap: break-word; word-wrap: break-word; word-break: break-word; }\n')
#         outfile.write('th { background-color: #f2f2f2; }\n')
#         outfile.write('col:nth-child(1), col:nth-child(2) { width: 10%; } /* Minimal space for the first two columns */\n')

#         # Dynamically set widths for the remaining columns
#         if column_count > 2:
#             remaining_width = (100 - 20) / (column_count - 2)  # Distribute 80% among remaining columns
#             for i in range(3, column_count + 1):
#                 outfile.write(f'col:nth-child({i}) {{ width: {remaining_width:.2f}%; }}\n')

#         outfile.write('select { margin-bottom: 10px; padding: 5px; }\n')

#         outfile.write('</style>\n')
#         outfile.write('</head>\n<body>\n')

#         # Dropdowns for filtering
#         outfile.write('<h1>Interactive Comparison Results</h1>\n')
#         outfile.write('<label for="filter1">Filter Column 1:</label>\n')
#         outfile.write('<select id="filter1">\n')
#         outfile.write('<option value="">All</option>\n')
#         for value in column1_values:
#             outfile.write(f'<option value="{html.escape(value)}">{html.escape(value)}</option>\n')
#         outfile.write('</select>\n')

#         outfile.write('<label for="filter2">Filter Column 2:</label>\n')
#         outfile.write('<select id="filter2">\n')
#         outfile.write('<option value="">All</option>\n')
#         for value in column2_values:
#             outfile.write(f'<option value="{html.escape(value)}">{html.escape(value)}</option>\n')
#         outfile.write('</select>\n')

#         # Table
#         outfile.write('<div style="overflow-x: auto;">\n')
#         outfile.write('<table id="data-table">\n')
#         outfile.write('<tr>' + ''.join(f'<th>{html.escape(header)}</th>' for header in headers) + '</tr>\n')

#         for row in rows:
#             outfile.write('<tr>')
#             for cell in row:
#                 escaped_cell = html.escape(cell).replace('\n', '<br>')  # Preserve newlines
#                 outfile.write(f'<td>{escaped_cell}</td>')
#             outfile.write('</tr>\n')

#         outfile.write('</table>\n')
#         outfile.write('</div>\n')

#         # JavaScript for filtering
#         outfile.write('''<script>
#         document.addEventListener('DOMContentLoaded', function() {
#             const filter1 = document.getElementById('filter1');
#             const filter2 = document.getElementById('filter2');
#             const table = document.getElementById('data-table');
#             const rows = Array.from(table.getElementsByTagName('tr')).slice(1); // Exclude header

#             function filterTable() {
#                 const value1 = filter1.value.toLowerCase();
#                 const value2 = filter2.value.toLowerCase();

#                 rows.forEach(row => {
#                     const cell1 = row.cells[0].textContent.toLowerCase();
#                     const cell2 = row.cells[1].textContent.toLowerCase();

#                     const matchesFilter1 = !value1 || cell1 === value1;
#                     const matchesFilter2 = !value2 || cell2 === value2;

#                     row.style.display = (matchesFilter1 && matchesFilter2) ? '' : 'none';
#                 });
#             }

#             filter1.addEventListener('change', filterTable);
#             filter2.addEventListener('change', filterTable);
#         });
#         </script>\n''')

#         outfile.write('</body>\n</html>')

def generate_interactive_html_report(csv_file, html_file):
    with open(csv_file, 'r') as infile, open(html_file, 'w') as outfile:
        reader = csv.reader(infile)
        headers = next(reader)

        column_count = len(headers)  # Determine the number of columns dynamically

        # Read the rows into memory to identify unique values for the first two columns
        rows = list(reader)
        column1_values = sorted(set(row[0] for row in rows))  # Unique values for column 1
        column2_values = sorted(set(row[1] for row in rows))  # Unique values for column 2

        outfile.write('<html>\n<head>\n<title>Interactive Comparison Results</title>\n')
        outfile.write('<style>\n')
        outfile.write('table { width: 100%; border-collapse: collapse; table-layout: fixed; }\n')
        outfile.write('th, td { border: 1px solid black; padding: 8px; text-align: left; overflow-wrap: break-word; word-wrap: break-word; word-break: break-word; }\n')
        outfile.write('th { background-color: #f2f2f2; }\n')
        outfile.write('select { margin-bottom: 10px; padding: 5px; }\n')
        outfile.write('</style>\n')
        outfile.write('</head>\n<body>\n')

        # Dropdowns for filtering
        outfile.write('<h1>Interactive Comparison Results</h1>\n')
        outfile.write('<label for="filter1">Filter Column 1:</label>\n')
        outfile.write('<select id="filter1">\n')
        outfile.write('<option value="">All</option>\n')
        for value in column1_values:
            outfile.write(f'<option value="{html.escape(value)}">{html.escape(value)}</option>\n')
        outfile.write('</select>\n')

        outfile.write('<label for="filter2">Filter Column 2:</label>\n')
        outfile.write('<select id="filter2">\n')
        outfile.write('<option value="">All</option>\n')
        for value in column2_values:
            outfile.write(f'<option value="{html.escape(value)}">{html.escape(value)}</option>\n')
        outfile.write('</select>\n')

        # Table with colgroup
        outfile.write('<div style="overflow-x: auto;">\n')
        outfile.write('<table id="data-table">\n')
        
        # Define column widths using colgroup
        outfile.write('<colgroup>\n')
        outfile.write('<col style="width: 10%;">\n')  # First column
        outfile.write('<col style="width: 10%;">\n')  # Second column
        if column_count > 2:
            remaining_width = (100 - 20) / (column_count - 2)  # Distribute 80% among remaining columns
            for _ in range(2, column_count):
                outfile.write(f'<col style="width: {remaining_width:.2f}%;">\n')
        outfile.write('</colgroup>\n')

        # Table header
        outfile.write('<tr>' + ''.join(f'<th>{html.escape(header)}</th>' for header in headers) + '</tr>\n')

        # Table rows
        for row in rows:
            outfile.write('<tr>')
            for cell in row:
                escaped_cell = html.escape(cell).replace('\n', '<br>')  # Preserve newlines
                outfile.write(f'<td>{escaped_cell}</td>')
            outfile.write('</tr>\n')

        outfile.write('</table>\n')
        outfile.write('</div>\n')

        # JavaScript for filtering
        outfile.write('''<script>
        document.addEventListener('DOMContentLoaded', function() {
            const filter1 = document.getElementById('filter1');
            const filter2 = document.getElementById('filter2');
            const table = document.getElementById('data-table');
            const rows = Array.from(table.getElementsByTagName('tr')).slice(1); // Exclude header

            function filterTable() {
                const value1 = filter1.value.toLowerCase();
                const value2 = filter2.value.toLowerCase();

                rows.forEach(row => {
                    const cell1 = row.cells[0].textContent.toLowerCase();
                    const cell2 = row.cells[1].textContent.toLowerCase();

                    const matchesFilter1 = !value1 || cell1 === value1;
                    const matchesFilter2 = !value2 || cell2 === value2;

                    row.style.display = (matchesFilter1 && matchesFilter2) ? '' : 'none';
                });
            }

            filter1.addEventListener('change', filterTable);
            filter2.addEventListener('change', filterTable);
        });
        </script>\n''')

        outfile.write('</body>\n</html>')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_html_report.py <input_csv_file> <output_html_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    html_file = sys.argv[2]
    # generate_html_report(csv_file, html_file)
    generate_interactive_html_report(csv_file, html_file)
