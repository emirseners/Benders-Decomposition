from __future__ import annotations
from collections import defaultdict
import re

def read_and_aggregate_capacities(sol_file_path):

    wind_capacity = 6.0

    capacities = defaultdict(lambda: defaultdict(lambda: {'solar': 0.0, 'wind': 0.0, 'battery': 0.0, 'heat_pump': 0.0, 'parabolic_trough': 0.0, 'heat_storage': 0.0}))

    pattern = re.compile(r'^plus_(\d+)\[(\w+),(\d+),(\d+)\]\s+([\d.eE\+\-]+)')

    with open(sol_file_path, 'r') as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue

            node_id = int(m.group(1))
            technology = m.group(2)
            version = int(m.group(3))
            year = int(m.group(4))
            value = float(m.group(5))

            if node_id == 0:
                continue

            if technology == 'solar':
                if version == 0:
                    capacities[node_id][year]['solar'] += 6 * value
                else:
                    capacities[node_id][year]['solar'] += 12 * value
            elif technology == 'wind':
                capacities[node_id][year]['wind'] += wind_capacity * value
            elif technology == 'electricity_storage':
                capacities[node_id][year]['battery'] += value
            elif technology == 'parabolic_trough':
                capacities[node_id][year]['parabolic_trough'] += 4.02 * value
            elif technology == 'heat_pump':
                capacities[node_id][year]['heat_pump'] += value * 246 * 0.001
            elif technology == 'heat_storage':
                capacities[node_id][year]['heat_storage'] += value

    return capacities

def format_capacity(value, capacity_type='general'):
    if abs(value) < 0.01:
        return '-'

    '''    
    if capacity_type in ['solar', 'wind', 'heat_pump', 'parabolic_trough']:
        return f"{round(value):.1f}"
    elif capacity_type in ['battery', 'heat_storage']:
        return f"{value:.1f}"
    else:
    '''
    return f"{value:.1f}"

def generate_latex_code(capacities, model_name):
    latex_code = []

    latex_code.append("\\clearpage")
    latex_code.append("\\begin{landscape}")
    latex_code.append("  \\vspace*{\\fill}")
    latex_code.append("    \\begin{figure}[H]")
    latex_code.append("    \\centering")
    latex_code.append("    \\begin{tikzpicture}[scale=0.95, transform shape, ->, draw=black!60, line width=0.05pt,")
    latex_code.append("        node style/.style={draw, font=\\scriptsize, align=center, text centered, minimum width=1cm, minimum height=1cm},")
    latex_code.append("        edge label/.style={font=\\normalsize}]")
    
    node_positions = {
        1: (-1.05, 0),
        2: (-8.948, -3.8),
        3: (-2.816, -3.8),
        4: (2.816, -3.8),
        5: (8.448, -3.8),
        6: (-10.87, -7.4),
        7: (-10.87, -10.5),
        8: (-7.544, -7.4),
        9: (-7.544, -10.5),
        10: (-4.528, -7.4),
        11: (-4.528, -10.5),
        12: (-1.512, -7.4),
        13: (-1.512, -10.5),
        14: (1.504, -7.4),
        15: (1.504, -10.5),
        16: (4.52, -7.4),
        17: (4.52, -10.5),
        18: (7.536, -7.4),
        19: (7.536, -10.5),
        20: (10.552, -7.4),
        21: (10.552, -10.5)
    }
    
    for node_id in range(1, 22):
        x, y = node_positions[node_id]
        
        node_data = capacities.get(node_id, {})
        
        if node_id == 1:
            node_latex = generate_node1_latex(node_id, x, y, node_data)
        elif node_id == 2:
            node_latex = generate_node2to5_latex(node_id, x, y, node_data)
        elif node_id in [3, 4, 5]:
            node_latex = generate_node3to5_simple_latex(node_id, x, y, node_data)
        else:
            node_latex = generate_node6to21_simple_latex(node_id, x, y, node_data)
        
        latex_code.extend(node_latex)
    
    latex_code.extend(generate_connections())
    
    latex_code.extend(generate_node_labels())
    
    latex_code.append("    \\end{tikzpicture}")
    latex_code.append("    \\vspace{-0.012\\textwidth}")
    latex_code.append(f"    \\caption{{Decision tree for the \\texttt{{{model_name} Case}}.}}")
    latex_code.append("    \\label{fig:base_model_installation_tree}")
    latex_code.append("    \\end{figure}")
    latex_code.append("  \\vspace*{\\fill}")
    latex_code.append("\\end{landscape}")
    latex_code.append("\\clearpage")
    
    return '\n'.join(latex_code)

def generate_node1_latex(node_id, x, y, node_data):
    latex_lines = []
    
    matrix_content = []
    matrix_content.append("{} & Solar (\\si{\\mega\\watt}) & Wind (\\si{\\mega\\watt}) & Elec. St. (\\si{\\mega\\watt\\hour}) & Heat Pump (\\si{\\mega\\watt}) & Par. Tro. (\\si{\\mega\\watt}) & Heat St. (\\si{\\mega\\watt\\hour}) \\\\")
    
    for year in range(1, 6):
        if year in node_data:
            solar = format_capacity(node_data[year]['solar'], 'solar')
            wind = format_capacity(node_data[year]['wind'], 'wind')
            battery = format_capacity(node_data[year]['battery'], 'battery')
            heat_pump = format_capacity(node_data[year].get('heat_pump', 0.0), 'heat_pump')
            parabolic_trough = format_capacity(node_data[year].get('parabolic_trough', 0.0), 'parabolic_trough')
            heat_storage = format_capacity(node_data[year].get('heat_storage', 0.0), 'heat_storage')
            matrix_content.append(f"Year {year} & {solar} & {wind} & {battery} & {heat_pump} & {parabolic_trough} & {heat_storage} \\\\")
        else:
            matrix_content.append(f"Year {year} & - & - & - & - & - & - \\\\")

    latex_lines.append(f"    \\node ({node_id}) [inner sep=0pt] at ({x}, {y}) {{")
    latex_lines.append("      \\begin{tikzpicture}[scale=0.51]")
    latex_lines.append("        \\matrix (M) [matrix of nodes, nodes={draw, minimum size=2.5cm, minimum height=0.52cm, anchor=center, text height=1.6ex, text depth=0.4ex, font=\\scriptsize, draw=black!60, line width=0.05pt}, column 1/.style={nodes={draw=none}}] {")

    for line in matrix_content:
        latex_lines.append(f"          {line}")

    latex_lines.append("        };")
    latex_lines.append("    \\draw[-, dashed, draw=black!60, line width=0.05pt]")
    latex_lines.append("        (M-2-1.north west) -- (M-2-1.north east)")
    latex_lines.append("        (M-3-1.north west) -- (M-3-1.north east)")
    latex_lines.append("        (M-4-1.north west) -- (M-4-1.north east)")
    latex_lines.append("        (M-5-1.north west) -- (M-5-1.north east)")
    latex_lines.append("        (M-6-1.north west) -- (M-6-1.north east)")
    latex_lines.append("        (M-6-1.south west) -- (M-6-1.south east)")
    latex_lines.append("        (M-2-1.north west) -- (M-6-1.south west);")
    latex_lines.append("        \\begin{scope}[on background layer]")
    latex_lines.append("          \\node[fill=solaryellow, fit=(M-1-2)(M-6-2), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=windcyan, fit=(M-1-3)(M-6-3), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=batterypurple, fit=(M-1-4)(M-6-4), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=heatpumporange, fit=(M-1-5)(M-6-5), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=parabolictroughbrown, fit=(M-1-6)(M-6-6), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=heatstoragepink, fit=(M-1-7)(M-6-7), inner sep=0pt] {};")
    latex_lines.append("        \\end{scope}")
    latex_lines.append("      \\end{tikzpicture}")
    latex_lines.append("    };")

    return latex_lines

def generate_node2to5_latex(node_id, x, y, node_data):
    latex_lines = []
    
    matrix_content = []
    for year in range(6, 11):
        if year in node_data:
            solar = format_capacity(node_data[year]['solar'], 'solar')
            wind = format_capacity(node_data[year]['wind'], 'wind')
            battery = format_capacity(node_data[year]['battery'], 'battery')
            heat_pump = format_capacity(node_data[year].get('heat_pump', 0.0), 'heat_pump')
            parabolic_trough = format_capacity(node_data[year].get('parabolic_trough', 0.0), 'parabolic_trough')
            heat_storage = format_capacity(node_data[year].get('heat_storage', 0.0), 'heat_storage')
            matrix_content.append(f"Year {year} & {solar} & {wind} & {battery} & {heat_pump} & {parabolic_trough} & {heat_storage} \\\\")
        else:
            matrix_content.append(f"Year {year} & - & - & - & - & - & - \\\\")

    latex_lines.append(f"    \\node ({node_id}) [inner sep=0pt] at ({x}, {y}) {{")
    latex_lines.append("      \\begin{tikzpicture}[scale=0.51]")
    latex_lines.append("        \\matrix (M) [matrix of nodes, nodes={draw, minimum size=0.6cm, minimum height=0.52cm, anchor=center, font=\\scriptsize, draw=black!60, line width=0.05pt}, column 1/.style={nodes={draw=none}}] {")

    for line in matrix_content:
        latex_lines.append(f"          {line}")

    latex_lines.append("        };")
    latex_lines.append("    \\draw[-, dashed, draw=black!60, line width=0.05pt]")
    latex_lines.append("        (M-1-1.north west) -- (M-1-1.north east)")
    latex_lines.append("        (M-2-1.north west) -- (M-2-1.north east)")
    latex_lines.append("        (M-3-1.north west) -- (M-3-1.north east)")
    latex_lines.append("        (M-4-1.north west) -- (M-4-1.north east)")
    latex_lines.append("        (M-5-1.north west) -- (M-5-1.north east)")
    latex_lines.append("        (M-5-1.south west) -- (M-5-1.south east)")
    latex_lines.append("        (M-1-1.north west) -- (M-5-1.south west);")
    latex_lines.append("        \\begin{scope}[on background layer]")
    latex_lines.append("          \\node[fill=solaryellow, fit=(M-1-2)(M-5-2), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=windcyan, fit=(M-1-3)(M-5-3), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=batterypurple, fit=(M-1-4)(M-5-4), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=heatpumporange, fit=(M-1-5)(M-5-5), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=parabolictroughbrown, fit=(M-1-6)(M-5-6), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=heatstoragepink, fit=(M-1-7)(M-5-7), inner sep=0pt] {};")
    latex_lines.append("        \\end{scope}")
    latex_lines.append("      \\end{tikzpicture}")
    latex_lines.append("    };")

    return latex_lines

def generate_node3to5_simple_latex(node_id, x, y, node_data):
    latex_lines = []
    
    matrix_content = []
    for year in range(6, 11):
        if year in node_data:
            solar = format_capacity(node_data[year]['solar'], 'solar')
            wind = format_capacity(node_data[year]['wind'], 'wind')
            battery = format_capacity(node_data[year]['battery'], 'battery')
            heat_pump = format_capacity(node_data[year].get('heat_pump', 0.0), 'heat_pump')
            parabolic_trough = format_capacity(node_data[year].get('parabolic_trough', 0.0), 'parabolic_trough')
            heat_storage = format_capacity(node_data[year].get('heat_storage', 0.0), 'heat_storage')
            matrix_content.append(f"{solar} & {wind} & {battery} & {heat_pump} & {parabolic_trough} & {heat_storage} \\\\")
        else:
            matrix_content.append("- & - & - & - & - & - \\\\")

    latex_lines.append(f"    \\node ({node_id}) [inner sep=0pt] at ({x}, {y}) {{")
    latex_lines.append("      \\begin{tikzpicture}[scale=0.51]")
    latex_lines.append("        \\matrix (M) [matrix of nodes, nodes={draw, minimum size=0.6cm, minimum height=0.52cm, anchor=center, font=\\scriptsize, draw=black!60, line width=0.05pt}] {")

    for line in matrix_content:
        latex_lines.append(f"          {line}")

    latex_lines.append("        };")
    latex_lines.append("        \\begin{scope}[on background layer]")
    latex_lines.append("          \\node[fill=solaryellow, fit=(M-1-1)(M-5-1), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=windcyan, fit=(M-1-2)(M-5-2), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=batterypurple, fit=(M-1-3)(M-5-3), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=heatpumporange, fit=(M-1-4)(M-5-4), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=parabolictroughbrown, fit=(M-1-5)(M-5-5), inner sep=0pt] {};")
    latex_lines.append("          \\node[fill=heatstoragepink, fit=(M-1-6)(M-5-6), inner sep=0pt] {};")
    latex_lines.append("        \\end{scope}")
    latex_lines.append("      \\end{tikzpicture}")
    latex_lines.append("    };")

    return latex_lines

def generate_node6to21_simple_latex(node_id, x, y, node_data):
    latex_lines = []
    
    if node_id == 6:
        matrix_content = []
        for year in range(11, 16):
            if year in node_data:
                solar = format_capacity(node_data[year]['solar'], 'solar')
                wind = format_capacity(node_data[year]['wind'], 'wind')
                battery = format_capacity(node_data[year]['battery'], 'battery')
                heat_pump = format_capacity(node_data[year].get('heat_pump', 0.0), 'heat_pump')
                parabolic_trough = format_capacity(node_data[year].get('parabolic_trough', 0.0), 'parabolic_trough')
                heat_storage = format_capacity(node_data[year].get('heat_storage', 0.0), 'heat_storage')
                matrix_content.append(f"Y. {year} & {solar} & {wind} & {battery} & {heat_pump} & {parabolic_trough} & {heat_storage} \\\\")
            else:
                matrix_content.append(f"Y. {year} & - & - & - & - & - & - \\\\")

        matrix_content.append("{} & {} & {} & {} & {} & {} & {} \\\\")
        matrix_content.append("{} & {} & {} & {} & {} & {} & {} \\\\")
        
        latex_lines.append(f"    \\node ({node_id}) [inner sep=0pt,")
        latex_lines.append(f"    label={{[yshift=-4pt]}}] at ({x}, {y}) {{")
        latex_lines.append("      \\begin{tikzpicture}[scale=0.5]")
        latex_lines.append("        \\matrix (M) [matrix of nodes, nodes={draw, minimum size=0.40cm, anchor=center, font=\\tiny, draw=black!60, line width=0.05pt},")
        latex_lines.append("        row 6/.style={nodes={draw=none, minimum height = 0.65cm}}, row 7/.style={nodes={draw=none, minimum height = 0.45cm}}, column 4/.style={nodes={minimum width=0.52cm}}, column 1/.style={nodes={draw=none, minimum width=0.62cm}}] {")
        
        for line in matrix_content:
            latex_lines.append(f"          {line}")
        
        latex_lines.append("        };")
        latex_lines.append(f"    \\node[font=\\LARGE, xshift=4pt] at (M-6-3) {{\\textbf{{N: {node_id}}}}};")
        
        s_label = "$S_{1}$: \\(ss \\smalltimes ss\\)"
        latex_lines.append("    \\node[font=\\Large, xshift=4pt, yshift=4pt] at (M-7-3) {\\textbf{" + s_label + "}};")
        latex_lines.append("    \\draw[-, dashed, draw=black!60, line width=0.05pt]")
        latex_lines.append("        (M-1-1.north west) -- (M-1-1.north east)")
        latex_lines.append("        (M-2-1.north west) -- (M-2-1.north east)")
        latex_lines.append("        (M-3-1.north west) -- (M-3-1.north east)")
        latex_lines.append("        (M-4-1.north west) -- (M-4-1.north east)")
        latex_lines.append("        (M-5-1.north west) -- (M-5-1.north east)")
        latex_lines.append("        (M-5-1.south west) -- (M-5-1.south east)")
        latex_lines.append("        (M-1-1.north west) -- (M-5-1.south west);")
        latex_lines.append("        \\begin{scope}[on background layer]")
        latex_lines.append("          \\node[fill=solaryellow, fit=(M-1-2)(M-5-2), inner sep=0pt] {};")
        latex_lines.append("          \\node[fill=windcyan, fit=(M-1-3)(M-5-3), inner sep=0pt] {};")
        latex_lines.append("          \\node[fill=batterypurple, fit=(M-1-4)(M-5-4), inner sep=0pt] {};")
        latex_lines.append("          \\node[fill=heatpumporange, fit=(M-1-5)(M-5-5), inner sep=0pt] {};")
        latex_lines.append("          \\node[fill=parabolictroughbrown, fit=(M-1-6)(M-5-6), inner sep=0pt] {};")
        latex_lines.append("          \\node[fill=heatstoragepink, fit=(M-1-7)(M-5-7), inner sep=0pt] {};")
        latex_lines.append("        \\end{scope}")
        latex_lines.append("      \\end{tikzpicture}")
        latex_lines.append("    };")
    else:
        matrix_content = []
        for year in range(11, 16):
            if year in node_data:
                solar = format_capacity(node_data[year]['solar'], 'solar')
                wind = format_capacity(node_data[year]['wind'], 'wind')
                battery = format_capacity(node_data[year]['battery'], 'battery')
                heat_pump = format_capacity(node_data[year].get('heat_pump', 0.0), 'heat_pump')
                parabolic_trough = format_capacity(node_data[year].get('parabolic_trough', 0.0), 'parabolic_trough')
                heat_storage = format_capacity(node_data[year].get('heat_storage', 0.0), 'heat_storage')
                matrix_content.append(f"{solar} & {wind} & {battery} & {heat_pump} & {parabolic_trough} & {heat_storage} \\\\")
            else:
                matrix_content.append("- & - & - & - & - & - \\\\")

        matrix_content.append("{} & {} & {} & {} & {} & {} & {} \\\\")
        matrix_content.append("{} & {} & {} & {} & {} & {} & {} \\\\")
        
        latex_lines.append(f"    \\node ({node_id}) [inner sep=0pt,")
        latex_lines.append(f"    label={{[yshift=-4pt]}}] at ({x}, {y}) {{")
        latex_lines.append("      \\begin{tikzpicture}[scale=0.5]")
        latex_lines.append("        \\matrix (M) [matrix of nodes, nodes={draw, minimum size=0.42cm, anchor=center, font=\\tiny, draw=black!60, line width=0.05pt}, column 3/.style={nodes={minimum width=0.52cm}},       row 6/.style={nodes={draw=none, minimum height = 0.65cm}}, row 7/.style={nodes={draw=none, minimum height = 0.45cm}}] {")
        
        for line in matrix_content:
            latex_lines.append(f"          {line}")
        
        latex_lines.append("        };")
        latex_lines.append(f"    \\node[font=\\LARGE, xshift=8pt] at (M-6-3) {{\\textbf{{N: {node_id}}}}};")
        
        s_labels = {
            7: "$S_{2}$: \\(ss \\smalltimes sf\\)", 
            8: "$S_{3}$: \\(ss \\smalltimes fs\\)",
            9: "$S_{4}$: \\(ss \\smalltimes ff\\)",
            10: "$S_{5}$: \\(sf \\smalltimes ss\\)",
            11: "$S_{6}$: \\(sf \\smalltimes sf\\)",
            12: "$S_{7}$: \\(sf \\smalltimes fs\\)", 
            13: "$S_{8}$: \\(sf \\smalltimes ff\\)",
            14: "$S_{9}$: \\(fs \\smalltimes ss\\)",
            15: "$S_{10}$: \\(fs \\smalltimes sf\\)",
            16: "$S_{11}$: \\(fs \\smalltimes fs\\)",
            17: "$S_{12}$: \\(fs \\smalltimes ff\\)",
            18: "$S_{13}$: \\(ff \\smalltimes ss\\)",
            19: "$S_{14}$: \\(ff \\smalltimes sf\\)",
            20: "$S_{15}$: \\(ff \\smalltimes fs\\)",
            21: "$S_{16}$: \\(ff \\smalltimes ff\\)"
        }
        
        s_label = s_labels.get(node_id, f"S_{node_id-5}: pattern")
        latex_lines.append("    \\node[font=\\Large, , xshift=8pt, yshift=4pt] at (M-7-3) {\\textbf{" + s_label + "}};")
        latex_lines.append("        \\begin{scope}[on background layer]")
        latex_lines.append("          \\node[fill=solaryellow, fit=(M-1-1)(M-5-1), inner sep=0pt] {};")
        latex_lines.append("          \\node[fill=windcyan, fit=(M-1-2)(M-5-2), inner sep=0pt] {};")
        latex_lines.append("          \\node[fill=batterypurple, fit=(M-1-3)(M-5-3), inner sep=0pt] {};")
        latex_lines.append("          \\node[fill=heatpumporange, fit=(M-1-4)(M-5-4), inner sep=0pt] {};")
        latex_lines.append("          \\node[fill=parabolictroughbrown, fit=(M-1-5)(M-5-5), inner sep=0pt] {};")
        latex_lines.append("          \\node[fill=heatstoragepink, fit=(M-1-6)(M-5-6), inner sep=0pt] {};")
        latex_lines.append("        \\end{scope}")
        latex_lines.append("      \\end{tikzpicture}")
        latex_lines.append("    };")

    return latex_lines

def generate_connections():
    connections = []
    connections.append("    \\draw ([xshift=-1.05cm]1.south) -- ([xshift=1.5cm]2.north) node [midway, left, shift={(-.45,0)}, edge label] {\\footnotesize \\textbf{\\textit{ss}}};")
    connections.append("    \\draw ([xshift=0.5cm]1.south) -- ([xshift=0.6cm]3.north) node [midway, left, shift={(-.05,0)}, edge label] {\\footnotesize \\textbf{\\textit{sf}}};")
    connections.append("    \\draw ([xshift=1.55cm]1.south) -- ([xshift=-0.6cm]4.north) node [midway, right, shift={(.05,0)}, edge label] {\\footnotesize \\textbf{\\textit{fs}}};")
    connections.append("    \\draw ([xshift=3.15cm]1.south) -- ([xshift=-1cm]5.north) node [midway, right, shift={(.45,0)}, edge label] {\\footnotesize \\textbf{\\textit{ff}}};")
#    connections.append("    \\draw ([xshift=-0.5cm]2.south) -- ([xshift=0.25cm]6.north) node [midway, left, shift={(-.13,0)}, edge label] {\\footnotesize \\textbf{\\textit{ss}}};")
#    connections.append("    \\draw ([xshift=0.2cm]2.south) -- (7.north) node [midway, left, shift={(.02,0)}, edge label] {\\footnotesize \\textbf{\\textit{sf}}};")
#    connections.append("    \\draw ([xshift=0.7cm]2.south) -- (8.north) node [midway, right, shift={(-.02,0)}, edge label] {\\footnotesize \\textbf{\\textit{fs}}};")
#    connections.append("    \\draw ([xshift=1.5cm]2.south) -- (9.north) node [midway, right, shift={(.13,0)}, edge label] {\\footnotesize \\textbf{\\textit{ff}}};")
#    connections.append("    \\draw (3) -- (10.north) node [midway, left, edge label] {\\footnotesize \\textbf{\\textit{ss}}};")
#    connections.append("    \\draw (3) -- (11.north) node [midway, left, edge label] {\\footnotesize \\textbf{\\textit{sf}}};")
#    connections.append("    \\draw (3) -- (12.north) node [midway, right, edge label] {\\footnotesize \\textbf{\\textit{fs}}};")
#    connections.append("    \\draw (3) -- (13.north) node [midway, right, edge label] {\\footnotesize \\textbf{\\textit{ff}}};")
#    connections.append("    \\draw (4) -- (14.north) node [midway, left, edge label] {\\footnotesize \\textbf{\\textit{ss}}};")
#    connections.append("    \\draw (4) -- (15.north) node [midway, left, edge label] {\\footnotesize \\textbf{\\textit{sf}}};")
#    connections.append("    \\draw (4) -- (16.north) node [midway, right, edge label] {\\footnotesize \\textbf{\\textit{fs}}};")
#    connections.append("    \\draw (4) -- (17.north) node [midway, right, edge label] {\\footnotesize \\textbf{\\textit{ff}}};")
#    connections.append("    \\draw (5) -- (18.north) node [midway, left, shift={(-.13,0)}, edge label] {\\footnotesize \\textbf{\\textit{ss}}};")
#    connections.append("    \\draw (5) -- (19.north) node [midway, left, shift={(.05,0)}, edge label] {\\footnotesize \\textbf{\\textit{sf}}};")
#    connections.append("    \\draw (5) -- (20.north) node [midway, right, shift={(-.03,0)}, edge label] {\\footnotesize \\textbf{\\textit{fs}}};")
#    connections.append("    \\draw (5) -- (21.north) node [midway, right, shift={(.13,0)}, edge label] {\\footnotesize \\textbf{\\textit{ff}}};")
    
    return connections

def generate_node_labels():
    labels = []
    labels.append("    \\node[above=2pt of 1, xshift=1.05cm] {\\small \\textbf{N: 1}};")
    labels.append("    \\node[above=2pt of 2, xshift=0.5cm] {\\small \\textbf{N: 2}};")
    labels.append("    \\node[above=2pt of 3] {\\small \\textbf{N: 3}};")
    labels.append("    \\node[above=2pt of 4] {\\small \\textbf{N: 4}};")
    labels.append("    \\node[above=2pt of 5] {\\small \\textbf{N: 5}};")
    return labels

def main():
    model_name = '3_5_1092'
    epsilon = 0
    folder_suffix = f"eps({epsilon})_base"
    sol_file = f"Results_{model_name}_{folder_suffix}/Results.sol"

    capacities = read_and_aggregate_capacities(sol_file)
    latex_output = generate_latex_code(capacities, model_name)

    output_name = 'DT_' + model_name + '.tex'
    with open(output_name, 'w') as f:
        f.write(latex_output)

if __name__ == "__main__":
    main()