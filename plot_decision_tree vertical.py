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
                capacities[node_id][year]['battery'] += 10 * value
            elif technology == 'parabolic_trough':
                capacities[node_id][year]['parabolic_trough'] += 4.02 * value
            elif technology == 'heat_pump':
                capacities[node_id][year]['heat_pump'] += value * 246 * 0.001
            elif technology == 'heat_storage':
                capacities[node_id][year]['heat_storage'] += 10 * value

    return capacities

def format_capacity(value, capacity_type='general'):
    if abs(value) < 0.01:
        return '-'
    return f"{value:.1f}"

def generate_latex_code(capacities, model_name):
    latex_code = []

    latex_code.append("\\clearpage")
    latex_code.append("    \\begin{figure}[H]")
    latex_code.append("    \\centering")
    latex_code.append("    \\begin{adjustbox}{max width=\\linewidth, max height=0.93\\textheight, keepaspectratio}")
    latex_code.append("    \\begin{tikzpicture}[scale=0.95, transform shape, ->, draw=black!55, line width=0.05pt,")
    latex_code.append("        node style/.style={draw, font=\\scriptsize, align=center, text centered, minimum width=1cm, minimum height=1cm},")
    latex_code.append("        edge label/.style={font=\\normalsize}]")
    
    node_positions = {
        1: (-0.36, -0.52),
        2: (-6.1, -3.7),
        3: (-1.85, -3.7),
        4: (2.05, -3.7),
        5: (6, -3.7),
        6: (-6.1, -7),
        7: (-6.1, -9.85),
        8: (-6.1, -12.7),
        9: (-6.1, -15.55),
        10: (-1.85, -7),
        11: (-1.85, -9.85),
        12: (-1.85, -12.7),
        13: (-1.85, -15.55),
        14: (2.1, -7),
        15: (2.1, -9.85),
        16: (2.1, -12.7),
        17: (2.1, -15.55),
        18: (6.05, -7),
        19: (6.05, -9.85),
        20: (6.05, -12.7),
        21: (6.05, -15.55)
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
    latex_code.extend(generate_group_rectangles())
    latex_code.extend(generate_stage_braces())
    
    latex_code.append("    \\end{tikzpicture}")
    latex_code.append("    \\end{adjustbox}")
    latex_code.append("    \\vspace{-0.012\\textwidth}")
    latex_code.append(f"    \\caption{{Decision tree for the base model.}}")
    latex_code.append("    \\label{fig:base_model_installation_tree}")
    latex_code.append("    \\end{figure}")
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
            matrix_content.append(f"Y. {year} & {solar} & {wind} & {battery} & {heat_pump} & {parabolic_trough} & {heat_storage} \\\\")
        else:
            matrix_content.append(f"Y. {year} & - & - & - & - & - & - \\\\")

    latex_lines.append(f"    \\node ({node_id}) [inner sep=0pt] at ({x}, {y}) {{")
    latex_lines.append("      \\begin{tikzpicture}[scale=0.45]")
    latex_lines.append("        \\matrix (M) [matrix of nodes, nodes={draw, text width=1.9cm, minimum width=2.1cm, minimum height=0.38cm, inner sep=1pt, align=center, anchor=center, text height=1.6ex, text depth=0.4ex, font={\\fontsize{6.8}{7.8}\\selectfont}, draw=black!55, line width=0.05pt}, column 1/.style={nodes={draw=none, text width=0.54cm, minimum width=0.63cm}}] {")

    for line in matrix_content:
        latex_lines.append(f"          {line}")

    latex_lines.append("        };")
    latex_lines.append("    \\draw[-, dashed, draw=black!55, line width=0.05pt]")
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
            matrix_content.append(f"Y. {year} & {solar} & {wind} & {battery} & {heat_pump} & {parabolic_trough} & {heat_storage} \\\\")
        else:
            matrix_content.append(f"Y. {year} & - & - & - & - & - & - \\\\")

    latex_lines.append(f"    \\node ({node_id}) [inner sep=0pt] at ({x}, {y}) {{")
    latex_lines.append("      \\begin{tikzpicture}[scale=0.53]")
    latex_lines.append("        \\matrix (M) [matrix of nodes, nodes={draw, text width=0.54cm, minimum width=0.58cm, minimum height=0.42cm, inner sep=0.5pt, align=center, anchor=center, text height=1.2ex, text depth=0.3ex, font={\\fontsize{6.8}{7.8}\\selectfont}, draw=black!55, line width=0.05pt}, column 1/.style={nodes={draw=none, text width=0.55cm, minimum width=0.59cm}}] {")

    for line in matrix_content:
        latex_lines.append(f"          {line}")

    latex_lines.append("        };")
    latex_lines.append("    \\draw[-, dashed, draw=black!55, line width=0.05pt]")
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
    latex_lines.append("      \\begin{tikzpicture}[scale=0.53]")
    latex_lines.append("        \\matrix (M) [matrix of nodes, nodes={draw, text width=0.54cm, minimum width=0.58cm, minimum height=0.42cm, inner sep=0.5pt, align=center, anchor=center, text height=1.2ex, text depth=0.3ex, font={\\fontsize{6.8}{7.8}\\selectfont}, draw=black!55, line width=0.05pt}] {")

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
    
    if node_id in [6, 7, 8, 9]:
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
        
        latex_lines.append(f"    \\node ({node_id}) [inner sep=0pt,")
        latex_lines.append(f"    label={{[yshift=-4pt]}}] at ({x}, {y}) {{")
        latex_lines.append("      \\begin{tikzpicture}[scale=0.52]")
        latex_lines.append("        \\matrix (M) [matrix of nodes, nodes={draw, text width=0.54cm, minimum width=0.58cm, minimum height=0.38cm, inner sep=0.5pt, align=center, anchor=center, text height=1.2ex, text depth=0.3ex, font={\\fontsize{6.4}{7.4}\\selectfont}, draw=black!55, line width=0.05pt}, row 6/.style={nodes={draw=none, minimum height = 0.50cm}}, column 1/.style={nodes={draw=none, text width=0.55cm, minimum width=0.59cm}}] {")
        
        for line in matrix_content:
            latex_lines.append(f"          {line}")
        
        latex_lines.append("        };")
        
        s_labels_with_year = {
            6: "$S_{1}$: \\(ss \\smalltimes ss\\)",
            7: "$S_{2}$: \\(ss \\smalltimes sf\\)",
            8: "$S_{3}$: \\(ss \\smalltimes fs\\)",
            9: "$S_{4}$: \\(ss \\smalltimes ff\\)",
        }
        s_label = s_labels_with_year[node_id]
        latex_lines.append("    \\node[font=\\normalsize, xshift=4pt] at (M-6-4) {" + s_label + "};")
        latex_lines.append("    \\draw[-, dashed, draw=black!55, line width=0.05pt]")
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

        matrix_content.append("{} & {} & {} & {} & {} & {} \\\\")
        
        latex_lines.append(f"    \\node ({node_id}) [inner sep=0pt,")
        latex_lines.append(f"    label={{[yshift=-4pt]}}] at ({x}, {y}) {{")
        latex_lines.append("      \\begin{tikzpicture}[scale=0.52]")
        latex_lines.append("        \\matrix (M) [matrix of nodes, nodes={draw, text width=0.54cm, minimum width=0.58cm, minimum height=0.38cm, inner sep=0.5pt, align=center, anchor=center, text height=1.2ex, text depth=0.3ex, font={\\fontsize{6.4}{7.4}\\selectfont}, draw=black!55, line width=0.05pt}, row 6/.style={nodes={draw=none, minimum height = 0.50cm}}] {")
        
        for line in matrix_content:
            latex_lines.append(f"          {line}")
        
        latex_lines.append("        };")
        
        s_labels = {
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
        latex_lines.append("    \\node[font=\\normalsize, xshift=8pt] at (M-6-3) {" + s_label + "};")
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
    connections.append("    \\draw ([xshift=-0.25cm]1.south) -- ([xshift=0.14cm]2.north) node [midway, left, shift={(-.25,0)}, edge label] {\\fontsize{7.2}{8.2}\\selectfont \\textit{ss}};")
    connections.append("    \\draw ([xshift=0.05cm]1.south) -- (3.north) node [midway, left, shift={(-.05,0)}, edge label] {\\fontsize{7.2}{8.2}\\selectfont \\textit{sf}};")
    connections.append("    \\draw ([xshift=0.35cm]1.south) -- (4.north) node [midway, right, shift={(.05,0)}, edge label] {\\fontsize{7.2}{8.2}\\selectfont \\textit{fs}};")
    connections.append("    \\draw ([xshift=0.65cm]1.south) -- (5.north) node [midway, right, shift={(.25,0)}, edge label] {\\fontsize{7.2}{8.2}\\selectfont \\textit{ff}};")

    return connections

def generate_node_labels():
    labels = []
    labels.append("    \\node[above=1.0pt of 1, xshift=0.3cm] {\\fontsize{6.4}{7.4}\\selectfont N: 1};")
    labels.append("    \\node[above=1.0pt of 2, xshift=0.2cm] {\\fontsize{6.4}{7.4}\\selectfont N: 2};")
    labels.append("    \\node[above=1.0pt of 3] {\\fontsize{6.4}{7.4}\\selectfont N: 3};")
    labels.append("    \\node[above=1.0pt of 4] {\\fontsize{6.4}{7.4}\\selectfont N: 4};")
    labels.append("    \\node[above=1.0pt of 5] {\\fontsize{6.4}{7.4}\\selectfont N: 5};")
    for nid in range(6, 22):
        labels.append(f"    \\node[above=1.0pt of {nid}] {{\\fontsize{{6.4}}{{7.4}}\\selectfont N: {nid}}};")
    return labels

def generate_group_rectangles():
    lines = []
    groups = [
        (2, [6, 7, 8, 9]),
        (3, [10, 11, 12, 13]),
        (4, [14, 15, 16, 17]),
        (5, [18, 19, 20, 21]),
    ]
    for parent, children in groups:
        fit_nodes = "".join(f"({c})" for c in children)
        group_name = f"group{parent}"
        lines.append(f"    \\node[draw, draw=black!45, line width=0.4pt, rounded corners=3pt, inner xsep=6pt, inner ysep=8.5pt, fit={fit_nodes}, yshift=7.2pt] ({group_name}) {{}};")
        lines.append(f"    \\draw[draw=black!45, line width=0.4pt] ({parent}.south) -- ({parent}.south |- {group_name}.north);")
    return lines

def generate_stage_braces():
    lines = []
    lines.append("    \\draw[-, decorate, decoration={brace, amplitude=5pt}, thick, draw=black!55] (-8.6, -1.79) -- (-8.6, 0.75) node[midway, left=6pt, font=\\scriptsize, rotate=90, anchor=south] {Stage 1};")
    lines.append("    \\draw[-, decorate, decoration={brace, amplitude=5pt}, thick, draw=black!55] (-8.6, -4.78) -- (-8.6, -2.6) node[midway, left=6pt, font=\\scriptsize, rotate=90, anchor=south] {Stage 2};")
    lines.append("    \\draw[-, decorate, decoration={brace, amplitude=5pt}, thick, draw=black!55] (-8.6, -16.82) -- (-8.6, -5.2) node[midway, left=6pt, font=\\scriptsize, rotate=90, anchor=south] {Stage 3};")
    return lines

def main():
    model_name = '3_5_4368'
    epsilon = 0
    worst_sp_incumbent_flag = False
    valid_inequalities_flag = True
    continuous_flag = True
    folder_suffix = f"eps({epsilon})_cont({continuous_flag})_vi({valid_inequalities_flag})_inc({worst_sp_incumbent_flag})_budget_relax"
    sol_file = f"Results_{model_name}_{folder_suffix}/Results.sol"

    capacities = read_and_aggregate_capacities(sol_file)
    latex_output = generate_latex_code(capacities, model_name)

    output_name = 'DT_' + model_name + '.tex'
    with open(output_name, 'w') as f:
        f.write(latex_output)

if __name__ == "__main__":
    main()