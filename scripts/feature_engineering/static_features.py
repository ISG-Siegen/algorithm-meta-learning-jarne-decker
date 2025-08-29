import inspect
import logging
import os
import pandas as pd
from radon.visitors import ComplexityVisitor
from radon.metrics import h_visit
from radon.raw import analyze
from config import ALGORITHM_PORTFOLIO, ALGO_FEATURES_DIR
import networkx as nx
from collections import Counter
import ast
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("algo-feature-extractor")



class GraphVisitor(ast.NodeVisitor):
    """
    Traverses an AST to build a networkx DiGraph (Directed Graph).
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def visit(self, node):
        # Add the current node to the graph
        self.graph.add_node(node)
        # Iterate through its children
        for child in ast.iter_child_nodes(node):
            # Add an edge from the parent (current node) to the child
            self.graph.add_edge(node, child)
        # Continue traversal
        self.generic_visit(node)

def extract_ast_graph_features(code_string: str) -> dict:
    """
    Parses source code into an AST, converts it to a graph, and extracts
    graph-based metrics.
    """
    features = {
        'ast_node_count': 0, 'ast_edge_count': 0, 'ast_avg_degree': 0.0,
        'ast_max_degree': 0, 'ast_transitivity': 0.0, 'ast_avg_clustering': 0.0,
        'ast_depth': 0
    }
    
    try:
        # 1. Parse code into an AST tree
        tree = ast.parse(code_string)
        
        # 2. Build the graph using our visitor
        visitor = GraphVisitor()
        visitor.visit(tree)
        graph = visitor.graph
        
        if not graph.nodes:
            return features

        # 3. Calculate graph metrics using networkx
        features['ast_node_count'] = graph.number_of_nodes()
        features['ast_edge_count'] = graph.number_of_edges()
        
        degrees = [d for n, d in graph.degree()]
        if degrees:
            features['ast_avg_degree'] = np.mean(degrees)
            features['ast_max_degree'] = np.max(degrees)

        # Transitivity and clustering are defined for undirected graphs
        undirected_graph = graph.to_undirected()
        features['ast_transitivity'] = nx.transitivity(undirected_graph)
        features['ast_avg_clustering'] = nx.average_clustering(undirected_graph)
        
        # Depth of the AST 
        features['ast_depth'] = nx.dag_longest_path_length(graph)
        
        # 4. Count different node types
        node_types = [node.__class__.__name__ for node in graph.nodes()]
        node_counts = Counter(node_types)
        features['ast_count_Assign'] = node_counts.get('Assign', 0)
        features['ast_count_Call'] = node_counts.get('Call', 0)
        features['ast_count_If'] = node_counts.get('If', 0)
        features['ast_count_For'] = node_counts.get('For', 0)

    except Exception as e:
        log.error(f"Could not extract AST graph features: {e}")
    
    return features

def extract_all_code_features(algorithm_map: dict, output_path: str):
    """
    Analyzes the source code of algorithms defined in a map, extracts metrics
    using Radon, and saves the results to a CSV file.
    """
    log.info("Starting source code analysis for all algorithms...")
    analysis_results = []

    for library_name, algorithms in algorithm_map.items():
        for algo_name, algo_config in algorithms.items():
            log.info(f"Analyzing: {library_name} - {algo_name}")
            try:
                algo_class = algo_config["class"]
                file_path = inspect.getfile(algo_class)
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Raw metrics
                raw_metrics = analyze(code)
                
                # Cyclomatic Complexity
                cc_visitor = ComplexityVisitor.from_code(code)
                block_complexities = [func.complexity for func in cc_visitor.functions]
                for cls in cc_visitor.classes:
                    block_complexities.extend(method.complexity for method in cls.methods)
                
                num_complexity_blocks = len(block_complexities)
                avg_cc_file = sum(block_complexities) / num_complexity_blocks if num_complexity_blocks > 0 else 0
                
                # Halstead metrics
                hal_metrics = h_visit(code).total

                current_algo_results = ({
                    "library": library_name,
                    "algorithm_identifier": algo_name,
                    "file_path": file_path,
                    "sloc": raw_metrics.sloc,
                    "lloc": raw_metrics.lloc,
                    "comments": raw_metrics.comments,
                    "blank_lines": raw_metrics.blank,
                    "average_cc_file": avg_cc_file,
                    "num_complexity_blocks": num_complexity_blocks,
                    "hal_volume": hal_metrics.volume,
                    "hal_difficulty": hal_metrics.difficulty,
                    "hal_effort": hal_metrics.effort,
                })
                
                ast_graph_features = extract_ast_graph_features(code)
                current_algo_results.update(ast_graph_features)
                analysis_results.append(current_algo_results)


            except FileNotFoundError:
                log.error(f"  Error: File not found for {library_name} - {algo_name}.")
            except TypeError:
                log.error(f"  Error with inspect.getfile for {library_name} - {algo_name} (likely a built-in or C-extension).")
            except Exception as e:
                log.error(f"  Error analyzing {library_name} - {algo_name}: {e}", exc_info=True)

    if not analysis_results:
        log.warning("No algorithm features were extracted. CSV will not be created.")
        return

    results_df = pd.DataFrame(analysis_results)
    log.info("\n--- Radon Analysis Results ---")
    print(results_df.head().to_string())

    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        log.info(f"Algorithm code features successfully saved to: {os.path.abspath(output_path)}")
    except Exception as e:
        log.error(f"Failed to save algorithm features CSV to {output_path}: {e}", exc_info=True)


if __name__ == '__main__':
    log.info("Running standalone algorithm feature extraction...")
    
    output_csv_path = os.path.join(ALGO_FEATURES_DIR, "algorithm_code_metrics.csv")
    
    extract_all_code_features(ALGORITHM_PORTFOLIO, output_csv_path)