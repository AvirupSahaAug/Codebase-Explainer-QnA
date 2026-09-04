#!/usr/bin/env python3
"""
Code Graph Builder: Extract AST-based code structure and relationships
Builds a graph of functions, classes, imports, and their dependencies
"""

import os
import ast
import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import networkx as nx
from langchain_core.documents import Document


class CodeGraphBuilder:
    """Build and analyze code dependency graph using AST"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.graph = nx.DiGraph()
        self.entities = defaultdict(dict)  # {file: {name: {type, lineno, dependencies}}}
        self.imports = defaultdict(set)  # {file: set of imported modules}
        self.file_summary = {}  # {file: summary}
        
    def build_graph(self, documents: List[Document]) -> nx.DiGraph:
        """Build AST-based code graph from documents"""
        print("🔗 Building code dependency graph...")
        
        python_docs = [doc for doc in documents if doc.metadata.get('file_type') == '.py']
        
        for doc in python_docs:
            self._analyze_python_file(doc)
        
        # Add edges based on dependencies
        self._add_dependency_edges()
        
        print(f"✅ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _analyze_python_file(self, doc: Document):
        """Analyze a Python file using AST"""
        filepath = doc.metadata.get('source', 'unknown')
        
        try:
            tree = ast.parse(doc.page_content)
        except SyntaxError:
            return
        
        file_node = f"file:{filepath}"
        self.graph.add_node(file_node, type='file', label=filepath)
        
        # Extract top-level definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = f"func:{filepath}:{node.name}"
                self.graph.add_node(func_name, type='function', label=node.name, file=filepath, lineno=node.lineno)
                self.graph.add_edge(file_node, func_name)
                
                if filepath not in self.entities:
                    self.entities[filepath] = {}
                self.entities[filepath][node.name] = {
                    'type': 'function',
                    'lineno': node.lineno,
                    'dependencies': self._extract_dependencies(node)
                }
            
            elif isinstance(node, ast.ClassDef):
                class_name = f"class:{filepath}:{node.name}"
                self.graph.add_node(class_name, type='class', label=node.name, file=filepath, lineno=node.lineno)
                self.graph.add_edge(file_node, class_name)
                
                if filepath not in self.entities:
                    self.entities[filepath] = {}
                self.entities[filepath][node.name] = {
                    'type': 'class',
                    'lineno': node.lineno,
                    'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    'dependencies': self._extract_dependencies(node)
                }
                
                # Add methods as sub-nodes
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = f"method:{filepath}:{node.name}.{item.name}"
                        self.graph.add_node(method_name, type='method', label=item.name, file=filepath)
                        self.graph.add_edge(class_name, method_name)
        
        # Extract imports
        self._extract_imports(tree, filepath)
    
    def _extract_dependencies(self, node: ast.AST) -> Set[str]:
        """Extract function/variable calls from an AST node"""
        dependencies = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.add(child.func.attr)
        return dependencies
    
    def _extract_imports(self, tree: ast.AST, filepath: str):
        """Extract import statements"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports[filepath].add(alias.name)
                    import_node = f"import:{alias.name}"
                    self.graph.add_node(import_node, type='import', label=alias.name)
                    self.graph.add_edge(f"file:{filepath}", import_node)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or 'unknown'
                self.imports[filepath].add(module)
                for alias in node.names:
                    import_node = f"import:{module}.{alias.name}"
                    self.graph.add_node(import_node, type='import', label=f"{module}.{alias.name}")
                    self.graph.add_edge(f"file:{filepath}", import_node)
    
    def _add_dependency_edges(self):
        """Connect nodes based on extracted dependencies"""
        for filepath, entities in self.entities.items():
            for entity_name, entity_info in entities.items():
                deps = entity_info.get('dependencies', set())
                for dep in deps:
                    # Try to find matching function/class in same or imported files
                    for other_file, other_entities in self.entities.items():
                        if dep in other_entities:
                            source = f"{entity_info['type']}:{filepath}:{entity_name}"
                            target = f"{other_entities[dep]['type']}:{other_file}:{dep}"
                            if self.graph.has_node(source) and self.graph.has_node(target):
                                self.graph.add_edge(source, target, type='calls')
    
    def get_context_for_entity(self, filepath: str, entity_name: str, depth: int = 2) -> str:
        """Get context for an entity including its callers and callees"""
        node_id = None
        if filepath in self.entities and entity_name in self.entities[filepath]:
            entity_type = self.entities[filepath][entity_name]['type']
            node_id = f"{entity_type}:{filepath}:{entity_name}"
        
        if not node_id or node_id not in self.graph:
            return ""
        
        context_parts = []
        context_parts.append(f"📦 {entity_name} (Entity in {filepath}):")
        
        # Get predecessors (callers)
        predecessors = list(self.graph.predecessors(node_id))
        if predecessors:
            context_parts.append("  Callers:")
            for pred in predecessors[:5]:
                context_parts.append(f"    - {pred}")
        
        # Get successors (callees)
        successors = list(self.graph.successors(node_id))
        if successors:
            context_parts.append("  Calls:")
            for succ in successors[:5]:
                context_parts.append(f"    - {succ}")
        
        return "\n".join(context_parts)
    
    def get_related_files(self, filepath: str, depth: int = 1) -> List[str]:
        """Get files related to the given file through imports and dependencies"""
        file_node = f"file:{filepath}"
        related = set()
        
        if file_node not in self.graph:
            return []
        
        visited = set()
        queue = [(file_node, 0)]
        
        while queue:
            node, d = queue.pop(0)
            if d >= depth or node in visited:
                continue
            visited.add(node)
            
            for neighbor in self.graph.neighbors(node):
                if neighbor.startswith('file:'):
                    related.add(neighbor.replace('file:', ''))
                queue.append((neighbor, d + 1))
        
        return list(related)
    
    def get_graph_summary(self) -> Dict:
        """Get summary statistics of the graph"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'node_types': self._count_node_types(),
            'density': nx.density(self.graph),
            'files': len(self.entities)
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type"""
        counts = defaultdict(int)
        for node, attrs in self.graph.nodes(data=True):
            counts[attrs.get('type', 'unknown')] += 1
        return dict(counts)
    
    def export_as_documents(self, documents: List[Document]) -> List[Document]:
        """Create enhanced documents with graph context"""
        enhanced_docs = []
        
        for doc in documents:
            if doc.metadata.get('file_type') == '.py':
                filepath = doc.metadata.get('source', '')
                related_files = self.get_related_files(filepath, depth=1)
                related_context = ", ".join(related_files) if related_files else "No related files"
                
                enhanced_content = f"""
{doc.page_content}

---CODE GRAPH CONTEXT---
File: {filepath}
Related Files: {related_context}
Entity Count: {len(self.entities.get(filepath, {}))}
"""
                enhanced_docs.append(Document(
                    page_content=enhanced_content,
                    metadata={
                        **doc.metadata,
                        'has_graph_context': True,
                        'related_files': related_files
                    }
                ))
            else:
                enhanced_docs.append(doc)
        
        return enhanced_docs
    
    def export_graph_json(self, filepath: str):
        """Export graph as JSON for visualization"""
        data = {
            'nodes': [],
            'edges': [],
            'summary': self.get_graph_summary()
        }
        
        for node, attrs in self.graph.nodes(data=True):
            data['nodes'].append({
                'id': node,
                **attrs
            })
        
        for source, target, attrs in self.graph.edges(data=True):
            data['edges'].append({
                'source': source,
                'target': target,
                **attrs
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"📊 Graph exported to {filepath}")
