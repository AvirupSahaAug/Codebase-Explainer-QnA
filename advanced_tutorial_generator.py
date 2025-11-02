import os
import re
from typing import List, Dict, Any
import ollama

class AdvancedRepoAnalyzer:
    def __init__(self, model_name="deepseek-coder:16b-q4_0"):
        self.model_name = model_name
        self.client = ollama.Client()
    
    def extract_key_components(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract key components from code files"""
        components = {
            "imports": [],
            "classes": [],
            "functions": [],
            "config_files": [],
            "entry_points": []
        }
        
        for doc in documents:
            content = doc.page_content
            file_path = doc.metadata["source"]
            
            # Extract imports
            imports = re.findall(r'^(?:import|from)\s+[\w\.]+', content, re.MULTILINE)
            if imports:
                components["imports"].extend(imports)
            
            # Extract classes
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            for class_name in classes:
                components["classes"].append({
                    "name": class_name,
                    "file": file_path
                })
            
            # Extract functions
            functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
            for func_name in functions:
                components["functions"].append({
                    "name": func_name,
                    "file": file_path
                })
            
            # Identify config files
            if any(ext in file_path for ext in ['.json', '.yaml', '.yml', '.config', '.ini']):
                components["config_files"].append(file_path)
            
            # Identify entry points
            if any(pattern in file_path for pattern in ['main', 'app', 'index', 'run', 'start']):
                if '__main__' in content or 'if __name__' in content:
                    components["entry_points"].append(file_path)
        
        return components

    def generate_detailed_analysis(self, components: Dict[str, Any], documents: List[Document]) -> str:
        """Generate detailed analysis using LLM"""
        component_summary = f"""
        Repository Components:
        - Classes: {len(components['classes'])}
        - Functions: {len(components['functions'])}
        - Config Files: {components['config_files']}
        - Entry Points: {components['entry_points']}
        """
        
        sample_code = "\n".join([doc.page_content for doc in documents[:10]])
        
        prompt = f"""
        Analyze this code repository and provide detailed insights:
        
        {component_summary}
        
        Sample Code:
        {sample_code[:6000]}
        
        Provide analysis covering:
        1. Architecture overview
        2. Main components and their responsibilities
        3. Data flow and processing
        4. External dependencies
        5. Configuration requirements
        6. Setup and deployment process
        """
        
        response = self.client.generate(model=self.model_name, prompt=prompt)
        return response['response']