import ast
from typing import Optional

class StringUtils:
    '''
    Class containing static methods that help handle strings, such as parsing
    '''

    @staticmethod
    def parse_object(code_string: str, object_name: str) -> Optional[str]:
        '''
        Parse the code string to find the object with the given name. If the object is not found, return None.
        '''
        try:
        # Parse the code string into an AST (Abstract Syntax Tree)
            tree = ast.parse(code_string)
        except SyntaxError as e:
            # print(f"SyntaxError: {e}")
            return None
        
        # Find the object in the AST
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == object_name:
                start_lineno = node.lineno - 1
                end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else None
                lines = code_string.split('\n')
                if end_lineno:
                    return '\n'.join(lines[start_lineno:end_lineno])
                else:
                    # If end_lineno is not available, find the ending manually
                    indent_level = len(lines[start_lineno]) - len(lines[start_lineno].lstrip())
                    for i in range(start_lineno + 1, len(lines)):
                        if len(lines[i]) - len(lines[i].lstrip()) <= indent_level:
                            end_lineno = i
                            break
                    if end_lineno:
                        return '\n'.join(lines[start_lineno:end_lineno])
                    else:
                        return '\n'.join(lines[start_lineno:])
        
        return None