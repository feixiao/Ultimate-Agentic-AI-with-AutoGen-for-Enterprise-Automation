from typing import Dict, Any
import ast
import operator
from textwrap import dedent
import autogen


class CalculatorTool:
    """A tool for evaluating mathematical expressions derived from natural language queries."""

    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize the CalculatorTool instance.

        Parameters:
        * llm_config: Configuration parameters for the LLM used in expression generation.
        """
        # Allowed operators for safe evaluation
        self.allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        # Create expression generator agent using LLM
        self.expression_generator = autogen.AssistantAgent(
            name="Expression_Generator",
            system_message=dedent("""
                Math expert. You are given a word problem that requires arithmetic computations.
                
                Your task is to generate a Python arithmetic expression to solve the problem.
                Think through each step carefully and proceed logically.
                
                Rules:
                - Use ONLY Python BODMAS operators (+, -, *, /, **, ())
                - Use only numeric values (no variables)
                - Output ONLY the arithmetic expression
                - Do not explain your thinking
                - Pay CLOSE attention to parentheses and order of operations
                - Output the expression in a single line
                
                Examples:
                PROBLEM: If my income is $75,000 and I am taxed at 22%, what will my tax be?
                EXPRESSION: 75000 * 0.22
                
                PROBLEM: Calculate future value of $10,000 with 6% annual interest for 5 years
                EXPRESSION: 10000 + (10000 * 0.06 * 5)
            """),
            llm_config=llm_config,
        )

        # Create user proxy agent to facilitate LLM interaction without human input
        self.user_proxy = autogen.UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression using an Abstract Syntax Tree (AST).

        Parameters:
        * expression: The arithmetic expression as a string.

        Returns:
        * The evaluated numerical result.

        Raises:
        * ValueError: If the expression contains invalid or disallowed operators.
        """
        print("Inside _safe_eval()")
        try:
            # Parse the expression into an AST
            tree = ast.parse(expression, mode="eval")

            # Define a visitor class to safely evaluate the AST nodes
            class SafeEvaluator(ast.NodeTransformer):
                def __init__(self, operators):
                    # Allowed operators passed from CalculatorTool
                    self.operators = operators

                def visit_BinOp(self, node):
                    # Evaluate binary operations (e.g., addition, subtraction)
                    left = self.visit(node.left)
                    right = self.visit(node.right)
                    if type(node.op) not in self.operators:
                        raise ValueError(
                            f"Operator {type(node.op).__name__} not allowed"
                        )
                    return self.operators[type(node.op)](left, right)

                def visit_Num(self, node):
                    # Return numeric literal value
                    return node.n

                def visit_UnaryOp(self, node):
                    # Evaluate unary operations (e.g., negation)
                    operand = self.visit(node.operand)
                    if type(node.op) not in self.operators:
                        raise ValueError(
                            f"Operator {type(node.op).__name__} not allowed"
                        )
                    return self.operators[type(node.op)](operand)

            # Instantiate the safe evaluator with allowed operators
            evaluator = SafeEvaluator(self.allowed_operators)
            # Evaluate the parsed AST safely and return the result
            result = evaluator.visit(tree.body)
            return result

        except Exception as e:
            raise ValueError(f"Error evaluating expression: {str(e)}")

    def run(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language math query and return the computed result.

        Parameters:
        * query: A natural language description of the math problem.

        Returns:
        * A dictionary containing:
          - "result": The numerical result of the computation.
          - "code_generated": The arithmetic expression generated by the LLM.
        * In case of error, returns a dictionary with an "error" key.
        """
        try:
            # Generate arithmetic expression using the LLM agent
            response = self.user_proxy.initiate_chat(
                self.expression_generator,
                message=f"PROBLEM: {query}\nEXPRESSION:",
                clear_history=True,
            )

            # Extract the generated expression from the chat history
            expression = response.chat_history[-1]["content"].strip()

            # Safely evaluate the generated arithmetic expression
            result = self._safe_eval(expression)

            return {"result": result, "code_generated": expression}

        except Exception as e:
            # Return error details if any exception occurs during processing
            return {"error": str(e)}
