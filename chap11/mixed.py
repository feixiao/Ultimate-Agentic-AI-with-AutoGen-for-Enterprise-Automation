## vague prompt
vague_prompt: str = """
Analyze the given code.
"""

## specific prompt
specific_prompt: str = """
Analyze the following Python code for potential security vulnerabilities, 
focusing specifically on input validation, authentication mechanisms, 
and data handling practices. 
Provide concrete suggestions for addressing each identified vulnerability with code examples.
"""

## delimited prompt
prompt: str = """
<instruction>
Translate the following English text into formal French, maintaining the original tone and meaning.
</instruction>

<input>
The board of directors has unanimously approved the proposed merger, pending regulatory review. Shareholders will be notified of the decision within five business days.
</input>

<output_format>
Provide the translation followed by a brief note on any cultural or linguistic adaptations made.
</output_format>
"""

## basic role definition
prompt: str = """
You are a financial analyst evaluating investment opportunities.
"""

## intermediate role definition
prompt: str = """
You are a senior financial analyst with 15 years of experience in evaluating technology startups. 
You specialize in SaaS business models and have a background in both quantitative analysis and 
qualitative market assessment.
"""


## advanced role definition
prompt: str = """
You are Dr. Sofia Chen, a senior financial analyst with Goldman Sachs' technology investment division. 
  You hold a Ph.D. in Financial Economics from MIT and have 15 years of experience evaluating early and 
  growth-stage technology companies, with particular expertise in SaaS, fintech, and AI startups.

Your analytical approach balances rigorous quantitative methods with industry insights gained 
  from your extensive professional network. You're known for your cautious outlook, 
  emphasis on sustainable unit economics, and attention to regulatory risks that might be 
  overlooked by more optimistic analysts.

When evaluating companies, you prioritize:
1. Cash flow sustainability and path to profitability
2. Defensibility of market position and technological moats
3. Alignment of incentives in management compensation structures
4. Regulatory and compliance robustness, particularly in data governance
5. Realistic assessment of addressable market size

Your communication style is direct and data-driven, with a preference for concise 
  explanations supported by specific metrics and comparisons to industry benchmarks.
"""

## input format
prompt: str = """
Provided below the patient data in the following format:
 Patient ID: [alphanumeric identifier]
 Age: [age in years]
 Gender: [M/F/Other]
 Vital Signs:
  - Blood Pressure: [systolic/diastolic mmHg]
  - Heart Rate: [beats per minute]
  - Temperature: [degrees Celsius]
  - Respiratory Rate: [breaths per minute]
 Primary Symptoms: [comma-separated list]
 Medical History: [relevant conditions]
 Current Medications: [comma-separated list]
"""

## output format
prompt: str = """
 Structure your analysis as follows:
  1. Executive Summary (2-3 sentences highlighting key findings)
  2. Methodology (brief description of analytical approach)
  3. Key Findings (3-5 bullet points with supporting data)
  4. Detailed Analysis (organized by theme with supporting evidence)
  5. Limitations (acknowledgment of constraints or uncertainties)
  6. Recommendations (actionable next steps, prioritized)
"""

## few shot prompting
prompt: str = """
Task: Classify the sentiment of customer reviews as positive, negative, or neutral.

Example 1:
Review: "The battery life on this laptop is incredible - I can go days without charging!"
Sentiment: Positive

Example 2:
Review: "The software is decent but the hardware feels cheap and the keyboard started failing after just three months."
Sentiment: Negative

Example 3:
Review: "Delivery was on time. Product works as expected."
Sentiment: Neutral

Review: "Interface is confusing at first but powerful once you get used to it. 
Customer service was unhelpful when I called with questions."
Sentiment:
"""

## zero shot CoT
prompt: str = """
Solve the following math problem step by step, showing your reasoning for each step:
 A store received a shipment of 240 items. 
 They sold 3/8 of the items on the first day and 1/4 of the remaining items on the second day. 
 How many items were left after the second day?
"""

## few shot CoT
prompt: str = """
Problem 1:
A store has 120 apples. If they sell 2/5 of the apples on Saturday and 1/3 of the remaining apples on Sunday, how many apples do they have left?

Step-by-step solution:
1) First, I need to find how many apples were sold on Saturday.
   2/5 of 120 apples = 2/5 × 120 = 48 apples sold on Saturday.

2) Next, I need to find how many apples remained after Saturday.
   120 - 48 = 72 apples remained after Saturday.

3) Then, I need to find how many apples were sold on Sunday.
   1/3 of the remaining 72 apples = 1/3 × 72 = 24 apples sold on Sunday.

4) Finally, I need to find how many apples remained after Sunday.
   72 - 24 = 48 apples remained after Sunday.

Therefore, the store has 48 apples left.

Problem 2:
A store received a shipment of 240 items. They sold 3/8 of the items on the first day and 1/4 of the remaining items on the second day. How many items were left after the second day?

Step-by-step solution:
"""

## carefully designed reasoning instructions
prompt: str = """
Solve this problem using the following approach:
1. Identify the key variables and constraints
2. Set up relevant equations
3. Solve the equations step by step
4. Verify your solution against the original constraints
5. State the final answer clearly

Problem: A rectangular garden has a perimeter of 100 meters and an area of 600 square meters. What are the dimensions of the garden?
"""


## standard self-consistency
prompt: str = """
A factory produces 120 units per hour. It runs for 8 hours per day. 
How many total units will it produce in 5 days? Provide only the numerical answer.
"""

# ------------------------------------------
# Multiple Responses Generation
# ------------------------------------------
# Generate multiple LLM responses and determine the most common answer.
num_samples = 10
responses = [get_llm_response(prompt) for _ in range(num_samples)]

# Import Counter to count the frequency of responses
from collections import Counter

answer_counts = Counter(responses)
most_common_answer = answer_counts.most_common(1)[0]

print(
    f"Most Common Answer: {most_common_answer[0]} (Appeared {most_common_answer[1]} times)"
)

# ------------------------------------------
# Weighted Self Consistency
# ------------------------------------------
prompt: str = """
A person takes out a $50,000 loan with a 5% annual interest rate for 10 years.
What is their monthly payment using the standard loan amortization formula?
Provide the numerical answer only.
"""

# Generate multiple responses, cleaning each response to convert to float.
num_samples = 10
responses = []
for _ in range(num_samples):
    response = get_llm_response(prompt)
    try:
        # Remove '$' and commas, then convert to float.
        num_response = float(response.replace("$", "").replace(",", "").strip())
        responses.append(num_response)
    except ValueError:
        pass  # Ignore invalid responses


# Function to calculate the true monthly payment using loan amortization formula.
def calculate_true_payment(P, r, n):
    return (P * r * (1 + r) ** n) / ((1 + r) ** n - 1)


# Define loan parameters.
P = 50000
r = 0.05 / 12  # Monthly interest rate
n = 10 * 12  # Total number of monthly payments
correct_value = calculate_true_payment(P, r, n)

# Compute weights for each response based on deviation from the correct value.
from collections import defaultdict
import numpy as np

weights = defaultdict(float)
for response in responses:
    deviation = abs(correct_value - response)
    weight = np.exp(
        -deviation / 10
    )  # Exponential weighting; lower deviation gives higher weight.
    weights[response] += weight

# Select and print the best weighted response.
best_answer = max(weights, key=weights.get)
print(f"Weighted Best Answer: ${best_answer:.2f} (Score: {weights[best_answer]:.4f})")

# ------------------------------------------
# Self Verification: Factorial Function
# ------------------------------------------
prompt: str = """
Write a Python function to compute the factorial of a number. Do not use the built-in factorial function.
"""


# Example of two possible AI-generated responses:
# Response 1:
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)


# Response 2:
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


# Prompt for only the function definition.
prompt = """
Write a Python function to compute the factorial of a number. Do not use the built-in factorial function.
Return only the function definition.
"""

# Generate multiple responses.
num_samples = 5
responses = [get_llm_response(prompt) for _ in range(num_samples)]

# Define test cases to verify the correctness of the factorial function.
test_cases = {0: 1, 1: 1, 5: 120, 7: 5040, 10: 3628800}


def verify_code(code):
    """
    Executes the provided code and verifies correctness against test cases.
    Returns a confidence score between 0 and 1.
    """
    try:
        # Compile and execute the generated code in global namespace.
        parsed_code = compile(code, "<string>", "exec")
        exec(parsed_code, globals())
        factorial_fn = globals().get("factorial")

        # Check if the function exists and is callable.
        if not callable(factorial_fn):
            return 0

        # Evaluate the function against each test case.
        correct_cases = sum(1 for x, y in test_cases.items() if factorial_fn(x) == y)
        confidence_score = correct_cases / len(test_cases)
        return confidence_score
    except Exception:
        return 0  # Return zero confidence if execution fails.


# Verify all responses and select the best verified function.
scores = [(response, verify_code(response)) for response in responses]
best_response, best_score = max(scores, key=lambda x: x[1])
print(f"Best Verified Function (Score: {best_score}):\n{best_response}")


# A final, fully commented factorial function with error handling.
def factorial(n):
    """
    Computes the factorial of a given number using recursion.

    Parameters:
    n (int): A non-negative integer.

    Returns:
    int: The factorial of the given number.

    Raises:
    ValueError: If n is negative or not an integer.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer.")
    if n == 0:
        return 1
    return n * factorial(n - 1)


# Example usage:
print(factorial(5))  # Expected output: 120
print(factorial(7))  # Expected output: 5040


# ------------------------------------------
# Diversity Sampling Functions
# ------------------------------------------
# Function to generate a response with adjustable temperature.
def generate_response(prompt, temperature=0.7):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,  # Controls randomness.
        max_tokens=100,
    )
    return response["choices"][0]["message"]["content"].strip()


# Function to generate a response using top_p sampling.
def generate_response_top_p(prompt, top_p=0.9):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=top_p,  # Controls diversity of token selection.
        max_tokens=100,
    )
    return response["choices"][0]["message"]["content"].strip()


# Function to generate multiple responses via beam search.
def generate_summary(prompt, num_responses=3):
    responses = []
    for _ in range(num_responses):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50,
        )
        responses.append(response["choices"][0]["message"]["content"].strip())
    return responses


# ------------------------------------------
# Knowledge Generation Prompting
# ------------------------------------------
def decompose_query(query):
    """
    Decomposes a complex query into simpler sub-questions.
    Returns a list of sub-queries.
    """
    if "deep learning" in query and "genomics" in query:
        return [
            "What are the foundational principles of deep learning in genomics?",
            "How has deep learning contributed to breakthroughs in gene sequencing?",
            "What specific applications of convolutional neural networks have been used in cancer genomics?",
        ]
    else:
        return ["Please provide more context for decomposition."]


# Example usage of query decomposition.
query = "Discuss the role of deep learning in genomics research, highlighting breakthroughs in gene sequencing and expression analysis."
sub_queries = decompose_query(query)
print("Decomposed Queries:")
for idx, sub_query in enumerate(sub_queries, start=1):
    print(f"{idx}. {sub_query}")

# ------------------------------------------
# PAL: Prompt-assisted Learning – Basic
# ------------------------------------------
# Simple arithmetic demonstration with clear steps.
apples = 5  # John starts with 5 apples.
apples += 2  # John buys 2 more apples.
apples -= 3  # John gives 3 apples to his friend.
print(f"John has {apples} apples.")  # Final output.


# ------------------------------------------
# PAL 2: Prompt-assisted Learning with Execution
# ------------------------------------------
def solve_with_pal(problem, model_call_function, execution_function):
    """
    Generates and executes Python code to solve a given problem.

    Parameters:
    - problem: A string describing the problem.
    - model_call_function: Function to obtain code from the LLM.
    - execution_function: Function to execute the generated code.

    Returns:
    - Tuple containing execution result, code solution, and execution output.
    """
    code_prompt: str = f"""
    Translate the following problem into Python code that solves it.
    Write code that is concise, efficient, and prints the final answer.
    
    Problem: {problem}
    
    Python solution:
    ```python
    """
    # Obtain the code solution from the LLM.
    code_solution = model_call_function(code_prompt)
    code_solution = code_solution.strip()
    if "```" in code_solution:
        # Extract code from markdown formatting.
        code_solution = code_solution.split("```python")[1].split("```")[0].strip()

    # Execute the generated code and return the result.
    try:
        execution_result = execution_function(code_solution)
        return execution_result, code_solution, execution_result
    except Exception as e:
        return f"Error: {str(e)}", code_solution, str(e)


# ------------------------------------------
# React Process: Iterative Reasoning
# ------------------------------------------
def initial_reasoning(data):
    """
    Performs initial reasoning on input data.
    Computes the average and sets a hypothesis based on its sign.
    """
    avg = sum(data) / len(data)
    hypothesis = (
        "The data has a balanced distribution."
        if avg > 0
        else "The data skews negative."
    )
    return hypothesis, avg


def act_on_reasoning(avg):
    """
    Applies a corrective normalization action on the computed average.
    """
    adjusted_avg = (avg - min(0, avg)) * 1.1
    return adjusted_avg


def react_process(data):
    # Step 1: Initial reasoning.
    hypothesis, avg = initial_reasoning(data)
    print("Initial Hypothesis:", hypothesis)
    print("Computed Average:", avg)

    # Step 2: Act on the reasoning.
    adjusted_avg = act_on_reasoning(avg)
    print("Adjusted Average after action:", adjusted_avg)

    # Step 3: Refine hypothesis based on the adjustment.
    refined_hypothesis = (
        "The adjustment indicates a need for further normalization."
        if adjusted_avg > avg
        else "The initial reasoning holds."
    )
    return refined_hypothesis, adjusted_avg


# Simulate the process with sample data.
data_samples = [3, 5, -2, 4, 1]
final_hypothesis, final_result = react_process(data_samples)
print("Final Hypothesis:", final_hypothesis)


### CoVe
def cove_prompt(query):
    """
    Generate a Chain of Verification prompt for a given query.

    Args:
        query (str): The user's question or request

    Returns:
        str: A structured CoVe prompt
    """
    prompt = f"""
    Question: {query}
    
    Instructions:
    1. Generate an initial detailed answer to the question
    2. Identify 3-5 specific factual claims in your answer that should be verified
    3. For each claim, critically evaluate its accuracy based on your knowledge
    4. If any claims appear incorrect or uncertain, revise them with more accurate information
    5. Provide a final, revised answer that incorporates these verifications
    
    Format your response as follows:
    
    Initial Answer: [Your initial response]
    
    Claims to Verify:
    1. [First claim]
    2. [Second claim]
    ...
    
    Verification:
    1. [Verification of first claim]
    2. [Verification of second claim]
    ...
    
    Final Revised Answer: [Your revised, more accurate response]
    """

    return prompt


## CoD
def cod_prompt(query):
    """
    Generate a Chain of Density prompt for a given query.

    Args:
        query (str): The user's question or request

    Returns:
        str: A structured CoD prompt
    """
    prompt = f"""
    Question: {query}
    
    Instructions:
    You will generate increasingly detailed explanations through multiple iterations. 
    For each iteration:
    1. Start with a basic explanation (1-2 sentences)
    2. Identify specific areas where more detail would enhance understanding
    3. Add new, non-redundant information in each iteration
    4. Ensure each iteration approximately doubles the information density
    5. Maintain clarity and readability
    
    Format:
    
    Initial Draft:
    [Basic 1-2 sentence explanation]
    
    Iteration 1:
    [Enhanced explanation with 2x information density]
    
    Iteration 2:
    [Further enhanced explanation with 4x information density]
    
    Iteration 3:
    [Comprehensive explanation with 8x information density]
    
    Final Polished Response:
    [The most comprehensive explanation, optimized for both information density and readability]
    """

    return prompt


## Flare
def flare_generation(query, knowledge_base, model_call_function, max_iterations=3):
    """
    Implement Forward-Looking Active RAG for a given query.

    Args:
        query (str): The user's question
        knowledge_base (callable): Function to retrieve information from a knowledge base
        model_call_function (callable): Function to call the language model
        max_iterations (int): Maximum number of retrieval-generation cycles

    Returns:
        str: The final generated response
    """
    # Initial retrieval based on the query
    initial_context = knowledge_base(query)

    # Begin generating the response
    current_response = ""
    full_context = f"Query: {query}\n\nRelevant Information: {initial_context}\n\n"

    for i in range(max_iterations):
        # Generate the next segment of the response
        next_segment_prompt = f"""
        {full_context}
        Current Response: {current_response}
        
        Continue the response. If you need additional information to provide an accurate and complete answer, 
        explicitly state what information you need in the format: [NEED: specific information needed].
        Otherwise, continue generating the response.
        """

        next_segment = model_call_function(next_segment_prompt)

        # Check if additional information is needed
        if "[NEED:" in next_segment:
            # Extract the information need
            need_start = next_segment.find("[NEED:") + 7
            need_end = next_segment.find("]", need_start)
            info_need = next_segment[need_start:need_end].strip()

            # Retrieve additional information
            additional_info = knowledge_base(info_need)

            # Add to context
            full_context += (
                f"\nAdditional Information about '{info_need}':\n{additional_info}\n"
            )

            # Remove the [NEED: ...] marker from the segment
            next_segment = next_segment.replace(f"[NEED: {info_need}]", "")

        # Append the next segment to the current response
        current_response += next_segment

    # Final refinement pass
    final_prompt = f"""
    {full_context}
    Draft Response: {current_response}
    
    Please provide a final, polished response that integrates all the information coherently.
    Ensure factual accuracy and comprehensive coverage of the question.
    """

    final_response = model_call_function(final_prompt)
    return final_response
