import json
from camel.agents import ChatAgent
from camel.datagen.cot_datagen import CoTDataGenerator
from camel.toolkits.sympy_toolkit import SymPyToolkit
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Create sympy toolkit instance
sympy_toolkit = SymPyToolkit()

# Compute the gold answer for the equation '2x^2-5x-3=0' using sympy
equation = '2*x**2 - 5*x - 3'
# Use sympy toolkit's solve_equation method
gold_answer_json = sympy_toolkit.solve_equation(equation, variable='x')
gold_answer_data = json.loads(gold_answer_json)
gold_answer = gold_answer_data.get('result', [])

# Format gold answer as string for comparison
# Join solutions with comma
formatted_gold_answer = ', '.join(gold_answer)

# Define the question
question = 'Solve the equation 2x^2 - 5x - 3 = 0'

# Create generator and verifier agents
# Use default model platform and type
generator_agent = ChatAgent(system_message="You are a helpful math problem solver. Please solve step by step.")
verifier_agent = ChatAgent(system_message="You are a strict verifier. Verify if the answer is correct.")

# Create golden answers dictionary
golden_answers = {question: formatted_gold_answer}

# Create CoTDataGenerator with generator and verifier agents
cot_generator = CoTDataGenerator(
    generator_agent=generator_agent,
    verifier_agent=verifier_agent,
    golden_answers=golden_answers,
    search_limit=10,
)

# Generate the CoT data (solve the question)
solution = cot_generator.solve(question)

# Verify the solution
is_correct = cot_generator.verify_answer(question, solution)

# Print results
print(f"Question: {question}")
print(f"Gold Answer: {formatted_gold_answer}")
print(f"Generated Solution:\n{solution}")
print(f"Is the generated solution correct? {is_correct}")

# Optionally export solutions to a file
cot_generator.export_solutions('cot_solutions.json')
