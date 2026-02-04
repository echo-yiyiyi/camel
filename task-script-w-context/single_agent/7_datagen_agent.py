import sympy
from sympy import Eq, symbols, solve

from camel.agents import ChatAgent
from camel.datagen import CoTDataGenerator
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Define the question
question = "2x^2-5x-3=0"

# Compute the gold answer using sympy
x = symbols('x')
equation = Eq(2*x**2 - 5*x - 3, 0)
solutions = solve(equation, x)
gold_answer = str(solutions)

golden_answers = {question: gold_answer}

# Create generator and verifier agents
system_message_generator = "You are a helpful assistant that solves math problems step-by-step."
system_message_verifier = "You are a strict verifier that checks if the answer is correct."

generator_agent = ChatAgent(system_message=system_message_generator, model=ModelFactory.create(ModelPlatformType.DEFAULT, ModelType.DEFAULT))
verifier_agent = ChatAgent(system_message=system_message_verifier, model=ModelFactory.create(ModelPlatformType.DEFAULT, ModelType.DEFAULT))

# Create CoTDataGenerator with generator and verifier agents
cot_generator = CoTDataGenerator(
    generator_agent=generator_agent,
    verifier_agent=verifier_agent,
    golden_answers=golden_answers,
    search_limit=10,
)

# Solve the question
solution = cot_generator.solve(question)

print(f"Question: {question}")
print(f"Gold answer: {gold_answer}")
print(f"Generated solution:\n{solution}")

# Export solutions to a JSON file
cot_generator.export_solutions('cot_solutions.json')
print("Solutions exported to cot_solutions.json")
