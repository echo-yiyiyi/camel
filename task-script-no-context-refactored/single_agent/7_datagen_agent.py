import sympy
from camel.agents import ChatAgent
from camel.datagen.cot_datagen import CoTDataGenerator
from camel.verifiers.math_verifier import MathVerifier

# Define the question
question = "Solve the equation: 2*x**2 - 5*x - 3 = 0"

# Compute the golden answer using sympy
x = sympy.symbols('x')
equation = sympy.Eq(2*x**2 - 5*x - 3, 0)
solutions = sympy.solve(equation, x)
golden_answer = str(solutions)

# Initialize generator agent (using ChatAgent with a simple system message)
generator_agent = ChatAgent(system_message="You are a helpful math problem solver.")

# Initialize verifier agent using MathVerifier
verifier_agent = ChatAgent(system_message="You are a math verifier. Verify if the answer is correct.")

# Prepare golden answers dictionary
golden_answers = {question: golden_answer}

# Create CoTDataGenerator instance
cot_generator = CoTDataGenerator(
    generator_agent=generator_agent,
    verifier_agent=verifier_agent,
    golden_answers=golden_answers,
    search_limit=10
)

# Generate solution
solution = cot_generator.solve(question)

print("Question:", question)
print("Golden Answer:", golden_answer)
print("Generated Solution:", solution)
