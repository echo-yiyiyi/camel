from camel.agents import ChatAgent
from camel.datagen.cot_datagen import CoTDataGenerator
from camel.models import ModelFactory
from camel.toolkits import SymPyToolkit
from camel.types import ModelType
import sympy as sp

# Define the question
question = '2*x**2 - 5*x - 3 = 0'

# Compute the gold answer using sympy
x = sp.symbols('x')
equation = sp.Eq(2*x**2 - 5*x - 3, 0)
solutions = sp.solve(equation, x)
gold_answer = str(solutions)

# Create model
model = ModelFactory.create(
    model_type=ModelType.DEFAULT,
)

# Create generator and verifier agents
sys_msg_generator = "You are a helpful assistant that thinks step by step to solve math problems."
sys_msg_verifier = "You are a strict verifier that checks if the answer is correct."

# Add SymPy tools to generator agent
tools = SymPyToolkit().get_tools()

generator_agent = ChatAgent(
    system_message=sys_msg_generator,
    model=model,
    tools=tools,
)

verifier_agent = ChatAgent(
    system_message=sys_msg_verifier,
    model=model,
)

# Create CoTDataGenerator with generator and verifier agents and golden answer
cot_generator = CoTDataGenerator(
    generator_agent=generator_agent,
    verifier_agent=verifier_agent,
    golden_answers={question: gold_answer},
)

# Solve the question
solution = cot_generator.solve(question)

print("Question:", question)
print("Generated solution:", solution)
print("Gold answer:", gold_answer)
