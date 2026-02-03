import sympy
from camel.agents import ChatAgent
from camel.datagen.cot_datagen import CoTDataGenerator
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Define the question
question = "Solve the quadratic equation 2x^2 - 5x - 3 = 0"

# Compute the gold answer using sympy
x = sympy.symbols('x')
equation = 2*x**2 - 5*x - 3
solutions = sympy.solve(equation, x)
gold_answer = ', '.join([str(sol.evalf()) for sol in solutions])

# Create generator and verifier agents
system_message_generator = "You are a helpful assistant that solves math problems step-by-step."
system_message_verifier = "You are a strict verifier that checks if the answer is correct."

model_generator = ModelFactory.create(
    model_platform=ModelPlatformType.TOGETHER,
    model_type=ModelType.TOGETHER_LLAMA_3_1_8B,
    model_config_dict={"temperature": 0}
)

model_verifier = ModelFactory.create(
    model_platform=ModelPlatformType.TOGETHER,
    model_type=ModelType.TOGETHER_LLAMA_3_1_8B,
    model_config_dict={"temperature": 0}
)

generator_agent = ChatAgent(system_message_generator, model=model_generator)
verifier_agent = ChatAgent(system_message_verifier, model=model_verifier)

# Prepare golden answers dictionary
golden_answers = {question: gold_answer}

# Create CoTDataGenerator instance
cot_generator = CoTDataGenerator(
    generator_agent=generator_agent,
    verifier_agent=verifier_agent,
    golden_answers=golden_answers,
    search_limit=10
)

# Generate solution
try:
    solution = cot_generator.solve(question)
except Exception as e:
    solution = f"Error during solution generation: {e}"

print("Question:", question)
print("Gold Answer:", gold_answer)
print("Generated Solution:", solution)

# Verify the generated solution
try:
    is_correct = cot_generator.verify_answer(question, solution)
    print("Is the generated solution correct?", is_correct)
except Exception as e:
    print("Verification failed with error:", e)

# Export solutions to a file
try:
    cot_generator.export_solutions('task-script/single_agent/7_datagen_agent_solutions.json')
    print("Solutions exported to task-script/single_agent/7_datagen_agent_solutions.json")
except Exception as e:
    print("Failed to export solutions with error:", e)
