"""
Agent with CoT data generation tool with generator and verifier agents inside to generate CoT data.
Solves '2x^2-5x-3=0' with gold answer computed via sympy.
Uses generator and verifier agents with Qwen2.5-14B-Instruct model.
"""

from sympy import symbols, Eq, solve
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Define the problem
problem = "Solve the equation 2x^2 - 5x - 3 = 0"

# Compute the gold answer using sympy
x = symbols('x')
equation = Eq(2*x**2 - 5*x - 3, 0)
solutions = solve(equation, x)
gold_answer = ', '.join([str(sol.evalf()) for sol in solutions])

# Create generator and verifier agents with Qwen2.5-14B-Instruct
model_generator = ModelFactory.create(model_platform=ModelPlatformType.QWEN, model_type=ModelType.QWEN_2_5_14B)
model_verifier = ModelFactory.create(model_platform=ModelPlatformType.QWEN, model_type=ModelType.QWEN_2_5_14B)

generator_agent = ChatAgent(model=model_generator)
verifier_agent = ChatAgent(model=model_verifier)

# Generate CoT solution from generator agent
prompt_generate = f"Question: {problem}\nPlease provide a detailed chain-of-thought solution."
generated_response = generator_agent.step(prompt_generate)
generated_cot = generated_response.msgs[-1].content

# Verify the generated CoT solution with verifier agent
prompt_verify = f"Question: {problem}\nGenerated solution: {generated_cot}\nIs the solution correct? Answer yes or no and explain."
verification_response = verifier_agent.step(prompt_verify)
verification_result = verification_response.msgs[-1].content

# Extract final answer from generated CoT (simple heuristic: last line containing '=' or 'answer')
final_answer = None
for line in reversed(generated_cot.splitlines()):
    if '=' in line.lower() or 'answer' in line.lower():
        final_answer = line.strip()
        break
if final_answer is None:
    final_answer = "Could not extract final answer"

# Print results
print("Problem:", problem)
print("Gold answer:", gold_answer)
print("Generated CoT:", generated_cot)
print("Verification result:", verification_result)
print("Final answer:", final_answer)

# Save results to a file
output_path = "task-script/single_agent/7_datagen_agent_output.txt"
with open(output_path, "w") as f:
    f.write(f"Problem: {problem}\n")
    f.write(f"Gold answer: {gold_answer}\n")
    f.write(f"Generated CoT: {generated_cot}\n")
    f.write(f"Verification result: {verification_result}\n")
    f.write(f"Final answer: {final_answer}\n")

print(f"Results saved to {output_path}")
