"""Agent script to generate Chain of Thought (CoT) data for solving the equation '2x^2 - 5x - 3 = 0' using generator and verifier agents."""

from camel.agents import ChatAgent
from camel.datagen import CoTDataGenerator
import sympy


def compute_gold_answer():
    # Define the variable
    x = sympy.symbols('x')
    # Define the equation
    equation = sympy.Eq(2 * x**2 - 5 * x - 3, 0)
    # Solve the equation
    solutions = sympy.solve(equation, x)
    # Format the solutions as a string
    gold_answer = ', '.join([str(sol) for sol in solutions])
    return gold_answer


def main():
    question = 'Solve the equation 2x^2 - 5x - 3 = 0'

    # Compute the gold answer using sympy
    gold_answer = compute_gold_answer()

    # Prepare golden answers dictionary
    golden_answers = {question: gold_answer}

    # Initialize generator and verifier agents with system messages
    generator_system_message = (
        "You are a helpful assistant that solves math problems step-by-step."
    )
    verifier_system_message = (
        "You are a strict verifier that checks if the solution to the math problem is correct."
    )

    generator_agent = ChatAgent(system_message=generator_system_message)
    verifier_agent = ChatAgent(system_message=verifier_system_message)

    # Initialize CoT data generator with the two agents and golden answers
    cot_generator = CoTDataGenerator(
        generator_agent=generator_agent,
        verifier_agent=verifier_agent,
        golden_answers=golden_answers,
        search_limit=10,
    )

    # Use the CoT data generator to solve the question
    solution = cot_generator.solve(question)

    # Verify the solution
    is_correct = cot_generator.verify_answer(question, solution)

    print(f"Question: {question}")
    print(f"Gold Answer: {gold_answer}")
    print(f"Generated Solution:\n{solution}")
    print(f"Is the generated solution correct? {is_correct}")


if __name__ == '__main__':
    main()
