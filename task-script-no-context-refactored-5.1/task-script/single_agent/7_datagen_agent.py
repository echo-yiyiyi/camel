import sympy
from camel.datagen.cot_datagen import CoTDataGenerator
from camel.agents import ChatAgent

# Define the question
question = "Solve the equation 2x^2 - 5x - 3 = 0"

# Compute the gold answer using sympy
x = sympy.symbols('x')
equation = sympy.Eq(2*x**2 - 5*x - 3, 0)
solutions = sympy.solve(equation, x)
gold_answer = str(solutions)

def main():
    # Instantiate the generator agent with a system message for step-by-step solving
    generator_agent = ChatAgent(system_message="You are a helpful assistant that solves math problems step-by-step.")

    # Instantiate the verifier agent with a system message for verification
    verifier_agent = ChatAgent(system_message="You are a strict verifier. Given a question, a student answer, and a correct answer, respond with 'true' if the student answer is correct, otherwise 'false'.")

    # Create golden answers dictionary
    golden_answers = {question: gold_answer}

    # Instantiate the CoT data generator with generator and verifier agents
    cot_generator = CoTDataGenerator(
        generator_agent=generator_agent,
        verifier_agent=verifier_agent,
        golden_answers=golden_answers,
        search_limit=10,
    )

    # Solve the question using the CoT data generator
    solution = cot_generator.solve(question)

    print(f"Question: {question}")
    print(f"Gold Answer: {gold_answer}")
    print(f"Generated Solution:\n{solution}")


if __name__ == '__main__':
    main()
