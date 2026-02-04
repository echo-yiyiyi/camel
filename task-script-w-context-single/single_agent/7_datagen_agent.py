import sympy
from camel.agents import ChatAgent
from camel.datagen import CoTDataGenerator
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Define the question
question = "2*x**2 - 5*x - 3 = 0"

# Compute the gold answer using sympy
x = sympy.symbols('x')
expr = 2*x**2 - 5*x - 3
solutions = sympy.solve(expr, x)
golden_answer = str(solutions)

def main():
    # Create models for generator and verifier agents
    generator_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )
    verifier_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )

    # Create system messages
    generator_system_message = "You are a helpful assistant that generates step-by-step reasoning and solutions."
    verifier_system_message = "You are a strict verifier that checks if the answer is correct."

    # Create ChatAgent instances
    generator_agent = ChatAgent(system_message=generator_system_message, model=generator_model)
    verifier_agent = ChatAgent(system_message=verifier_system_message, model=verifier_model)

    # Prepare golden answers dictionary
    golden_answers = {question: golden_answer}

    # Instantiate CoTDataGenerator
    cot_generator = CoTDataGenerator(
        generator_agent=generator_agent,
        verifier_agent=verifier_agent,
        golden_answers=golden_answers,
        search_limit=10,
    )

    # Solve the question
    solution = cot_generator.solve(question)

    print("Question:", question)
    print("Golden Answer:", golden_answer)
    print("Generated Solution:", solution)


if __name__ == "__main__":
    main()
