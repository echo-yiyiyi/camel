from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.datagen.cot_datagen import CoTDataGenerator
from camel.types import ModelType


def main():
    # Define the question
    question = 'Solve the equation 2x^2 - 5x - 3 = 0'

    # Compute the gold answer using sympy (precomputed here for simplicity)
    gold_answer = "x = 3, x = -0.5"

    # Create generator and verifier agents
    model = ModelFactory.create(model_type=ModelType.GPT_4O_MINI)
    generator_agent = ChatAgent(system_message="You are a helpful generator agent.", model=model)
    verifier_agent = ChatAgent(system_message="You are a helpful verifier agent.", model=model)

    # Create CoTDataGenerator with generator and verifier agents
    cot_generator = CoTDataGenerator(
        generator_agent=generator_agent,
        verifier_agent=verifier_agent,
        golden_answers={question: gold_answer},
        search_limit=10,
    )

    # Solve the question using the CoTDataGenerator
    solution = cot_generator.solve(question)

    print("Question:", question)
    print("Generated solution:", solution)


if __name__ == '__main__':
    main()
