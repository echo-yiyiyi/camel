from sympy import symbols, Eq, solve as sympy_solve
from camel.agents import ChatAgent
from camel.datagen.cot_datagen import CoTDataGenerator

# Define the question
question = "Solve 2x^2-5x-3=0"

# Use sympy to compute the golden answer
x = symbols('x')
equation = Eq(2*x**2 - 5*x - 3, 0)
roots = sympy_solve(equation, x)
golden_answer = ', '.join([str(root.evalf()) for root in roots])

# Prepare golden answers dictionary
golden_answers = {question: golden_answer}

# Create mock ChatAgent classes for generator and verifier
# For demonstration, we create simple dummy agents that return fixed responses
class DummyGeneratorAgent(ChatAgent):
    def reset(self):
        pass
    def step(self, prompt, response_format=None):
        # Return a detailed step-by-step solution for the quadratic
        content = (
            "Step 1: Identify coefficients a=2, b=-5, c=-3.\n"
            "Step 2: Calculate discriminant D = b^2 - 4ac = (-5)^2 - 4*2*(-3) = 25 + 24 = 49.\n"
            "Step 3: Since D > 0, two real roots exist.\n"
            "Step 4: Roots are (-b +- sqrt(D)) / (2a) = (5 +- 7) / 4.\n"
            "Step 5: Root1 = (5 + 7)/4 = 12/4 = 3.\n"
            "Step 6: Root2 = (5 - 7)/4 = -2/4 = -0.5.\n"
            "Final answer: 3, -0.5"
        )
        class Msg:
            def __init__(self, content):
                self.content = content
        class Response:
            def __init__(self, content):
                self.msgs = [Msg(content)]
        return Response(content)

class DummyVerifierAgent(ChatAgent):
    def reset(self):
        pass
    def step(self, prompt, response_format=None):
        # Check if the answer contains the golden answer roots approximately
        # For simplicity, just check if '3' and '-0.5' are in the answer
        is_correct = '3' in prompt and '-0.5' in prompt
        class Parsed:
            def __init__(self, is_correct):
                self.is_correct = is_correct
        class Msg:
            def __init__(self, parsed):
                self.parsed = parsed
        class Response:
            def __init__(self, is_correct):
                self.msgs = [Msg(Parsed(is_correct))]
        return Response(is_correct)

# Instantiate agents
generator_agent = DummyGeneratorAgent()
verifier_agent = DummyVerifierAgent()

# Create CoTDataGenerator
cot_generator = CoTDataGenerator(
    generator_agent=generator_agent,
    verifier_agent=verifier_agent,
    golden_answers=golden_answers,
    search_limit=10
)

# Generate solution
solution = cot_generator.solve(question)
print("Generated solution:\n", solution)

# Verify solution
is_correct = cot_generator.verify_answer(question, solution)
print("Is the solution correct?", is_correct)
