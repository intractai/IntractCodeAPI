
import dspy
from dspy.signatures import Signature
from random import choice, randint, sample

# Define a custom signature for software engineering problem descriptions
class SEProblemSignature(Signature):
    """Generate self-contained software engineering problem descriptions."""
    topic = dspy.InputField()
    complexity = dspy.InputField()
    keywords = dspy.InputField()  # Adding keywords to ensure diversity
    description = dspy.OutputField(desc="A clear and concise problem statement for a single-script solution.")

# Configure the language model
lm = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=lm)

# Instantiate the signature
se_problem_signature = SEProblemSignature()

# Define topics, complexities, and a pool of programming keywords
topics = ['algorithms', 'data structures', 'design patterns', 'databases', 'networking']
complexities = ['easy', 'medium', 'hard']
programming_keywords = ['recursion', 'inheritance', 'polymorphism', 'hashing', 'caching', 'concurrency', 'asynchronous', 'API', 'data normalization', 'encryption']

class ProblemGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_description = dspy.ChainOfThought(SEProblemSignature())

    def forward(self, topic, complexity, keywords):
        return self.generate_description(topic=topic, complexity=complexity, keywords=keywords)

def generate_problem():
    # Randomly select a topic, complexity, and three programming keywords
    topic = choice(topics)
    complexity = choice(complexities)
    keywords = ', '.join(sample(programming_keywords, 3))  # Select three random keywords
    
    # Instantiate ProblemGenerator with the SEProblemSignature
    problem_generator = ProblemGenerator(se_problem_signature)
    
    # Generate the problem description
    problem_description = problem_generator(topic=topic, complexity=complexity, keywords=keywords)
    
    # Return the generated problem
    return {
        'topic': topic,
        'complexity': complexity,
        'keywords': keywords,
        'description': problem_description.description
    }

# Example usage
if __name__ == "__main__":
    problem = generate_problem()
    print(f"Topic: {problem['topic']}")
    print(f"Complexity: {problem['complexity']}")
    print(f"Keywords: {problem['keywords']}")
    print(f"Description:\n{problem['description']}")




