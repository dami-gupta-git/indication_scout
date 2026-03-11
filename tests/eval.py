from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric

# Define a test case
test_case = LLMTestCase(
    input="What is cancer immunotherapy?",
    actual_output="Cancer immunotherapy harnesses the immune system to fight tumors.",
    expected_output="Immunotherapy uses the body's immune system to target cancer cells.",
    context=["Immunotherapy is a type of cancer treatment that boosts the body's natural defenses."]
)

# Pick metrics
relevancy = AnswerRelevancyMetric(threshold=0.7)
hallucination = HallucinationMetric(threshold=0.5)

# Run evaluation
evaluate(test_cases=[test_case], metrics=[relevancy, hallucination])