from trustllm import config
from trustllm.task import robustness
from trustllm.utils import file_process
from trustllm.generation.oai_generation import (
    OpenAILLMGeneration,
    TestType,
    RobustnessDataset,
    EthicsDataset,
    FairnessDataset,
    SafetyDataset,
    PrivacyDataset,
    TruthfulnessDataset,
)

## Example 1: Using OpenAI GPT-4
llm_gen_openai = OpenAILLMGeneration(
    test_type=TestType.ROBUSTNESS,
    dataset=RobustnessDataset.ADVGLUE,
    data_path="../dataset/dataset",
    model_name="gpt-4o-mini",
    api_key="sk-proj-xxx",  # or set OPENAI_API_KEY environment variable
    base_url="https://api.openai.com/v1",
    max_new_tokens=1024,
    max_rows=20,
    debug=False,
)

llm_gen_openai.generation_results()

config.openai_key = "sk-proj-xxx"
generated_results = "generation_results/gpt-4o-mini"
evaluator = robustness.RobustnessEval()
advglue_data = file_process.load_json(f"{generated_results}/robustness/AdvGLUE.json")
results = evaluator.advglue_eval(advglue_data)
print(results)
