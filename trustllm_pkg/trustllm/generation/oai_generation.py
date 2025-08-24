import time
import os
import json
import threading
import traceback
from enum import Enum
from typing import Optional, Union
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


class TestType(Enum):
    """Supported test types in TrustLLM."""

    ETHICS = "ethics"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    TRUTHFULNESS = "truthfulness"
    ROBUSTNESS = "robustness"
    SAFETY = "safety"


class EthicsDataset(Enum):
    """Ethics evaluation datasets."""

    AWARENESS = "awareness.json"
    EXPLICIT_MORALCHOICE = "explicit_moralchoice.json"
    IMPLICIT_ETHICS = "implicit_ETHICS.json"
    IMPLICIT_SOCIALCHEMISTRY101 = "implicit_SocialChemistry101.json"


class PrivacyDataset(Enum):
    """Privacy evaluation datasets."""

    PRIVACY_AWARENESS_CONFAIDE = "privacy_awareness_confAIde.json"
    PRIVACY_AWARENESS_QUERY = "privacy_awareness_query.json"
    PRIVACY_LEAKAGE = "privacy_leakage.json"


class FairnessDataset(Enum):
    """Fairness evaluation datasets."""

    DISPARAGEMENT = "disparagement.json"
    PREFERENCE = "preference.json"
    STEREOTYPE_AGREEMENT = "stereotype_agreement.json"
    STEREOTYPE_QUERY_TEST = "stereotype_query_test.json"
    STEREOTYPE_RECOGNITION = "stereotype_recognition.json"


class TruthfulnessDataset(Enum):
    """Truthfulness evaluation datasets."""

    EXTERNAL = "external.json"
    HALLUCINATION = "hallucination.json"
    GOLDEN_ADVFACTUALITY = "golden_advfactuality.json"
    INTERNAL = "internal.json"
    SYCOPHANCY = "sycophancy.json"


class RobustnessDataset(Enum):
    """Robustness evaluation datasets."""

    OOD_DETECTION = "ood_detection.json"
    OOD_GENERALIZATION = "ood_generalization.json"
    ADVGLUE = "AdvGLUE.json"
    ADVINSTRUCTION = "AdvInstruction.json"


class SafetyDataset(Enum):
    """Safety evaluation datasets."""

    JAILBREAK = "jailbreak.json"
    EXAGGERATED_SAFETY = "exaggerated_safety.json"
    MISUSE = "misuse.json"


# Type alias for dataset enums
DatasetType = Union[
    EthicsDataset,
    PrivacyDataset,
    FairnessDataset,
    TruthfulnessDataset,
    RobustnessDataset,
    SafetyDataset,
]


class OpenAILLMGeneration:
    def __init__(
        self,
        test_type: Union[TestType, str],
        data_path: str,
        model_name: str,
        dataset: Optional[Union[DatasetType, str]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_new_tokens: int = 512,
        debug: bool = False,
        group_size: int = 8,
    ):
        """
        Initialize OpenAI LLM Generation for TrustLLM evaluation.

        :param test_type: Type of test to run (TestType enum or string)
        :param data_path: Path to the TrustLLM dataset
        :param model_name: OpenAI model name (e.g., 'gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo')
        :param dataset: Optional specific dataset to run (DatasetType enum or string)
        :param api_key: OpenAI API key (or set api_key env var)
        :param base_url: Custom base URL for OpenAI-compatible APIs
        :param max_new_tokens: Maximum tokens to generate
        :param debug: Enable debug mode
        :param group_size: Number of parallel requests for processing
        """

        # Handle enum or string inputs
        self.test_type = (
            test_type.value if isinstance(test_type, TestType) else test_type
        )
        self.dataset = dataset.value if hasattr(dataset, "value") else dataset

        self.model_name = model_name
        self.data_path = data_path
        self.max_new_tokens = max_new_tokens
        self.debug = debug
        self.group_size = group_size

        # Validate test_type
        valid_tests = [t.value for t in TestType]
        if self.test_type not in valid_tests:
            raise ValueError(f"Invalid test_type. Must be one of: {valid_tests}")

        # Initialize OpenAI client
        api_key = api_key or os.getenv("api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set api_key environment variable or pass api_key parameter."
            )

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.openai_client = OpenAI(**client_kwargs)

        if self.debug:
            print(f"Initialized OpenAI client for model: {self.model_name}")
            print(f"Test type: {self.test_type}")
            if self.dataset:
                print(f"Specific dataset: {self.dataset}")

    @staticmethod
    def get_available_datasets(test_type: Union[TestType, str]) -> list:
        """Get available datasets for a specific test type."""
        test_name = test_type.value if isinstance(test_type, TestType) else test_type

        dataset_mapping = {
            TestType.ETHICS.value: list(EthicsDataset),
            TestType.PRIVACY.value: list(PrivacyDataset),
            TestType.FAIRNESS.value: list(FairnessDataset),
            TestType.TRUTHFULNESS.value: list(TruthfulnessDataset),
            TestType.ROBUSTNESS.value: list(RobustnessDataset),
            TestType.SAFETY.value: list(SafetyDataset),
        }

        return dataset_mapping.get(test_name, [])

    def _generation_openai(self, prompt, temperature=0.0):
        """
        Generate response using OpenAI API.

        :param prompt: Input prompt string
        :param temperature: Temperature for generation
        :return: Generated text
        """
        try:
            # Convert string prompt to messages format
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_new_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise e

    def process_element(self, element, index, temperature, key_name="prompt"):
        """
        Process a single data element.

        :param element: Dictionary containing the data point
        :param index: Index of the element
        :param temperature: Temperature for generation
        :param key_name: Key name for the prompt in the element
        """
        try:
            # Generate response if not already present
            if "res" not in element or not element["res"]:
                res = self._generation_openai(
                    prompt=element[key_name], temperature=temperature
                )
                element["res"] = res

                if self.debug:
                    print(f"Generated response for element {index}")

        except Exception as e:
            print(f"Error processing element at index {index}: {e}")
            if self.debug:
                traceback.print_exc()

    def process_file(self, data_path, save_path, file_config, key_name="prompt"):
        """
        Process a single file with multiple data points.

        :param data_path: Path to input data file
        :param save_path: Path to save processed results
        :param file_config: Configuration for file processing
        :param key_name: Key name for prompts in data
        """
        filename = os.path.basename(data_path)

        if filename not in file_config:
            print(f"{filename} not in file_config, skipping...")
            return

        # Load original data
        with open(data_path, "r") as f:
            original_data = json.load(f)

        # Load existing results if available
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                saved_data = json.load(f)
        else:
            saved_data = original_data.copy()

        # Process in groups for parallel execution
        for i in tqdm(
            range(0, len(saved_data), self.group_size),
            desc=f"Processing {filename}",
            leave=False,
        ):
            group_data = saved_data[i : i + self.group_size]
            threads = []

            # Create threads for parallel processing
            for idx, element in enumerate(group_data):
                temperature = file_config.get(filename, 0.0)

                thread = threading.Thread(
                    target=self.process_element,
                    args=(element, i + idx, temperature, key_name),
                )
                thread.start()
                threads.append(thread)

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Save progress after each group
            self._save_results(saved_data, save_path)

        # Final save
        self._save_results(saved_data, save_path)

    def _save_results(self, data, save_path):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    def _run_task(self, base_dir, file_config, key_name="prompt"):
        """
        Run evaluation task on files in a directory.

        :param base_dir: Base directory containing test files
        :param file_config: Configuration for file processing
        :param key_name: Key name for prompts
        """
        if not os.path.exists(base_dir):
            print(f"Directory {base_dir} does not exist, skipping...")
            return

        # Create results directory
        section = os.path.basename(base_dir)
        results_dir = os.path.join("generation_results", self.model_name, section)
        os.makedirs(results_dir, exist_ok=True)

        # If specific dataset is provided, only process that file
        if self.dataset:
            if self.dataset in file_config:
                filename = self.dataset
                data_path = os.path.join(base_dir, filename)
                save_path = os.path.join(results_dir, filename)

                if os.path.exists(data_path):
                    print(f"Processing specific dataset: {filename}")
                    self.process_file(data_path, save_path, file_config, key_name)
                else:
                    print(f"Dataset file {filename} not found in {base_dir}")
            else:
                print(
                    f"Dataset {self.dataset} not found in configuration for {section}"
                )
            return

        # Process all JSON files in the directory if no specific dataset
        json_files = [f for f in os.listdir(base_dir) if f.endswith(".json")]

        for filename in tqdm(json_files, desc=f"Processing {section} files"):
            data_path = os.path.join(base_dir, filename)
            save_path = os.path.join(results_dir, filename)

            self.process_file(data_path, save_path, file_config, key_name)

    def run_ethics(self):
        """Run ethics evaluation."""
        base_dir = os.path.join(self.data_path, "ethics")
        file_config = {
            "awareness.json": 0.0,
            "explicit_moralchoice.json": 1.0,
            "implicit_ETHICS.json": 0.0,
            "implicit_SocialChemistry101.json": 0.0,
        }
        self._run_task(base_dir, file_config)

    def run_privacy(self):
        """Run privacy evaluation."""
        base_dir = os.path.join(self.data_path, "privacy")
        file_config = {
            "privacy_awareness_confAIde.json": 0.0,
            "privacy_awareness_query.json": 1.0,
            "privacy_leakage.json": 1.0,
        }
        self._run_task(base_dir, file_config)

    def run_fairness(self):
        """Run fairness evaluation."""
        base_dir = os.path.join(self.data_path, "fairness")
        file_config = {
            "disparagement.json": 1.0,
            "preference.json": 1.0,
            "stereotype_agreement.json": 1.0,
            "stereotype_query_test.json": 1.0,
            "stereotype_recognition.json": 0.0,
        }
        self._run_task(base_dir, file_config)

    def run_truthfulness(self):
        """Run truthfulness evaluation."""
        base_dir = os.path.join(self.data_path, "truthfulness")
        file_config = {
            "external.json": 0.0,
            "hallucination.json": 0.0,
            "golden_advfactuality.json": 1.0,
            "internal.json": 1.0,
            "sycophancy.json": 1.0,
        }
        self._run_task(base_dir, file_config)

    def run_robustness(self):
        """Run robustness evaluation."""
        base_dir = os.path.join(self.data_path, "robustness")
        file_config = {
            "ood_detection.json": 1.0,
            "ood_generalization.json": 0.0,
            "AdvGLUE.json": 0.0,
            "AdvInstruction.json": 1.0,
        }
        self._run_task(base_dir, file_config)

    def run_safety(self):
        """Run safety evaluation."""
        base_dir = os.path.join(self.data_path, "safety")
        file_config = {
            "jailbreak.json": 1.0,
            "exaggerated_safety.json": 1.0,
            "misuse.json": 1.0,
        }
        self._run_task(base_dir, file_config)

    def _run_single_test(self):
        """Execute the specified test type."""
        print(
            f"Beginning generation with {self.test_type} evaluation using OpenAI API."
        )
        print(f"Model: {self.model_name}")

        test_functions = {
            "robustness": self.run_robustness,
            "truthfulness": self.run_truthfulness,
            "fairness": self.run_fairness,
            "ethics": self.run_ethics,
            "safety": self.run_safety,
            "privacy": self.run_privacy,
        }

        test_func = test_functions.get(self.test_type)
        if test_func:
            test_func()
            return "OK"
        else:
            print(
                f"Invalid test_type '{self.test_type}'. Valid options: {list(test_functions.keys())}"
            )
            return None

    def generation_results(self, max_retries=10, retry_interval=3):
        """
        Main function to run the evaluation with retries.
        Compatible with original TrustLLM interface.

        :param max_retries: Maximum number of retry attempts
        :param retry_interval: Seconds to wait between retries
        :return: Result status
        """
        if not os.path.exists(self.data_path):
            print(f"Dataset path {self.data_path} does not exist.")
            return None

        for attempt in range(max_retries):
            try:
                result = self._run_single_test()
                if result:
                    print(f"Test function successful on attempt {attempt + 1}")
                    return result

            except Exception as e:
                print(f"Test function failed on attempt {attempt + 1}")
                if self.debug:
                    traceback.print_exc()

                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)

        print("Test failed after maximum retries.")
        return None
