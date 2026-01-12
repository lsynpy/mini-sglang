import os

os.environ["LOG_LEVEL"] = "DEBUG"

from minisgl.core import SamplingParams
from minisgl.llm import LLM
from minisgl.utils.logger import init_logger

logger = init_logger(__name__)


def main():
    llm = LLM("Qwen/Qwen3-0.6B", max_running_req=4, memory_ratio=0.3)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=8)

    print("\n-------- first batch ----------\n")
    prompts1 = [
        "The capital of France is",
    ]

    outputs1 = llm.generate(prompts1, sampling_params)

    for prompt, output in zip(prompts1, outputs1):
        print(f"\nPrompt: {repr(prompt)}")
        print(f"Completion: {output['text']}")

    print("\n-------- second batch ----------\n")
    prompts2 = [
        "The capital of China is",
    ]

    outputs2 = llm.generate(prompts2, sampling_params)

    for prompt, output in zip(prompts2, outputs2):
        print(f"\nPrompt: {repr(prompt)}")
        print(f"Completion: {output['text']}")

    print("\n-------- third batch ----------\n")
    prompts3 = [
        "The capital of China is a",
        "The capital of China is an",
        "The capital of China is pretty",
    ]

    outputs3 = llm.generate(prompts3, sampling_params)

    for prompt, output in zip(prompts3, outputs3):
        print(f"\nPrompt: {repr(prompt)}")
        print(f"Completion: {output['text']}")


if __name__ == "__main__":
    main()
