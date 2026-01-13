import os

os.environ["LOG_LEVEL"] = "DEBUG"

from minisgl.core import SamplingParams
from minisgl.llm import LLM
from minisgl.utils.logger import init_logger

logger = init_logger(__name__)

OUTPUT_LEN = 3


def main():
    llm = LLM("Qwen/Qwen3-0.6B", max_running_req=4, memory_ratio=0.3, disable_cuda_graph=True)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_LEN)
    prompts = [
        # "List the first ten prime numbers:",
        "The capital of France is",
        # "Once upon a time in a land far, far away,",
        # "List 10 numbers only contains digit 1:",
    ]

    # Generate completions for the prompts
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        logger.info("Prompt: %r", prompt)
        logger.info("Completion: %s <%s>", output["text"], output["token_ids"])


if __name__ == "__main__":
    main()
