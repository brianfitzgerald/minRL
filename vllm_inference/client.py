# Copied from TRL
import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests
import torch
from requests import ConnectionError
from torch import nn
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from fire import Fire

logger = logging.getLogger(__name__)


@dataclass
class GenerateResponse:
    completion_ids: list[list[int]]
    generated_logprobs: list[list[dict[int, float]]] | None = None


class VLLMClient:
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions and update model weights in a single GPU setup.
    Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient
        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.update_model_params(model)
        ```
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        connection_timeout: float = 0.0,
    ):
        self.session = requests.Session()
        self.host = host
        self.server_port = server_port
        self.check_server(connection_timeout)  # check server and fail after timeout

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} "
                        "seconds."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return None

            # Retry logic: wait before trying again
            logger.info(
                f"Server is not up yet. Retrying in {retry_interval} seconds..."
            )
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: list[str | list[int]],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
    ) -> GenerateResponse:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """
        url = f"http://{self.host}:{self.server_port}/generate/"
        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
                "logprobs": logprobs,
                "prompt_logprobs": prompt_logprobs,
            },
        )
        if response.status_code == 200:
            response_json = response.json()
            response = GenerateResponse(completion_ids=response_json["completion_ids"])
            if "generated_logprobs" in response_json:
                response.generated_logprobs = response_json["generated_logprobs"]
            return response
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"http://{self.host}:{self.server_port}/update_named_param/"
        response = self.session.post(
            url, json={"name": name, "dtype": dtype, "shape": shape}
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            # Update each parameter individually
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"http://{self.host}:{self.server_port}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")


def main(use_token_ids: bool = False, update_params: bool = False):
    client = VLLMClient()

    for _ in range(10):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        prompt: list[str | list[int]] = []
        if use_token_ids:
            prompt = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Hello, AI!"}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            ]
        else:
            prompt = ["Hello, AI!"]
        responses = client.generate(prompt, max_tokens=256)
        for response in responses.completion_ids:
            logger.info(tokenizer.decode(response))

    if update_params:
        device = "cuda" if torch.cuda.is_available() else "mps"
        logger.info(f"Loading model on device: {device}")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)
        logger.info("updating model params")
        client.update_model_params(model)
        logger.info("done")


if __name__ == "__main__":
    Fire(main)
