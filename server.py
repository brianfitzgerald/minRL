# Copied from TRL
import logging
import os
import signal
import sys
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, RequestOutput, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.utils import get_open_port

from minrl.constants import TrainerConfig

logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)


class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and server worker.
    This version works with a single GPU and uses direct tensor operations instead of NCCL.
    """

    def __init__(self):
        self.device = torch.device("cuda")
        self.model_runner = None

    def update_named_param(
        self, name: str, dtype: torch.dtype, shape: Sequence[int]
    ) -> None:
        """
        Receives updated weights and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`torch.dtype`):
                Data type of the weight tensor (e.g., `torch.float32`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        if self.model_runner is None:
            raise RuntimeError("Model runner not initialized")

        # Allocate memory for the incoming weight tensor on the correct device
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Load the received weights into the model
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        """
        Closes the weight update mechanism. In single GPU mode, this is a no-op.
        """
        pass


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
        enforce_eager (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the
            model in eager mode. If `False` (default behavior), we will use CUDA graph and eager execution in hybrid.
        log_level (`str`, *optional*, defaults to `"info"`):
            Log level for uvicorn. Possible choices: `"critical"`, `"error"`, `"warning"`, `"info"`, `"debug"`,
            `"trace"`.
    """

    model: str = field(
        metadata={"help": "Model name or path to load the model from."},
    )
    revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "Revision to use for the model. If not specified, the default branch will be used."
        },
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.4,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: Optional[int] = field(
        default=4096,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    enforce_eager: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always "
            "execute the model in eager mode. If `False` (default behavior), we will use CUDA graph and eager "
            "execution in hybrid."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', "
            "'trace'."
        },
    )


def llm_worker(
    script_args: ScriptArguments,
    data_parallel_rank: int,
    master_port: int,
    connection: Connection,
) -> None:
    # Set required environment variables for DP to work with vLLM
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    def signal_handler(signum, frame):
        logger.info(
            f"Worker {data_parallel_rank} received interrupt signal. Shutting down..."
        )
        try:
            llm.collective_rpc(method="close_communicator")
        except Exception as e:
            logger.error(f"Error closing communicator: {e}")
            pass
        os._exit(0)

    # Register signal handler for worker
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        enforce_eager=script_args.enforce_eager,
        dtype=script_args.dtype,
        enable_prefix_caching=script_args.enable_prefix_caching,
        max_model_len=script_args.max_model_len,
        worker_extension_cls="server.WeightSyncWorkerExtension",
    )

    # Send ready signal to parent process
    connection.send({"status": "ready"})

    try:
        while True:
            # Wait for commands from the parent process
            try:
                command = connection.recv()
            except (EOFError, ConnectionError):
                break

            # Handle commands
            if command["type"] in ["call", "fire_and_forget"]:
                method_name = command["method"]
                args, kwargs = command.get("args", ()), command.get("kwargs", {})
                method = getattr(llm, method_name)
                result = method(*args, **kwargs)
                if command["type"] == "call":
                    connection.send(result)
            elif command["type"] == "shutdown":
                break
    finally:
        try:
            llm.collective_rpc(method="close_communicator")
        except Exception as e:
            logger.error(f"Error closing communicator: {e}")
            pass
        connection.close()


def main(script_args: ScriptArguments):
    # Spawn dp workers, and setup pipes for communication
    master_port = get_open_port()
    connections = []
    processes = []

    def signal_handler(signum, frame):
        logger.info("Received interrupt signal. Shutting down...")
        # Send shutdown command to all workers
        for connection in connections:
            try:
                connection.send({"type": "shutdown"})
            except Exception as e:
                logger.error(f"Error sending shutdown command: {e}")
                pass
        # Terminate all processes
        for process in processes:
            if process.is_alive():
                process.terminate()
        # Wait for processes to terminate
        for process in processes:
            process.join(timeout=5)
        # Exit the main process
        sys.exit(0)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for data_parallel_rank in range(script_args.data_parallel_size):
        parent_connection, child_connection = Pipe()
        process = Process(
            target=llm_worker,
            args=(script_args, data_parallel_rank, master_port, child_connection),
        )
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Wait for all workers to send "ready"
        ready_connections = set()
        while len(ready_connections) < script_args.data_parallel_size:
            for connection in connections:
                msg = connection.recv()
                if isinstance(msg, dict) and msg.get("status") == "ready":
                    ready_connections.add(connection)

        yield

        # Wait for processes to terminate
        for process in processes:
            process.join(timeout=10)  # Wait for 10 seconds for the process to terminate
            if process.is_alive():
                logger.warning(
                    f"Process {process} is still alive after 10 seconds, attempting to terminate..."
                )
                process.terminate()
                process.join()  # ensure process termination after calling terminate()

    app = FastAPI(lifespan=lifespan)

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        """
        Retrieves the world size of the LLM engine, which is `tensor_parallel_size * data_parallel_size`.

        Returns:
            `dict`:
                A dictionary containing the world size.

        Example response:
        ```json
        {"world_size": 8}
        ```
        """
        return {
            "world_size": script_args.tensor_parallel_size
            * script_args.data_parallel_size
        }

    class GenerateRequest(BaseModel):
        prompts: list[str | list[int]]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None
        logprobs: int | None = None
        prompt_logprobs: int | None = None

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]
        generated_logprobs: list[list[dict[int, float]]] | None = None

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(
                backend="outlines", regex=request.guided_decoding_regex
            )
        else:
            guided_decoding = None

        # Sampling parameters
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
            logprobs=request.logprobs,
            prompt_logprobs=request.prompt_logprobs,
        )

        # Send the prompts to the worker
        prompts = request.prompts
        if isinstance(prompts, list):
            prompts = [
                {"prompt_token_ids": prompt} if isinstance(prompt, list) else prompt
                for prompt in prompts
            ]
        kwargs = {"prompts": prompts, "sampling_params": sampling_params}
        connections[0].send({"type": "call", "method": "generate", "kwargs": kwargs})

        # Receive results
        request_outputs: list[RequestOutput] = connections[0].recv()
        completion_ids = [
            list(output.token_ids)
            for outputs in request_outputs
            for output in outputs.outputs
        ]
        out = {"completion_ids": completion_ids}

        if request.logprobs is not None:
            generated_logprobs: list = []
            for request_output in request_outputs:
                req_logprobs = []
                for completion_output in request_output.outputs:
                    assert completion_output.logprobs is not None
                    for cmpl_logprobs in completion_output.logprobs:
                        logprobs_dict = {k: v.logprob for k, v in cmpl_logprobs.items()}
                        req_logprobs.append(logprobs_dict)
                generated_logprobs.append(req_logprobs)
            out["generated_logprobs"] = generated_logprobs

        return out

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        """
        Updates the model weights with the provided tensor.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight tensor.
        """
        dtype = getattr(torch, request.dtype.split(".")[-1])
        kwargs = {
            "method": "update_named_param",
            "args": (request.name, dtype, tuple(request.shape)),
        }
        connections[0].send(
            {"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs}
        )
        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        connections[0].send({"type": "call", "method": "reset_prefix_cache"})
        success = connections[0].recv()
        return {
            "message": "Request received, resetting prefix cache status: "
            + str(success)
        }

    # Start the server
    uvicorn.run(
        app,
        host=script_args.host,
        port=script_args.port,
        log_level=script_args.log_level,
    )


if __name__ == "__main__":
    config = TrainerConfig()
    main(ScriptArguments(model=config.model_id))
