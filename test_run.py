from minrl.constants import QWEN_25_05B
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from vllm import LLM

vllm_model = LLM(
    model=QWEN_25_05B,
    device="cuda:0",
    max_model_len=1024,
    max_seq_len_to_capture=1024,
    enforce_eager=True,
    gpu_memory_utilization=0.2,
)

tokenizer = AutoTokenizer.from_pretrained(QWEN_25_05B)
model = AutoModelForCausalLM.from_pretrained(QWEN_25_05B)

prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
outputs = model(inputs)
