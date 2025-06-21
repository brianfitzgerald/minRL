from minrl.constants import QWEN_3_0_6B
from vllm import LLM
from minrl.tasks.connections import CONNECTIONS_PROMPT

vllm_model = LLM(
    model=QWEN_3_0_6B,
    device="cuda:0",
    max_model_len=1024,
    max_seq_len_to_capture=1024,
    enforce_eager=True,
    gpu_memory_utilization=0.2,
)

output = vllm_model.generate(
    CONNECTIONS_PROMPT
    + "User: candle, crayon, honeycomb, seal, defense, excuse, out, reason, kettles, mittens, raindrops, whiskers, canine, fang, molar, tusk"
)
print(output)
