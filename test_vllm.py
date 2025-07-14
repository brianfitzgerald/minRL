from minrl.constants import QWEN_3_0_6B
from vllm import LLM, SamplingParams
from minrl.tasks.connections import CONNECTIONS_PROMPT
from minrl.constants import TrainerConfig
from vllm.envs import set_vllm_use_v1

cfg = TrainerConfig()
set_vllm_use_v1(True)

vllm_model = LLM(
    model=QWEN_3_0_6B,
    device="cuda:0",
    dtype="bfloat16",
    enforce_eager=True,
)

output = vllm_model.generate(
    CONNECTIONS_PROMPT
    + "User: candle, crayon, honeycomb, seal, defense, excuse, out, reason, kettles, mittens, raindrops, whiskers, canine, fang, molar, tusk",
    sampling_params=SamplingParams(max_tokens=512, n=16),
)
print(output[0].outputs[0].text)
