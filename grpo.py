import gc
from typing import Callable, List

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from data_types import Episode, MiniBatch


@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    batch: MiniBatch,
    tokenizer: PreTrainedTokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    """Generate multiple responses for each prompt in the batch."""
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Prepare input_ids for generation
    input_ids = []
    for prefix_ids in batch.prefix_token_ids:
        for _ in range(num_answer_per_question):
            input_ids.append(prefix_ids)
    
    # Convert to tensor and move to device
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    
    # Generate responses
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_gen_len,
        pad_token_id=pad_token_id,
        eos_token_id=end_token_id,
        do_sample=True,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    
    # Clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()
    
    # Process outputs and create episodes
    episodes = []
    for i in range(len(batch.prefix)):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = outputs.sequences[idx][len(batch.prefix_token_ids[i]):].tolist()
            
            # Remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[:generated_token_ids.index(pad_token_id)]
            
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            
            # Calculate rewards
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=end_token,
            )
            
            # Create episode
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=end_token_id in generated_token_ids,
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    
    return episodes
