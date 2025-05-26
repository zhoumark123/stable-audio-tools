import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from transformers import AutoTokenizer


device = "mps"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
max_token_length = 128


model = model.to(device)

example_prompt = "80s bass guitar"
example_seconds = 2

for p in model.parameters():
    p.requires_grad = False

model.eval()


# Monkey patch the checkpoint function
def no_checkpoint(function, *args, **kwargs):
    return function(*args)

# Replace checkpoint calls with direct function calls during tracing
import stable_audio_tools.models.transformer as transformer_module
import stable_audio_tools.models.local_attention as local_attention_module
import stable_audio_tools.models.autoencoders as autoencoders_module
import stable_audio_tools.models.convnext as convnext_module
import stable_audio_tools.models.encodec as encodec_module
import stable_audio_tools.models.discriminators as discriminators_module

transformer_module.checkpoint = no_checkpoint
local_attention_module.checkpoint = no_checkpoint
autoencoders_module.checkpoint = no_checkpoint
convnext_module.checkpoint = no_checkpoint
encodec_module.checkpoint = no_checkpoint
discriminators_module.checkpoint = no_checkpoint


def generate_audio_from_tokens(input_ids, seconds_total_val):
    # seconds_total_val needs to be a float tensor of shape torch.Size([1])
    global model, sample_size, device, max_token_length
    
    # Pad input_ids to max_token_length if needed
    current_length = input_ids.shape[1]
    if current_length < max_token_length:
        padding = torch.zeros((input_ids.shape[0], max_token_length - current_length), dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, padding], dim=1)
    elif current_length > max_token_length:
        print(f"WARNING: Input ids are too long, truncating to {max_token_length} tokens")
        input_ids = input_ids[:, :max_token_length]

    attention_mask = (input_ids != 0).to(torch.bool)    
    token_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    print("padded token_ids: ", token_ids)
    print("padded attn mask: ", attention_mask)

    with torch.no_grad():
        prompt_conditioner = model.conditioner.conditioners["prompt"]
        t5_model = prompt_conditioner.model.to(device)
        t5_model.eval()
        prompt_hidden = t5_model(input_ids=token_ids, attention_mask=attention_mask)["last_hidden_state"]
        if not isinstance(prompt_conditioner.proj_out, torch.nn.Identity):    
            prompt_hidden = prompt_hidden.to(next(model.model.parameters()).dtype)
            prompt_hidden = prompt_conditioner.proj_out(prompt_hidden)
        prompt_hidden = prompt_hidden * attention_mask.unsqueeze(-1).float()


        # continue the loop in MultiConditioner after T5Conditioner
        seconds_start_val = torch.tensor([0], device=device, dtype=torch.float32) # use default of 0
        seconds_start_conditioner = model.conditioner.conditioners["seconds_start"]
        seconds_start_embeds, seconds_start_mask = seconds_start_conditioner(seconds_start_val, device)
    

        seconds_total_conditioner = model.conditioner.conditioners["seconds_total"]
        seconds_embeds, seconds_mask = seconds_total_conditioner(seconds_total_val, device)

        conditioning_tensors = {
            "prompt": [prompt_hidden, attention_mask],
            "seconds_start": [seconds_start_embeds, seconds_start_mask],
            "seconds_total": [seconds_embeds, seconds_mask]
        }

        output = generate_diffusion_cond(
            model,
            steps=10,
            cfg_scale=7,
            conditioning_tensors=conditioning_tensors,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )

    return output

t5_tokenizer = model.conditioner.conditioners["prompt"].tokenizer
pretrained_tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenized_inputs_dummy = t5_tokenizer(
    example_prompt,
    return_tensors="pt",
)

tokenized_inputs_dummy = {
    "input_ids": tokenized_inputs_dummy["input_ids"].to(device),
    "attention_mask": tokenized_inputs_dummy["attention_mask"].to(device)
}

dummy_seconds_tensor = torch.tensor([example_seconds], device=device, dtype=torch.float32) 

print(f"Tokenized inputs shape: {tokenized_inputs_dummy['input_ids'].shape}")
print(f"Tokenized inputs: {tokenized_inputs_dummy['input_ids']}")
print(f"Tokenized inputs attention mask: {tokenized_inputs_dummy['attention_mask']}")

# generate_audio_traced = torch.jit.trace(generate_audio_from_tokens, 
#                                         (tokenized_inputs_dummy["input_ids"], dummy_seconds_tensor), 
#                                         check_trace=False)
# print("Successfully traced generate_audio_from_tokens.")

# print("Running inference with the traced model...")
torch.mps.empty_cache()
import gc
gc.collect()

output = generate_audio_from_tokens(tokenized_inputs_dummy["input_ids"], dummy_seconds_tensor)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")


# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)

# # print("Successfully saved audio to output.wav")
# print("Attempting to save traced model with torch.jit.save...")
# torch.jit.save(generate_audio_traced, "traced_stable_audio_open.pt")
# print("Successfully saved with torch.jit.save")