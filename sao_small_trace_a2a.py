import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import torch.nn as nn
from transformers import AutoTokenizer

device = "cpu"

model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
max_token_length = model.conditioner.conditioners["prompt"].max_length # is 64 from configs

model = model.to(device)

model.pretransform.model_half = False
model.to(torch.float32)

t5_model = model.conditioner.conditioners["prompt"].model
t5_model.to(torch.float32)

for p in model.parameters():
    p.requires_grad = False

model.eval()

def generate_audio_from_tokens(input_ids, seconds_total_val, init_audio_tensor, init_noise_level):
    # seconds_total_val needs to be a float tensor of shape torch.Size([1])
    global model, sample_rate, device, max_token_length

    input_ids = input_ids.to(device)
    seconds_total_val = seconds_total_val.to(device)
    init_audio_tensor = init_audio_tensor.to(device)
    init_noise_level = init_noise_level.to(device)
    # Pad input_ids to max_token_length if needed
    current_length = input_ids.shape[1]
    if current_length < max_token_length:
        padding = torch.zeros((input_ids.shape[0], max_token_length - current_length), dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, padding], dim=1)
    elif current_length > max_token_length:
        print(f"WARNING: Input ids are too long, truncating to {max_token_length} tokens")
        input_ids = input_ids[:, :max_token_length]
    
    attention_mask = (input_ids != 0).to(torch.bool)
    
    token_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad(): 
        
        # Taken from T5Conditioner
        prompt_conditioner = model.conditioner.conditioners["prompt"]
        t5_model = prompt_conditioner.model.to(device)
        t5_model.eval()
        prompt_hidden = t5_model(input_ids=token_ids, attention_mask=attention_mask)["last_hidden_state"]

        if not isinstance(prompt_conditioner.proj_out, nn.Identity):    
            prompt_hidden = prompt_hidden.to(next(model.model.parameters()).dtype)
            prompt_hidden = prompt_conditioner.proj_out(prompt_hidden)
        prompt_hidden = prompt_hidden * attention_mask.unsqueeze(-1).float()

        # continue the loop in MultiConditioner after T5Conditioner
        seconds_conditioner = model.conditioner.conditioners["seconds_total"]
        seconds_embeds, seconds_mask = seconds_conditioner(seconds_total_val, device)

        conditioning_tensors = {
            "prompt": [prompt_hidden, attention_mask],
            "seconds_total": [seconds_embeds, seconds_mask]
        }

        output = generate_diffusion_cond(
            model,
            steps=8,
            conditioning_tensors=conditioning_tensors,
            sample_size=sample_size,
            sampler_type="pingpong",
            cfg_scale=1,
            device=str(device), 
            use_checkpointing=False,
            init_audio=(sample_rate, init_audio_tensor),
            init_noise_level=init_noise_level
        )
        output_processed_traced = rearrange(output, "b d n -> d (b n)")
    return output_processed_traced

t5_tokenizer = model.conditioner.conditioners["prompt"].tokenizer

example_prompt = "Helicopter Circling Around in Stereo"
example_seconds = 11
tokenized_inputs_dummy = t5_tokenizer(
    example_prompt,
    return_tensors="pt"
)
tokenized_inputs_dummy = {
    "input_ids": tokenized_inputs_dummy["input_ids"].to(device),
    "attention_mask": tokenized_inputs_dummy["attention_mask"].to(device)
}
dummy_seconds_tensor = torch.tensor([example_seconds], device=device, dtype=torch.float32)

dummy_init_audio_tensor, dummy_init_audio_sample_rate = torchaudio.load("latin funk drumset 115 bpm.wav")
dummy_init_audio_tensor = torchaudio.transforms.Resample(dummy_init_audio_sample_rate, sample_rate)(dummy_init_audio_tensor)
dummy_init_noise_level = torch.tensor(1, device=device, dtype=torch.float32)


print("Attempting to trace generate_audio_from_tokens...")
traced_generate_audio_fn = torch.jit.trace(
    generate_audio_from_tokens,
    (tokenized_inputs_dummy["input_ids"], dummy_seconds_tensor, dummy_init_audio_tensor, dummy_init_noise_level), 
    check_trace=False
)
print("Successfully traced generate_audio_from_tokens.")
print("Running inference with the traced model...")

# test with different inputs
init_audio_tensor, init_audio_sample_rate = torch.zeros(2, 44100 * 11), 44100
init_audio_tensor = torchaudio.transforms.Resample(init_audio_sample_rate, sample_rate)(init_audio_tensor)
init_noise_level = torch.tensor(1, device=device, dtype=torch.float32)
prompt = "bright slap funk bassline"
seconds_total = 11
seconds_total_tensor = torch.tensor([seconds_total], device=device, dtype=torch.float32)


# Test with pretrained tokenizer
pretrained_tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenized_inputs = pretrained_tokenizer(
    prompt,
    return_tensors="pt",
)
print(f"Tokenized inputs shape: {tokenized_inputs['input_ids'].shape}")
print(f"Tokenized inputs: {tokenized_inputs['input_ids']}")
print(f"Tokenized inputs attention mask: {tokenized_inputs['attention_mask']}")

for i in range(5):

    output_from_traced = traced_generate_audio_fn(tokenized_inputs["input_ids"], seconds_total_tensor, init_audio_tensor, init_noise_level)
    
    print(f"Output from traced model shape: {output_from_traced.shape}")
    # Process and save the output from the traced model
    output_processed_traced = output_from_traced.to(torch.float32).div(torch.max(torch.abs(output_from_traced))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    traced_output_path = f"output_from_traced_{i}.wav"
    torchaudio.save(traced_output_path, output_processed_traced, sample_rate)
    print(f"Saved audio from traced model to {traced_output_path}")

torch.jit.save(traced_generate_audio_fn, f"traced_saos_a2a_{device}.pt")