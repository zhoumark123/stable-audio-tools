import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import torch.nn as nn
from transformers import AutoTokenizer

device = "cpu"

example_prompt = "Helicopter Circling Around in Stereo"
example_seconds = 11

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
max_token_length = model.conditioner.conditioners["prompt"].max_length # is 64 from configs

model = model.to(device)

model.pretransform.model_half = False
model.to(torch.float32)

# force T5 model to float32
t5_model = model.conditioner.conditioners["prompt"].model
t5_model.to(torch.float32)

for p in model.parameters():
    p.requires_grad = False

model.eval()

def verify_model_dtypes(model):
    """Verify all model components are in float32."""
    print("=== Model dtype verification ===")
    
    # Check main model parameters
    dtypes = set()
    for name, param in model.named_parameters():
        dtypes.add(param.dtype)
        if param.dtype != torch.float32:
            print(f"WARNING: {name} has dtype {param.dtype}")
    
    print(f"Main model parameter dtypes: {dtypes}")
    
    # Check pretransform
    if hasattr(model, 'pretransform') and model.pretransform is not None:
        print(f"Pretransform model_half: {getattr(model.pretransform, 'model_half', 'N/A')}")
        if hasattr(model.pretransform, 'model') and model.pretransform.model is not None:
            pretransform_dtypes = set()
            for name, param in model.pretransform.model.named_parameters():
                pretransform_dtypes.add(param.dtype)
                if param.dtype != torch.float32:
                    print(f"WARNING: pretransform.{name} has dtype {param.dtype}")
            print(f"Pretransform parameter dtypes: {pretransform_dtypes}")
    
    # Check T5 model
    if hasattr(model, 'conditioner') and hasattr(model.conditioner, 'conditioners'):
        if 'prompt' in model.conditioner.conditioners:
            t5_conditioner = model.conditioner.conditioners['prompt']
            if hasattr(t5_conditioner, 'model'):
                t5_dtypes = set()
                for name, param in t5_conditioner.model.named_parameters():
                    t5_dtypes.add(param.dtype)
                    if param.dtype != torch.float32:
                        print(f"WARNING: T5.{name} has dtype {param.dtype}")
                print(f"T5 model parameter dtypes: {t5_dtypes}")
    
    print("=== End verification ===\n")

verify_model_dtypes(model)

def generate_audio_from_tokens(input_ids, seconds_total_val):
    # seconds_total_val needs to be a float tensor of shape torch.Size([1])
    global model, sample_rate, sample_size, device, max_token_length

    input_ids = input_ids.to(device)
    seconds_total_val = seconds_total_val.to(device)
    # Pad input_ids to max_token_length if needed
    current_length = input_ids.shape[1]
    if current_length < max_token_length:
        padding = torch.zeros((input_ids.shape[0], max_token_length - current_length), dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, padding], dim=1)
    elif current_length > max_token_length:
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
            use_checkpointing=False
        )
        output_processed_traced = rearrange(output, "b d n -> d (b n)")
    return output_processed_traced

# Get the T5 tokenizer from the model
t5_tokenizer = model.conditioner.conditioners["prompt"].tokenizer

print(f"Using T5 tokenizer with max_length: {max_token_length}")

tokenized_inputs_dummy = t5_tokenizer(
    example_prompt,
    return_tensors="pt"
)
tokenized_inputs_dummy = {
    "input_ids": tokenized_inputs_dummy["input_ids"].to(device),
    "attention_mask": tokenized_inputs_dummy["attention_mask"].to(device)
}
dummy_seconds_tensor = torch.tensor([example_seconds], device=device, dtype=torch.float32)


print("Attempting to trace generate_audio_from_tokens...")
traced_generate_audio_fn = torch.jit.trace(
    generate_audio_from_tokens,
    (tokenized_inputs_dummy["input_ids"], dummy_seconds_tensor), 
    check_trace=False
)
print("Successfully traced generate_audio_from_tokens.")

print("Running inference with the traced model...")


# test with different inputs
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
    output_from_traced = traced_generate_audio_fn(tokenized_inputs["input_ids"], seconds_total_tensor)
    
    print(f"Output from traced model shape: {output_from_traced.shape}")
    # Process and save the output from the traced model
    output_processed_traced = output_from_traced.to(torch.float32).div(torch.max(torch.abs(output_from_traced))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    traced_output_path = f"output_from_traced_saos_{i}.wav"
    torchaudio.save(traced_output_path, output_processed_traced, sample_rate)
    print(f"Saved audio from traced model to {traced_output_path}")

torch.jit.save(traced_generate_audio_fn, f"traced_saos_{device}.pt")