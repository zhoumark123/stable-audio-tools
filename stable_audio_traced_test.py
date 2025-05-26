import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from transformers import AutoTokenizer

device = "mps"
sample_rate = 44100

example_prompt = "80s bass guitar"
example_seconds = 5

traced_model_path = "traced_stable_audio_open.pt"

print(f"Loading traced model from {traced_model_path}...")
generate_audio_traced = torch.jit.load(traced_model_path)
print("Successfully loaded traced model")

pretrained_tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenized_inputs_dummy = pretrained_tokenizer(
    example_prompt,
    return_tensors="pt",
)

tokenized_inputs_dummy = {
    "input_ids": tokenized_inputs_dummy["input_ids"].to(device),
    "attention_mask": tokenized_inputs_dummy["attention_mask"].to(device)
}

dummy_seconds_tensor = torch.tensor([example_seconds], device=device, dtype=torch.float32) 


print("Running inference with the traced model...")

output = generate_audio_traced(tokenized_inputs_dummy["input_ids"], dummy_seconds_tensor)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")



# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)
