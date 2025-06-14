import torch
import torchaudio
from einops import rearrange
from transformers import AutoTokenizer
import os
import time
# Define device
device = "cpu"
print(f"Using device: {device}")

start_time = time.time()
traced_model = torch.jit.load(f"traced_saos_a2a_{device}.pt", map_location=device)
print(f"Successfully loaded traced_saos_a2a_{device}.pt in {time.time() - start_time} seconds")
traced_model.eval()

init_audio_tensor, init_audio_sample_rate = torch.zeros(2, 44100 * 11), 44100
init_noise_level = torch.tensor(1, device=device, dtype=torch.float32)

test_prompt = "bright slap funk bassline"
test_seconds = 11
sample_rate = 44100

tokenizer = AutoTokenizer.from_pretrained("t5-base")

tokenized_inputs = tokenizer(
    test_prompt,
    return_tensors="pt",
)

input_ids = tokenized_inputs["input_ids"].to(device)
seconds_total_tensor = torch.tensor([test_seconds], device=device, dtype=torch.float32)

print(f"Test prompt: \"{test_prompt}\"")
print(f"Input IDs shape: {input_ids.shape}")
print(f"Seconds tensor: {seconds_total_tensor}")

start_time = time.time()
output_from_traced = traced_model(tokenized_inputs["input_ids"], seconds_total_tensor, init_audio_tensor, init_noise_level)
print(f"Time taken to generate audio from traced model: {time.time() - start_time} seconds")
print(f"Output from traced model shape: {output_from_traced.shape}")


output_from_traced = output_from_traced.to(torch.float32).div(torch.max(torch.abs(output_from_traced))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
traced_output_path = f"output_from_loaded_traced_a2a.wav"
torchaudio.save(traced_output_path, output_from_traced, sample_rate)
print(f"Saved audio from traced model to {traced_output_path}")