import torch
import torchaudio
from transformers import AutoTokenizer
import os
import time

device = "cpu"
print(f"Using device: {device}")

start_time = time.time()
traced_model = torch.jit.load("traced_saos_cpu.pt", map_location="cpu")
print(f"Time taken to load traced_saos.pt: {time.time() - start_time}")
print("nodes :", len(list(traced_model.graph.nodes())))
print("consts:", sum(1 for n in traced_model.graph.nodes() if n.kind() == "prim::Constant"))
traced_model.eval()
traced_model.to(device)

test_prompt = "A funky bassline in a minor key"
test_seconds = 5  # Duration in seconds
output_filename = "test_output_from_loaded_traced_model.wav"
sample_rate = 44100 # Assuming sample rate is 44100, adjust if different from training

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
output_from_traced = traced_model(tokenized_inputs["input_ids"], seconds_total_tensor)
print(f"Time taken to run traced model: {time.time() - start_time}")
print(f"Output from traced model shape: {output_from_traced.shape}")


output_from_traced = output_from_traced.to(torch.float32).div(torch.max(torch.abs(output_from_traced))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
traced_output_path = f"output_from_loaded_traced.wav"
torchaudio.save(traced_output_path, output_from_traced, sample_rate)
print(f"Saved audio from traced model to {traced_output_path}")