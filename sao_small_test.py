import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

device = "mps"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# Set up text and timing conditioning
conditioning = [{
    "prompt": "Latin Funk Drumset 115 BPM Stereo",
    "seconds_total": 11
}]

# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=8,
    conditioning=conditioning,
    sample_size=sample_size,
    sampler_type="pingpong",
    cfg_scale=1,
    device=device
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)