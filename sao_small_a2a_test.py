import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

device = "mps"


model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

print(f"Model sample rate: {sample_rate}")

model = model.to(device)

init_audio_waveform, init_audio_sample_rate = torchaudio.load("latin funk drumset 115 bpm.wav")
init_audio_waveform = init_audio_waveform.to(device)
# If input audio is longer than sample_size, it will be truncated by the prepare_audio function.
# If it's shorter, it will be padded.


conditioning = [{
    "prompt": "E minor syncopated guitar 115 bpm",
    "seconds_total": init_audio_waveform.shape[-1] / init_audio_sample_rate
}]


# >0.8 starts mixing the new audio with the original audio, otherwise sounds very similar to original
init_noise_level = 0.85


output = generate_diffusion_cond(
    model,
    steps=8,
    conditioning=conditioning,
    sample_size=sample_size,
    sampler_type="pingpong",
    cfg_scale=1,
    device=device,
    init_audio=(init_audio_sample_rate, init_audio_waveform),
    init_noise_level=init_noise_level
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output_float = output.to(torch.float32)
# Check if output is all zeros before normalization
if torch.max(torch.abs(output_float)) == 0:
    print("Warning: Output audio is all zeros. Skipping normalization.")
    output_int16 = output_float.mul(32767).to(torch.int16).cpu()
else:
    output_int16 = output_float.div(torch.max(torch.abs(output_float))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

torchaudio.save("output_audio_to_audio.wav", output_int16, sample_rate)
print("Audio-to-audio generation complete. Saved to output_audio_to_audio.wav")
