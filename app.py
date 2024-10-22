import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Define the prompt
prompt = (
    "Create a cartoon animation of a young boy named Ali, around 8 years old, sitting in a cartoon-style old, dusty room. The wooden bookshelf behind him is filled with colorful, cartoonish books. The atmosphere is playful, with a warm, dim cartoonish glow in the room. Ali looks curious as he brushes off cartoon dust from the bookshelf, excited to find something special."
)

# Load the model
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)

# Offload model to CPU and optimize memory
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Generate the video
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

# Export the video
export_to_video(video, "output.mp4", fps=8)
