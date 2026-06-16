import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPProcessor

DEVICE = "cuda:0"
DTYPE = torch.bfloat16


torch.set_float32_matmul_precision("high")

# Check if weights/model.torchscript exists, if not, download it from huggingface
repo_id = "yermandy/deepfake-detection"
filename = "model.torchscript"

model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="weights")

# Load checkpoint
model = torch.jit.load(model_path, map_location=DEVICE)

# Load preprocessing function
preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load some images
paths = [
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/000.png",
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/045.png",
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/030.png",
    "datasets/CDFv2/Celeb-synthesis/id0_id1_0000/015.png",
    "datasets/CDFv2/YouTube-real/00000/000.png",
    "datasets/CDFv2/YouTube-real/00000/014.png",
    "datasets/CDFv2/YouTube-real/00000/028.png",
    "datasets/CDFv2/YouTube-real/00000/043.png",
    "datasets/CDFv2/Celeb-real/id0_0000/045.png",
    "datasets/CDFv2/Celeb-real/id0_0000/030.png",
    "datasets/CDFv2/Celeb-real/id0_0000/015.png",
    "datasets/CDFv2/Celeb-real/id0_0000/000.png",
]

# To pillow images
pillow_images = [Image.open(image) for image in paths]

# To tensors
batch_images = torch.stack(
    [preprocess(images=image, return_tensors="pt")["pixel_values"][0] for image in pillow_images]
)

# Set model to evaluation mode
model.eval()

# Move model to the correct device and dtype
model = model.to(DEVICE).to(DTYPE)

# Move inputs to the correct device and dtype
batch_images = batch_images.to(DEVICE).to(DTYPE)

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        # Forward pass
        output = model(batch_images)

        softmax_output = output.softmax(dim=1).cpu().numpy()

for path, (p_real, p_fake) in zip(paths, softmax_output):
    print(f"p(real) = {p_real:.4f}, p(fake) = {p_fake:.4f}, image: {path}")
