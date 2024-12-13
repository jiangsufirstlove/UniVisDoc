import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MAX_NUM = 6

# Paths
IMAGE_DIR = '/data1/doc_data/ysh/4_tasks_realenv/publaynet_resized'
OUTPUT_DIR = '/data1/doc_data/ysh/4_tasks_realenv/publaynet_model_output'
MODEL_PATH = '/data1/doc_data/zrb/output/830-newmerged'

# Build transform
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

# Dynamic preprocess
def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# Load image
def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Main function to run on each GPU
def main(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().to(rank)

    # Use Distributed Data Parallel
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Get list of image files and distribute the workload
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    # Split the list for each process
    image_files = image_files[rank::world_size]

    # Use tqdm to add progress bar
    for image_file in tqdm(image_files, desc=f"GPU {rank}", unit="image", position=rank):
        image_path = os.path.join(IMAGE_DIR, image_file)
        pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).to(rank)

        generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
        )

        question = '<image>\nPlease identify the objects in <image> that belong to <ref>text</ref>, <ref>title</ref>, <ref>list</ref>, <ref>table</ref> and <ref>figure</ref>.'
        response = model.module.chat(tokenizer, pixel_values, question, generation_config)

        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_file)[0]}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response)

    # Cleanup
    dist.destroy_process_group()

# Entry point
if __name__ == "__main__":
    world_size = 8  # Number of GPUs
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
