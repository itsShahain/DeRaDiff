import os
import torch


def gen(custom_pipe, image_id, prompt, device, guidance_scale=5.0,
        num_inference_steps=50, seed=0, output_dir="results", save_image=True):
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"Prompt: {prompt}")

    im = custom_pipe(
        prompt=prompt,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    if save_image:
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        output_fpath = os.path.join(image_dir, f"{image_id}_seed{seed}.png")
        im.save(output_fpath)

    return im
