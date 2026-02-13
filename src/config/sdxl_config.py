from dataclasses import dataclass


@dataclass
class SDXLConfig:
    
    model_type: str = "SDXL"
    pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    guidance_scale: float = 5.0
    variant: str = "fp16"
    num_inference_steps: int = 50
    seed: int = 0
    test_prompts_path: str = "./data/test_prompts.yaml"
