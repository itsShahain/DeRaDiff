from dataclasses import dataclass


@dataclass
class SD15Config:
    
    model_type: str = "SD15"
    pretrained_model_name: str = "runwayml/stable-diffusion-v1-5"
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: int = 0
    test_prompts_path: str = "./data/test_prompts.yaml"
