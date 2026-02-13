import torch
import argparse

from src.config.sd15_config import SD15Config
from src.eval.evaluator import Evaluator
from src.pipeline.custom_sd15_pipeline import load_deradiff_sd15_pipeline
from src.utils.gen_utils import gen
from src.utils.load_utils import load_yaml_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run DeRaDiff SDXL sampling with a custom λ value."
    )

    parser.add_argument(
        "--Beta",
        type=int,
        default=1000,
        help="Aligned model regularization strength."
    )

    
    parser.add_argument(
        "--Lambda",
        type=float,
        default=1.0,
        help="Lambda value for realignment, approximate regularization strength is β/λ."
    )
    
    parser.add_argument(
        "--anchor_model_ckpt_path",
        "--a",
        type=str,
        default=None,
        help='Name of the model, e.g. "beta2000_64acc_600epoch".'
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/sd15_eval",
        help="Directory to save evaluation results.",
    )

    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Whether to save generated images.",
    )

    parser.add_argument(
        "--cuda_id",
        type=str,
        default="0",
        help="Accelerator device ID to use.",
    )

    args = parser.parse_args()
    # NOTE: change device as needed
    device = f"cuda:{args.cuda_id}"

    custom_pipe = load_deradiff_sd15_pipeline(
        anchor_model_ckpt_path=args.anchor_model_ckpt_path,
        lambda_val=args.Lambda,
        device=device
    )

    evaluator = Evaluator(device, output_dir=args.output_dir)
    test_prompts = load_yaml_file(SD15Config.test_prompts_path)["test_prompts"]

    with torch.no_grad():
        for i, p in enumerate(test_prompts, start=1):
            im = gen(
                custom_pipe, image_id=i, prompt=p, device=device,
                guidance_scale=SD15Config.guidance_scale,
                num_inference_steps=SD15Config.num_inference_steps,
                seed=SD15Config.seed,
                output_dir=args.output_dir,
                save_image=args.save_images
            )
            evaluator.score_image(im, p)

    evaluator.save_results(SD15Config.model_type, args.Lambda, args.Beta)
