from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils import (
    deprecate,
    is_torch_xla_available,
)

from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from src.config.sd15_config import SD15Config

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

class CustomStableDiffusion15Pipeline(StableDiffusionPipeline):
    # 1. Define the _components attribute for your custom pipeline
    # Start with the components from the parent class and add your new ones.
    # Do NOT add 'lambda_val' here.
    _components = [
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "unet",                # This refers to the main UNet passed to super().__init__
        "scheduler",
        "image_encoder",      
        "feature_extractor",   
        "unet_beta"            # Your additional UNet model
    ]

    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel, # base unet
                 dpo_unet: UNet2DConditionModel, # dpo unet
                 scheduler: EulerAncestralDiscreteScheduler, # Instance of CustomCombinedDDPMScheduler
                 safety_checker: StableDiffusionSafetyChecker, 
                 feature_extractor: CLIPImageProcessor,
                 image_encoder: CLIPVisionModelWithProjection,
                 requires_safety_checker: bool = True,
                 lmbda: float = 0.5): # Default weight for the base unet

        super().__init__(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
            safety_checker=safety_checker, feature_extractor=feature_extractor, image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker
        )

        self.scheduler = scheduler
        # Register dpo_unet so it's also moved to device, etc.
        # self.dpo_unet = dpo_unet # Direct assignment
        self.register_modules(dpo_unet=dpo_unet) # Use this to properly register
        self.lmbda = lmbda

        @torch.no_grad()
        # @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            clip_skip: Optional[int] = None,
            callback_on_step_end = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
        ):




            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )

            # 0. Default height and width to unet
            if not height or not width:
                height = (
                    self.unet.config.sample_size
                    if self._is_unet_config_sample_size_int
                    else self.unet.config.sample_size[0]
                )
                width = (
                    self.unet.config.sample_size
                    if self._is_unet_config_sample_size_int
                    else self.unet.config.sample_size[1]
                )
                height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
            # to deal with lora scaling and other possible forward hooks

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                ip_adapter_image,
                ip_adapter_image_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs
            self._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # 3. Encode input prompt
            lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                )

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler=self.scheduler, num_inference_steps=num_inference_steps, device=device, timesteps=timesteps, sigmas=sigmas
            )

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 6.1 Add image embeds for IP-Adapter
            added_cond_kwargs = (
                {"image_embeds": image_embeds}
                if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
                else None
            )

            # 6.2 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)


                    base_noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    dpo_noise_pred = self.dpo_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        base_noise_pred_uncond, base_noise_pred_text = base_noise_pred.chunk(2)
                        base_noise_pred = base_noise_pred_uncond + self.guidance_scale * (base_noise_pred_text - base_noise_pred_uncond)

                        dpo_noise_pred_uncond, dpo_noise_pred_text = dpo_noise_pred.chunk(2)
                        dpo_noise_pred = dpo_noise_pred_uncond + self.guidance_scale * (dpo_noise_pred_text - dpo_noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://huggingface.co/papers/2305.08891
                        base_noise_pred = rescale_noise_cfg(base_noise_pred, base_noise_pred_text, guidance_rescale=self.guidance_rescale)
                        dpo_noise_pred = rescale_noise_cfg(dpo_noise_pred, dpo_noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    # NOTE: Changes for DeRaDiff
                    noise_pred = (1 - self.lmbda) * base_noise_pred + self.lmbda * dpo_noise_pred
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

                    if XLA_AVAILABLE:
                        xm.mark_step()

            if not output_type == "latent":
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                    0
                ]
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

def load_deradiff_sd15_pipeline(anchor_model_ckpt_path: str, lambda_val: float, device: str) -> CustomStableDiffusion15Pipeline:

    dpo_unet = UNet2DConditionModel.from_pretrained(
        anchor_model_ckpt_path,         
        subfolder='unet',
        torch_dtype=torch.float16
    ).to(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        SD15Config.pretrained_model_name,
        torch_dtype=torch.float16
    ).to(device)

    pipe.safety_checker = None # Prevent trigger-happy blacking out

    # NOTE: We illustrate DeRaDiff using the Euler-A scheduler
    custom_scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, steps_offset=0)
    passed_image_encoder = getattr(pipe, 'image_encoder', None)
    custom_pipe = CustomStableDiffusion15Pipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,          # Base UNet
        dpo_unet=dpo_unet,       # DPO UNet
        scheduler=custom_scheduler,
        safety_checker=pipe.safety_checker,
        feature_extractor=pipe.feature_extractor,
        image_encoder=passed_image_encoder,
        requires_safety_checker=(pipe.safety_checker is not None),
        lmbda=lambda_val
    )

    return custom_pipe
