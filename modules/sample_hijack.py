import torch
import ldm_patched.modules.samplers
import ldm_patched.modules.model_management

from collections import namedtuple
from ldm_patched.contrib.external_align_your_steps import AlignYourStepsScheduler
from ldm_patched.contrib.external_custom_sampler import SDTurboScheduler
from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
from ldm_patched.modules.samplers import normal_scheduler, simple_scheduler, ddim_scheduler
from ldm_patched.modules.model_base import SDXLRefiner, SDXL
from ldm_patched.modules.conds import CONDRegular
from ldm_patched.modules.anima import AnimaModel
from ldm_patched.modules.sample import convert_cond
from ldm_patched.modules.sample import get_additional_models, get_models_from_cond, cleanup_additional_models
from ldm_patched.modules.samplers import resolve_areas_and_cond_masks, wrap_model, calculate_start_end_timesteps, \
    create_cond_with_same_area_if_none, pre_run_control, apply_empty_x_to_equal_area, encode_model_conds


current_refiner = None
refiner_switch_step = -1


@torch.no_grad()
@torch.inference_mode()
def clip_separate_inner(c, p, target_model=None, target_clip=None):
    if target_model is None or isinstance(target_model, SDXLRefiner):
        c = c[..., -1280:].clone()
    elif isinstance(target_model, SDXL):
        c = c.clone()
    else:
        p = None
        c = c[..., :768].clone()

        final_layer_norm = target_clip.cond_stage_model.clip_l.transformer.text_model.final_layer_norm

        final_layer_norm_origin_device = final_layer_norm.weight.device
        final_layer_norm_origin_dtype = final_layer_norm.weight.dtype

        c_origin_device = c.device
        c_origin_dtype = c.dtype

        final_layer_norm.to(device='cpu', dtype=torch.float32)
        c = c.to(device='cpu', dtype=torch.float32)

        c = torch.chunk(c, int(c.size(1)) // 77, 1)
        c = [final_layer_norm(ci) for ci in c]
        c = torch.cat(c, dim=1)

        final_layer_norm.to(device=final_layer_norm_origin_device, dtype=final_layer_norm_origin_dtype)
        c = c.to(device=c_origin_device, dtype=c_origin_dtype)
    return c, p


@torch.no_grad()
@torch.inference_mode()
def clip_separate(cond, target_model=None, target_clip=None):
    if isinstance(target_model, AnimaModel):
        results = []
        for context, metadata in cond:
            if 'anima_refiner_conditioning' in metadata:
                common = {key: value for key, value in metadata.items() if key not in (
                    'anima_refiner_conditioning', 'pooled_output', 'model_conds', 'cross_attn', 't5xxl_ids', 't5xxl_weights')}
                results.extend([[refiner_context, {**common, **refiner_metadata}]
                                for refiner_context, refiner_metadata in metadata['anima_refiner_conditioning']])
            elif 't5xxl_ids' in metadata:
                results.append([context, {key: value for key, value in metadata.items()
                                          if key != 'anima_refiner_conditioning'}])
            else:
                raise ValueError('Anima refiner requires Qwen/T5 conditioning. Encode prompts after loading the refiner.')
        return [[context.clone(), {key: value.clone() if isinstance(value, torch.Tensor) else value
                                   for key, value in metadata.items()}] for context, metadata in results]
    results = []

    for c, px in cond:
        p = px.get('pooled_output', None)
        c, p = clip_separate_inner(c, p, target_model=target_model, target_clip=target_clip)
        p = {} if p is None else {'pooled_output': p.clone()}
        results.append([c, p])

    return results


@torch.no_grad()
@torch.inference_mode()
def clip_separate_after_preparation(cond, target_model=None, target_clip=None):
    if isinstance(target_model, AnimaModel):
        raw = [[entry['cross_attn'], {key: value for key, value in entry.items()
                                     if key not in ('model_conds', 'cross_attn')}] for entry in cond]
        return convert_cond(clip_separate(raw, target_model=target_model))
    results = []

    for x in cond:
        p = x.get('pooled_output', None)
        c = x['model_conds']['c_crossattn'].cond

        c, p = clip_separate_inner(c, p, target_model=target_model, target_clip=target_clip)

        result = {'model_conds': {'c_crossattn': CONDRegular(c)}}

        if p is not None:
            result['pooled_output'] = p.clone()

        results.append(result)

    return results


@torch.no_grad()
@torch.inference_mode()
def sample_hacked(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    global current_refiner

    positive = positive[:]
    negative = negative[:]

    resolve_areas_and_cond_masks(positive, noise.shape[2], noise.shape[3], device)
    resolve_areas_and_cond_masks(negative, noise.shape[2], noise.shape[3], device)

    model_wrap = wrap_model(model)

    calculate_start_end_timesteps(model, negative)
    calculate_start_end_timesteps(model, positive)

    if latent_image is not None:
        latent_image = model.process_latent_in(latent_image)

    if hasattr(model, 'extra_conds'):
        positive = encode_model_conds(model.extra_conds, positive, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask)
        negative = encode_model_conds(model.extra_conds, negative, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask)

    #make sure each cond area has an opposite one with the same area
    for c in positive:
        create_cond_with_same_area_if_none(negative, c)
    for c in negative:
        create_cond_with_same_area_if_none(positive, c)

    # pre_run_control(model, negative + positive)
    pre_run_control(model, positive)  # negative is not necessary in Fooocus, 0.5s faster.

    apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
    apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

    extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": model_options, "seed":seed}

    if current_refiner is not None and hasattr(current_refiner.model, 'extra_conds'):
        positive_refiner = clip_separate_after_preparation(positive, target_model=current_refiner.model)
        negative_refiner = clip_separate_after_preparation(negative, target_model=current_refiner.model)

        if not isinstance(current_refiner.model, AnimaModel):
            positive_refiner = encode_model_conds(current_refiner.model.extra_conds, positive_refiner, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask)
            negative_refiner = encode_model_conds(current_refiner.model.extra_conds, negative_refiner, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask)

    def refiner_switch():
        nonlocal positive_refiner, negative_refiner
        cleanup_additional_models(set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control")))

        # clear ip-adapter for refiner
        extra_args['model_options'] = {k: {} if k == 'transformer_options' else v for k, v in extra_args['model_options'].items()}

        models, inference_memory = get_additional_models(positive_refiner, negative_refiner, current_refiner.model_dtype())
        ldm_patched.modules.model_management.load_models_gpu(
            [current_refiner] + models,
            current_refiner.model.memory_required([noise.shape[0] * 2] + list(noise.shape[1:])) + inference_memory)

        if isinstance(current_refiner.model, AnimaModel):
            for conditioning in (positive_refiner, negative_refiner):
                resolve_areas_and_cond_masks(conditioning, noise.shape[2], noise.shape[3], device)
                calculate_start_end_timesteps(current_refiner.model, conditioning)
            positive_refiner = encode_model_conds(current_refiner.model.extra_conds, positive_refiner, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask)
            negative_refiner = encode_model_conds(current_refiner.model.extra_conds, negative_refiner, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask)
        extra_args["cond"] = positive_refiner
        extra_args["uncond"] = negative_refiner

        model_wrap.inner_model = current_refiner.model
        print('Refiner Swapped')
        return

    def callback_wrap(step, x0, x, total_steps):
        switch_step = refiner_switch_step
        if current_refiner is not None and isinstance(current_refiner.model, AnimaModel):
            switch_step = refiner_switch_step - 1 if 0 < refiner_switch_step < total_steps else -1
        if step == switch_step and current_refiner is not None:
            refiner_switch()
        if callback is not None:
            # residual_noise_preview = x - x0
            # residual_noise_preview /= residual_noise_preview.std()
            # residual_noise_preview *= x0.std()
            callback(step, x0, x, total_steps)

    samples = sampler.sample(model_wrap, sigmas, extra_args, callback_wrap, noise, latent_image, denoise_mask, disable_pbar)
    output_model = model_wrap.inner_model if isinstance(model, AnimaModel) else model
    return output_model.process_latent_out(samples.to(torch.float32))


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas_scheduler_hacked(model, scheduler_name, steps):
    if scheduler_name == "karras":
        sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=float(model.model_sampling.sigma_min), sigma_max=float(model.model_sampling.sigma_max))
    elif scheduler_name == "exponential":
        sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=float(model.model_sampling.sigma_min), sigma_max=float(model.model_sampling.sigma_max))
    elif scheduler_name == "normal":
        sigmas = normal_scheduler(model, steps)
    elif scheduler_name == "simple":
        sigmas = simple_scheduler(model, steps)
    elif scheduler_name == "ddim_uniform":
        sigmas = ddim_scheduler(model, steps)
    elif scheduler_name == "sgm_uniform":
        sigmas = normal_scheduler(model, steps, sgm=True)
    elif scheduler_name == "turbo":
        sigmas = SDTurboScheduler().get_sigmas(model=model, steps=steps, denoise=1.0)[0]
    elif scheduler_name == "align_your_steps":
        model_type = 'SDXL' if isinstance(model.latent_format, ldm_patched.modules.latent_formats.SDXL) else 'SD1'
        sigmas = AlignYourStepsScheduler().get_sigmas(model_type=model_type, steps=steps, denoise=1.0)[0]
    else:
        raise TypeError("error invalid scheduler")
    return sigmas


ldm_patched.modules.samplers.calculate_sigmas_scheduler = calculate_sigmas_scheduler_hacked
ldm_patched.modules.samplers.sample = sample_hacked
