# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import psutil
import logging
import warnings
import numpy as np
from PIL import Image
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union
from multiprocessing import Pool

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
)
from transformers.utils import cached_file
from trl import AutoModelForCausalLMWithValueHead

from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import MAX_RATIO

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy
from verl.utils.svg_utils import export_svg_to_img


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_answer(action_string: str) -> Optional[str]:
    """
    Extracts the answer from the action string.

    Args:
        action_string: The string containing the answer in XML tags.

    Returns:
        The extracted answer if found, otherwise None.
    """
    answer_match = re.search(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
    if answer_match:
        return answer_match.group(1)
    return ""


def may_convert_svg_to_png(response: str, image: Image.Image):
    response_answer = extract_answer(response)
    if not (response_answer.startswith("```svg\n") and response_answer.endswith("```")):
        logger.error(f"Invalid SVG format: {response_answer}")
        return 'error', Image.new('RGB', (100, 100), color='black')
    svg_code = response_answer[7:-3]
    try:
        rendered_image = export_svg_to_img(svg_code, image)
        width, height = rendered_image.size
        if max(height, width) / min(height, width) > MAX_RATIO:
            return 'error', Image.new('RGB', (100, 100), color='black')
        return 'good', rendered_image
    except Exception as e:
        logger.error(f"Error converting SVG to PNG: {e}")
        logger.error(f"response: {response}")
        return 'error', Image.new('RGB', (100, 100), color='black')


def load_valuehead_params(path_or_repo_id: str) -> dict[str, torch.Tensor]:
    r"""Load value head parameters from Hugging Face Hub or local disk.

    Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.
    """
    kwargs = {"path_or_repo_id": path_or_repo_id, "cache_dir": None, "token": None}
    err_text = ""

    try:
        from safetensors import safe_open

        vhead_file = cached_file(filename="value_head.safetensors", **kwargs)
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        err_text = str(err)

    try:
        vhead_file = cached_file(filename="value_head.bin", **kwargs)
        return torch.load(vhead_file, map_location="cpu")
    except Exception as err:
        err_text = str(err)

    logger.info_rank0(f"Provided path ({path_or_repo_id}) does not contain value head weights: {err_text}.")
    logger.info_rank0("Ignore the above message if you are not resuming the training of a value head model.")
    return None


class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForTokenClassification

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path)
        
        self.processor = hf_processor(local_path, use_fast=True, **config.get("processor_kwargs", {}))  # used for multimodal LLM, could be none

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            # reward_module = AutoModelForTokenClassification.from_pretrained(
            #     pretrained_model_name_or_path=local_path,
            #     config=model_config,
            #     torch_dtype=torch.bfloat16,
            #     attn_implementation="flash_attention_2",
            #     trust_remote_code=trust_remote_code,
            # )
            if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                load_class = AutoModelForVision2Seq
            elif type(model_config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
            elif type(model_config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(model_config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen2_5_omni
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM
            
            print(f"loading reward model from {local_path}")
            
            reward_module = load_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            
            reward_module = AutoModelForCausalLMWithValueHead.from_pretrained(reward_module)
            vhead_params = load_valuehead_params(local_path)
            if vhead_params is not None:
                reward_module.load_state_dict(vhead_params, strict=False)
            
            print(f"Loaded valuehead from checkpoint: {local_path}")
            # TODO: path valuehead

            if config.model.get("use_remove_padding", False) or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=reward_module, ulysses_sp_size=self.ulysses_sequence_parallel_size)

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=False,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch, multi_modal_micro_batch):
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

        from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            pixel_values = multi_modal_micro_batch.get("pixel_values", None)
            image_grid_thw = multi_modal_micro_batch.get("image_grid_thw", None)
            
            assert (input_ids==151655).sum().item() == pixel_values.shape[0]//4, f"input_ids: {(input_ids==151655).sum().item()}, pixel_values: {pixel_values.shape[0]//4}"

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False)  # prevent model thinks we are generating
                _, _, reward_rmpad = output
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False)
                _, _, rm_score = output
                rm_score = rm_score.squeeze(-1)  # (bsz, seqlen)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores
    
    def _switch_rm_input(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]
        
        # system_prompt = (
        #     "你是一个经验丰富的排版设计师，擅长在用户给定的背景图上将指定的文本优雅地排版。\n"
        #     "你深知如何运用独特的美学原则设计出既专业又吸引人的排版布局，请根据用户给定的背景图和文本内容设计最终的排版方案，将排版好的图片直接提供给用户。\n"
        # )
        system_prompt = (
            "You are an experienced layout designer, skilled at elegantly arranging the specified text on the background image provided by the user.\n"
            "You know well how to use unique aesthetic principles to design a professional and appealing layout. Please design the final layout plan according to the background image and text content provided by the user, and directly provide the typeset image to the user.\n"
        )
        
        rm_input_ids = []
        rm_attention_mask = []
        rm_multi_modal_inputs = []
        rm_format_rewards = []
        
        responses = []
        
        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                chat: list = data.non_tensor_batch["raw_prompt"][i]
            else:
                chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()
            
            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # decode
            response = self.processor.decode(valid_response_ids)
            # remove bos and eos
            response: str = response.replace(self.processor.tokenizer.eos_token, "")
            responses.append(response)
        
        params = [(responses[i], data.non_tensor_batch["origin_multi_modal_data"][i]["image"][0]) for i in range(data.batch.batch_size[0])]
        
        with Pool(processes=min(data.batch.batch_size[0], os.cpu_count() // 8)) as pool:
            render_images = pool.starmap(may_convert_svg_to_png, params)
            
        for i in range(data.batch.batch_size[0]):
            text_content = chat[1]['content'][2]['text'].split("texts: \n")[1]
            messages = [
                {
                    'role': 'system',
                    'content': [
                        {
                            'type': 'text',
                            'text': system_prompt
                        }
                    ]
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': f"background-image.png:\n"
                        },
                        {
                            'type': 'image',
                            'image': data.non_tensor_batch["multi_modal_data"][i]["image"][0],
                            "max_pixels": 512 * 28 * 28
                        },
                        {
                            'type': 'text',
                            'text': f"\nPlease arrange the following text on the background image: \n{text_content}"
                        }
                    ]
                },
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'type': 'image',
                            'image': render_images[i][1],
                            "max_pixels": 512 * 28 * 28
                        }
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            image_inputs, video_inputs = process_vision_info(messages)
            model_inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length
            
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "error"),
            )  # truncate from the right
            
            assert (model_inputs["input_ids"] == 151655).sum().item() == (input_ids == 151655).sum().item(), f"input_ids: {(model_inputs['input_ids'] == 151655).sum().item()}, input_ids: {(input_ids == 151655).sum().item()}"
            
            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)
            # rm_pixel_values.append(model_inputs["pixel_values"])
            # rm_image_grid_thw.append(model_inputs["image_grid_thw"])
            rm_multi_modal_inputs.append({
                "pixel_values": model_inputs["pixel_values"],
                "image_grid_thw": model_inputs["image_grid_thw"],\
            })
            rm_format_rewards.append(render_images[i][0])
        
        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)
        # rm_pixel_values = torch.cat(rm_pixel_values, dim=0)
        # rm_image_grid_thw = torch.cat(rm_image_grid_thw, dim=0)
        
        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)
        
        rm_inputs = {
            "input_ids": rm_input_ids,
            "attention_mask": rm_attention_mask,
            "position_ids": rm_position_ids,
            # "pixel_values": rm_pixel_values,
            # "image_grid_thw": rm_image_grid_thw,
            "multi_modal_inputs": np.array(rm_multi_modal_inputs, dtype=object),
            "format_rewards": np.array(rm_format_rewards, dtype=object),
        }
        
        return DataProto.from_single_dict(rm_inputs)
        

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        
        rm_data = self._switch_rm_input(data)
        
        # Support all hardwares
        rm_data.batch = rm_data.batch.to(torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
                indices = [[i for i in range(idx, min(idx + self.config.micro_batch_size_per_gpu, len(rm_data.batch)))] 
                          for idx in range(0, len(rm_data.batch), self.config.micro_batch_size_per_gpu)]
            # first, we need to handle the multimodal data
            multi_modal_micro_batches = []
            for i, micro_batch in enumerate(micro_batches):
                micro_batch_indices = indices[i]
                multi_modal_micro_batches.append({})
                
                # Get the multimodal data for this micro batch
                pixel_values = []
                image_grid_thw = []
                
                for idx in micro_batch_indices:
                    if idx < len(rm_data.non_tensor_batch["multi_modal_inputs"]):
                        modal_input = rm_data.non_tensor_batch["multi_modal_inputs"][idx]
                        if "pixel_values" in modal_input:
                            pixel_values.append(modal_input["pixel_values"])
                        if "image_grid_thw" in modal_input:
                            image_grid_thw.append(modal_input["image_grid_thw"])
                
                if pixel_values:
                    multi_modal_micro_batches[i]["pixel_values"] = torch.cat(pixel_values, dim=0).to(rm_data.batch.device)
                if image_grid_thw:
                    multi_modal_micro_batches[i]["image_grid_thw"] = torch.cat(image_grid_thw, dim=0).to(rm_data.batch.device)
            
            output = []
            for i, micro_batch in enumerate(micro_batches):
                rm_score = self._forward_micro_batch(micro_batch, multi_modal_micro_batches[i])   # tensor([ 1.5312, -8.4375], device='cuda:0', dtype=torch.bfloat16)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]
            max_rm_score = scores.max()
            min_rm_score = scores.min()
            for i in range(len(scores)):
                if rm_data.non_tensor_batch['format_rewards'][i] == 'error':
                    scores[i] = min_rm_score
            token_level_scores = self._expand_to_token_level(data, scores)  # 把eos位置的scores设为rm_score，其他位置设为0
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output