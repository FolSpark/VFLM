# llamafactory-cli export merge_config.yaml

# PYTHONPATH=src/ python -m src.llamafactory.cli export scripts/lora_merge/llava_lora_sft.yaml

LORA_MERGE_DIR=work_dirs/llava1.5_7b_OpenCOLE_TextLayout_lora_2026-01-28/checkpoint-5120


PYTHONPATH=src/ python -m src.llamafactory.cli export \
    --model_name_or_path models/llava-hf/llava-1.5-7b-hf \
    --adapter_name_or_path ${LORA_MERGE_DIR} \
    --template llava \
    --trust_remote_code true \
    --export_dir ${LORA_MERGE_DIR}/lora_merged \
    --export_size 5 \
    --export_device cpu \
    --export_legacy_format false
