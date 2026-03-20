## SFT 训练
1.  构建数据格式及转换parquet格式 参考 examples/data_preprocess/distill_r1_110k.py  
2. [可选] 修改model的cah拼接 apply_chat_template
3. 初始化多机环境examples/ray/tengxun/run_worker8.sh，单节点上启动 examples/sft/think/run_qwen05b_128k_8nodes_tx.sh
