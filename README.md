# AlBert Chinese NER

基于ALBert的中文NER算法（命名实体识别）
来源：https://github.com/ProHiryu/albert-chinese-ner

## install

    pip install -r requirements.txt
    
## prepare dataset

准备数据集如目录（**./data/**）中所示。

## Model

https://github.com/MagicianQi/albert-chinese-ner/releases/download/v0.1/albert_base_zh.zip

## train and eval

    python albert_ner_train_eval.py \
        --task_name ner \
        --do_train true \
        --do_eval true \
        --data_dir data \
        --vocab_file ./albert_base_zh/vocab.txt \
        --bert_config_file ./albert_base_zh/albert_config_base.json \
        --max_seq_length 128 \
        --train_batch_size 64 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --output_dir albert_base_ner_checkpoints
   
## predict

详细见 **albert_ner_predict.py** 代码 658 行

    python albert_ner_predict.py
