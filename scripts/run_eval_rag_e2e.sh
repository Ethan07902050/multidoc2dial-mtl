#!/bin/sh

export PYTHONPATH="../":"${PYTHONPATH}"
# domain=$1 # all dmv va ssa studentaid
# seg=$2  # token structure
# score=$3 # original reranking reranking_original
# task=$4 # grounding generation
# split=$5 # val test
domain=all
seg=structure
score=original
task=generation
split=val

dpr=dpr-$domain-$seg
DATA_DIR=../data/mdd_$domain/dd-$task-$seg
KB_FOLDER=../data/mdd_kb/knowledge_dataset-$dpr
MODEL_PATH=$CHECKPOINTS/hps
PRED_PATH=$MODEL_PATH/generation_step_5363_preds.txt

python rag/eval_rag.py \
--model_type rag_token_dialdoc \
--scoring_func $score \
--gold_pid_path $DATA_DIR/$split.pids \
--passages_path $KB_FOLDER/my_knowledge_dataset \
--index_path $KB_FOLDER/my_knowledge_dataset_index.faiss \
--index_name dialdoc \
--n_docs 5 \
--model_name_or_path $MODEL_PATH \
--eval_mode e2e \
--evaluation_set $DATA_DIR/$split.source \
--gold_data_path $DATA_DIR/$split.target \
--gold_data_mode ans \
--predictions_path $PRED_PATH \
--eval_batch_size 2 \
--num_beams 8