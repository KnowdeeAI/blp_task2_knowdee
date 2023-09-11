python ./train/train.py \
	--data_dir=./data/icl_10folds \
	--folds=10 \
	--test_path=./data/icl_10folds/icl_blp23_sentiment_dev.tsv \
	--output_dir=./output/banglabert_large \
	--model_path=./ptm/csebuetnlp-banglabert_large/ \
	--icl \
	--max_seq_length=384