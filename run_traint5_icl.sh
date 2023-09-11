python ./train/train_t5.py \
	--data_dir='./data/icl_10folds/' \
	--test_dir='./data/icl_10folds/blp23_sentiment_dev.tsv' \
	--output_dir='./output/mt5/' \
	--model_path='./ptm/google-mt5-large/' \
	--folds=10 \
	--icl \
	--max_seq_length=384