python ./train/train_t5.py \
	--data_dir='./data/10folds/' \
	--test_dir='./data/blp23_sentiment_dev.tsv' \
	--output_dir='./output/mt5/' \
	--model_path='./ptm/google-mt5-large/' \
	--folds=10