python ./train/train.py \
	--data_dir=./data/10folds \
	--folds=10 \
	--test_path=./data/blp23_sentiment_test.tsv \
	--output_dir=./output/banglabert_large \
	--model_path=./ptm/csebuetnlp-banglabert_large/

