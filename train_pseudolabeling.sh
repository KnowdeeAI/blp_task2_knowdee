python ./train/train.py \
	--data_dir=./data/10folds \
	--folds=10 \
	--test_path=./data/blp23_sentiment_test.tsv \
	--output_dir=./output/banglabert_large \
	--model_path=./ptm/csebuetnlp-banglabert_large/ \

python ./data_process/pseudo_labelling.py \
	--predict_file_dir=./output/banglabert_large/predict/ \
	--output_dir=./output/banglabert_large/ \
	--folds=10

python ./train/train.py \
	--data_dir=./data/10folds \
	--folds=10 \
	--test_path=./data/blp23_sentiment_dev.tsv \
	--output_dir=./output/banglabert_large_pl \
	--model_path=./ptm/csebuetnlp-banglabert_large/ \
	--pseudo_labeling=./output/banglabert_large/pseudo_labelling.csv

python ./train/predict.py \
	--test_path=./data/blp23_sentiment_test.tsv \
	--model_dir=./output/banglabert_large_pl \
	--save_path=./output/banglabert_large_pl/final_predict \
	--folds=10 \
	--model_type=class \
	--batch_size=32 \
	--max_seq_length=128