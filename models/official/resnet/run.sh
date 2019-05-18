export PYTHONPATH=$PYTHONPATH:/home/yangyouucb/tpu-1/models/
export TPU_NAME=v3-256-4tb
export INFER_TPU_NAME=infer

gsutil rm -R -f gs://bert-pretrain-data/imagenet/$INFER_TPU_NAME/*
python3 resnet_main.py --tpu=$TPU_NAME --data_dir=gs://bert-pretrain-data/imagenet/imagenet-2012-tfrecord/ --model_dir=gs://bert-pretrain-data/imagenet/$INFER_TPU_NAME/ --num_cores=256 --train_batch_size=8192 --enable_lars=True --train_steps=14074 --mode=train
python3 resnet_main.py --tpu=$INFER_TPU_NAME --data_dir=gs://bert-pretrain-data/imagenet/imagenet-2012-tfrecord/ --model_dir=gs://bert-pretrain-data/imagenet/$INFER_TPU_NAME/ --num_cores=8 --train_batch_size=8192 --enable_lars=True --train_steps=14074 --mode=eval
