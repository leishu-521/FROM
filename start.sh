python -u train.py --cfg experiments/CASIA-112x96-LMDB-Mask.yaml \
                    --model 'LResNet50E_IR_FPN'\
                    --batch_size 8 \
                    --gpus '0' \
                    --lr 0.01 \
                    --weight_pred 1  \
                    --optim 'sgd'\
                    --ratio 3 \
                    --pattern 5 \
                    --debug 0
