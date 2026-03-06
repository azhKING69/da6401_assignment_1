
python train.py \
  -d mnist \
  -e 50 \
  -b 128 \
  -l cross_entropy \
  -o rmsprop \
  -lr 0.0016 \
  -wd 0.0001 \
  -nhl 3 \
  -sz 128 128 128 \
  -a tanh \
  -w_i xavier \
  -w_p da6401_a1_



=============================================
          INFERENCE RESULTS
=============================================
  Dataset   : mnist
  Samples   : 10000
  Loss      : 0.0801
  Accuracy  : 0.9774
  Precision : 0.9772  (macro)
  Recall    : 0.9773  (macro)
  F1-Score  : 0.9772  (macro)
=============================================

Per-class report:
              precision    recall  f1-score   support

           0       0.97      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.97      0.98      1032
           3       0.98      0.97      0.97      1010
           4       0.98      0.97      0.98       982
           5       0.97      0.97      0.97       892
           6       0.97      0.99      0.98       958
           7       0.99      0.97      0.98      1028
           8       0.95      0.98      0.97       974
           9       0.98      0.97      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000



