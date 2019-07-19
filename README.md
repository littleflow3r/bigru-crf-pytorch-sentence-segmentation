#A minimal PyTorch implementation of bidirectional LSTM-CRF for sequence labelling.

Supported features:
- Lookup, CNNs, RNNs and/or self-attentive encoding in the embedding layer
- A PyTorch implementation of conditional random field (CRF)
- Vectorized computation of CRF loss
- Vectorized Viterbi decoding
- Mini-batch training with CUDA

## Usage

Training data should be formatted as below:
```
token/tag token/tag token/tag ...
token/tag token/tag token/tag ...
...
```

To prepare data:
```
python3 prepare.py training_data

# word segmentation
python3 char+iob.py training_data
python3 prepare.py training_data.char+iob

# sentence segmentation
python3 word+iob.py training_data
python3 prepare.py training_data.word+iob
```

To train:
```
python3 train.py model char_to_idx word_to_idx tag_to_idx training_data.csv (validation_data) num_epoch
```

To predict:
```
python3 predict.py model.epochN char_to_idx word_to_idx tag_to_idx test_data
```
