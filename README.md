# AbstractiveTextSummarization
Abstractive Text Summarization using Sequence-to-Sequence models

We train a sequence to sequence model on Amazon Food reviews dataset. The input sequence are food reviews and output sequence are title for the reviews. We have a stack of three LSTMs as encoder with the third LSTM being a Bidirectional LSTM. We use LSTM along with Bahandanau's attention as our decoder. We use a custom attention.py file to implement Attention. We extend the work in this [article](https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/)  using Bidirectional LSTMs and Beam Search as a decoder. 

### train_seq2seqmodel.ipynb:
notebook to load the dataset, preprocess text and create training and validation set and train a sequence to sequence model and save the model weights in a file.

### inference_seq2seqmodel.ipynb:
notebook to load the dataset, preproces text and create training and validation set and load the trained model and predict summaries. Contains examples where model works well and some examples where model fails.

## Good Examples


