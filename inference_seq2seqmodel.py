#!/usr/bin/env python
# coding: utf-8

# We train a sequence to sequence model on Amazon Food reviews dataset. The input sequence are food reviews and output sequence are title for the reviews. We have a stack of three LSTMs as encoder with the third LSTM being a Bidirectional LSTM. We use LSTM along with Bahandanau's attention as our decoder. We use a custom attention.py file to implement Attention.

# In[1]:


# Import libraries

from attention import AttentionLayer
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
nltk.download('stopwords')
from tensorflow.keras import backend as K


import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


# ### Read the dataset
# This dataset consists of reviews of fine foods from Amazon. The data spans a period of more than 10 years, including all ~500,000 reviews up to October 2012. These reviews include product and user information, ratings, plain text review, and summary. It also includes reviews from all other Amazon categories.
# 
# We train our model on 100,000 training samples to reduce the computational overhead on the GPU. Using more samples leads to memory allocation error on DSMLP.

# In[2]:


data=pd.read_csv("Reviews.csv", nrows=100000)


# ### Drop Duplicates and NA values

# In[3]:


data.drop_duplicates(subset=['Text'],inplace=True)
data.dropna(axis=0,inplace=True)#dropping na
data.head()


# #### Information about dataset
# 
# Let us look at datatypes and shape of the dataset

# In[4]:


data.info()


# ### Preprocessing
# 
# We perform basic preprocessing on the the text data to clean them. We use the below dictionary to expand our contractions.

# In[5]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}


# We will perform the below preprocessing tasks for our data:
# 
# 1.Convert everything to lowercase
# 
# 2.Remove HTML tags
# 
# 3.Contraction mapping
# 
# 4.Remove (‘s)
# 
# 5.Remove any text inside the parenthesis ( )
# 
# 6.Eliminate punctuations and special characters
# 
# 7.Remove stopwords
# 
# 8.Remove short words
# 

# In[6]:


stop_words = set(stopwords.words('english')) 

def text_cleaner(text,num):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num==0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens=newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:                                                 #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()


# In[7]:


# Clean reviews data
cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t,0)) 


# Let us look at the first five preprocessed reviews

# In[8]:


cleaned_text[:5]  


# In[9]:


#call the function
cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(text_cleaner(t,1))


# Let us look at the first 10 preprocessed summaries

# In[10]:


cleaned_summary[:10]


# In[11]:


data['cleaned_text']=cleaned_text
data['cleaned_summary']=cleaned_summary


# #Drop empty rows

# In[12]:


data.replace('', np.nan, inplace=True)
data.dropna(axis=0,inplace=True)


# ### Distibution of the sequences
# 
# We analyze the length of the reviews and the summary to estimate the distribution of the sequence length for reviews and titles. This helps us to decide the maximum sequence length for both reviews and titles.

# In[13]:


import matplotlib.pyplot as plt

text_word_count = []
summary_word_count = []

# populate the lists with sentence lengths
for i in data['cleaned_text']:
      text_word_count.append(len(i.split()))

for i in data['cleaned_summary']:
      summary_word_count.append(len(i.split()))

length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})

length_df.hist(bins = 30)
plt.show()


# Interesting. We can fix the maximum length of the summary to 8 since that seems to be the majority summary length.
# 
# Let us understand the proportion of the length of summaries below 8

# In[14]:


cnt=0
text_len = 30
sum_len = 8
for i in data['cleaned_summary']:
    if(len(i.split())<=sum_len):
        cnt=cnt+1
print(cnt/len(data['cleaned_summary']))


# We observe that 94% of the summaries have length below 8. So, we can fix maximum length of summary to 8.
# 
# Similarly fixing the maximum length of review to 60 seems sound.

# In[15]:


max_text_len=60
max_summary_len=8


# Let us select the reviews and summaries whose length falls below or equal to **max_text_len** and **max_summary_len**

# In[16]:


cleaned_text =np.array(data['cleaned_text'])
cleaned_summary=np.array(data['cleaned_summary'])

short_text=[]
short_summary=[]

for i in range(len(cleaned_text)):
    if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])
        
df=pd.DataFrame({'text':short_text,'summary':short_summary})
df.head()


# Next we will add start and end tokens at the beginning and end of each summary in the datset. Make sure that this tokens don't exist in our vocabulary.

# In[17]:


df['summary'] = df['summary'].apply(lambda x : 'sostok '+ x + ' eostok')


# We split the data into training and validation set. We use the validation set to decide later when to stop training. (Early Stopping).

# In[18]:


from sklearn.model_selection import train_test_split
x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['text']),np.array(df['summary']),test_size=0.1,random_state=0,shuffle=True) 


# ### Tokenizer
# A tokenizer builds the vocabulary and converts a word sequence to an integer sequence. 

# In[19]:


from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences

#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_tr))


# ### Rarewords and its Coverage
# 
# We look at the proportion of the rare words and their coverage.
# 
# We define threshold as 4 which means word whose count is below 4 is considered as a rare word

# In[20]:


thresh=4

cnt=0
tot_cnt=0
freq=0
tot_freq=0

for key,value in x_tokenizer.word_counts.items():
    tot_cnt=tot_cnt+1
    tot_freq=tot_freq+value
    if(value<thresh):
        cnt=cnt+1
        freq=freq+value
    
print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
print("Total Coverage of rare words:",(freq/tot_freq))
print(tot_cnt-cnt)


# 
# * **tot_cnt** gives the size of vocabulary (which means every unique words in the text)
#  
# *   **cnt** gives me the no. of rare words whose count falls below threshold
# 
# *  **tot_cnt - cnt** gives me the top most common words 
# 
# Let us define the tokenizer with top most common words for reviews.

# In[21]:


#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
x_tokenizer.fit_on_texts(list(x_tr))

#convert text sequences into integer sequences
x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)

#padding zero upto maximum length
x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
x_val   =   pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

#size of vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1


# ### Tokenizer for Summaries (titles)

# In[22]:


#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_tr))


# ### Rarewords and its Coverage
# 
# Proportion rare words and its total coverage in the entire summary
# 
# Threshold is 6 which means word whose count is below 6 is considered as a rare word

# In[23]:


thresh=6

cnt=0
tot_cnt=0
freq=0
tot_freq=0

for key,value in y_tokenizer.word_counts.items():
    tot_cnt=tot_cnt+1
    tot_freq=tot_freq+value
    if(value<thresh):
        cnt=cnt+1
        freq=freq+value
    
print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
print("Total Coverage of rare words:",(freq/tot_freq)*100)


# Define a tokenizer with the top most frequent words 

# In[24]:


#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
y_tokenizer.fit_on_texts(list(y_tr))

#convert text sequences into integer sequences
y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr) 
y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) 

#padding zero upto maximum length
y_tr    =   pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val   =   pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

#size of vocabulary
y_voc  =   y_tokenizer.num_words +1


# Delete rows that contain only **START** and **END** tokens

# In[25]:


ind=[]
for i in range(len(y_tr)):
    cnt=0
    for j in y_tr[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_tr=np.delete(y_tr,ind, axis=0)
x_tr=np.delete(x_tr,ind, axis=0)


# In[26]:


ind=[]
for i in range(len(y_val)):
    cnt=0
    for j in y_val[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_val=np.delete(y_val,ind, axis=0)
x_val=np.delete(x_val,ind, axis=0)


# # Model 
# 
# We define a sequence to sequence model using a encoder-decoder architecture. We use a Embedding Matrix and a stack of three LSTMs as our encoder. The third LSTM is Bidirectional LSTM. The decoer is an LSTM decoder along with Bahadanau's attention. 
# We use sparse categorical crossentropy as our loss since it converts the integer sequence to a one-hot vector on the fly. This overcomes any memory issues.
# 

# In[27]:


from tensorflow.keras import backend as K 

K.clear_session()

latent_dim = 128
embedding_dim=100

# Encoder
encoder_inputs = Input(shape=(max_text_len,))

#encoder embedding layer
enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)

#encoder lstm 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output1 = encoder_lstm1(enc_emb)

#encoder lstm 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output2 = encoder_lstm2(encoder_output1)

#encoder lstm 3 Bidirectional LSTM
encoder_lstm3=Bidirectional(LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4), merge_mode='concat')
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm3(encoder_output2)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])



decoder_inputs = Input(shape=(None,))

#embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

# decoder lstm
decoder_lstm = LSTM(2*latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(dec_emb,initial_state=[state_h, state_c])


# Attention layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])



# Concat attention input and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#dense layer
decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary() 


# In[28]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')


# In[41]:



file_path2 = "./pretrained_models/model_best_val_bi001.h5"
model.load_weights(file_path2)


# 

# In[42]:



reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index


# # Inference
# 
# Set up the inference for the encoder and decoder:

# In[43]:


# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(2*latent_dim,))
decoder_state_input_c = Input(shape=(2*latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len,2*latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs) 
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat) 

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])


# We are defining a function below which is the implementation of the inference process (which we covered [here](https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/)):

# In[44]:


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


# ### Beam Search Decoder

# In[45]:


def beam_search(model, src_input, k=3, sequence_max_len=8):
    k_beam = [(0, [target_word_index['sostok']])]
    for l in range(sequence_max_len):
        all_k_beams = []
        for prob, sent_predict in k_beam:
            predicted = model.predict([np.array([src_input]), np.array([sent_predict])])[0]
            # top k!
            possible_k = predicted[l].argsort()[-k:][::-1]

            all_k_beams += [
                (
                    sum(np.log(predicted[i][sent_predict[i+1]]) for i in range(l)) + np.log(predicted[l][next_wid]),
                    list(sent_predict[:l+1])+[next_wid]+[0]*(sequence_max_len-l-1)
                )
                for next_wid in possible_k
            ]

#         # top k
        k_beam = sorted(all_k_beams)[-k:]

    return k_beam


# In[46]:


def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


# Here are a few summaries generated by the model:

# In[47]:


for i in range(0,30):
    print("Index: ",i)
    print("Review:",seq2text(x_val[i]))
    print("Original summary:",seq2summary(y_val[i]))
    print("Greedy summary:",decode_sequence(x_val[i].reshape(1,max_text_len)))
    print("Beam Search Result: ",[seq2summary(sent) for prob, sent in beam_search(model,x_val[i])])
    print("\n")


# ### Good Examples

# In[54]:


good_examples = [3, 7, 9,10, 16, 17, 23, 24, 29, 32, 35, 44, 58]
for ind,i in enumerate(good_examples):
    print("Review {}: {}\n".format(ind+1,seq2text(x_val[i])))
    print("Original summary: {}\n".format(seq2summary(y_val[i])))
    print("Greedy Decoder Result: {}\n".format(decode_sequence(x_val[i].reshape(1,max_text_len))))
    print("Beam Search Result: {}\n".format([seq2summary(sent) for prob, sent in beam_search(model,x_val[i])]))
    print("\n")


# ### Not so good examples

# In[52]:


examples = [ 19, 21]
for ind,i in enumerate(examples):
    print("Review{}: {}\n".format(ind+1,seq2text(x_val[i])))
    print("Original summary:{}\n".format(seq2summary(y_val[i])))
    print("Greedy Decoder Result:{}\n".format(decode_sequence(x_val[i].reshape(1,max_text_len))))
    print("Beam Search Result:{}\n".format([seq2summary(sent) for prob, sent in beam_search(model,x_val[i])]))
    print("\n")


# ## Try your own example

# In[50]:


sentence = ' Tea is okay. Pasta is bad.'
sent1 = text_cleaner(sentence,1)
x_seq = x_tokenizer.texts_to_sequences([sent1]) 
x = pad_sequences(x_seq,  maxlen=max_text_len, padding='post')
print("Greedy Decoder Result:{}\n".format(decode_sequence(x.reshape(1,max_text_len))))
print("Beam Search Result:{}\n".format([seq2summary(sent) for prob, sent in beam_search(model,x[0])]))

