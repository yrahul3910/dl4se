#!/usr/bin/env python
# coding: utf-8

# DeepTriage code from https://bugtriage.mybluemix.net/#code
#
# Some changes were made:
# * **Data:** - Adjusted to conform to the DASENet data
# * **Architecture:** - Simplified slightly; output changed to DASENet settings.

# In[4]:


import string
import nltk
import re
import json
from tensorflow.keras.callbacks import EarlyStopping
import tempfile
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Wrapper, InputSpec, BatchNormalization, Dense, TimeDistributed, Bidirectional, LSTM, Input, Dropout
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from gensim.models import Word2Vec
from nltk.corpus import wordnet
import numpy as np
np.random.seed(1337)
#from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, Bidirectional, TimeDistributed

# In[6]:


nltk.download('punkt')


# In[24]:


bugs_json = './chromium.json'


# The hyperparameters required for the entire code can be initialized upfront as follows:

# In[2]:


# 1. Word2vec parameters
min_word_frequency_word2vec = 5
embed_size_word2vec = 200
context_window_word2vec = 5

# 2. Classifier hyperparameters
numCV = 10
max_sentence_len = 50
min_sentence_length = 15
rankK = 10
batch_size = 256


# The bugs are loaded from the JSON file and the preprocessing is performed as follows:

# In[25]:


with open(bugs_json) as data_file:
    data = json.load(data_file, strict=False)

all_data = []
all_owner = []
all_y = []
for item in data:
    # 1. Remove \r
    current_title = item['issue_title'].replace('\r', ' ')
    current_desc = item['description'].replace('\r', ' ')
    # 2. Remove URLs
    current_desc = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)
    # 3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    # 4. Remove hex code
    current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
    current_title = re.sub(r'(\w+)0x\w+', '', current_title)
    # 5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()
    # 6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    # 7. Strip trailing punctuation marks
    current_desc_filter = [word.strip(string.punctuation)
                           for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation)
                            for word in current_title_tokens]
    # 8. Join the lists
    current_data = current_title_filter + current_desc_filter
    current_data = filter(None, current_data)
    all_data.append(current_data)
    all_owner.append(item['owner'])
    all_y.append(item['y'])


# In[26]:


all_data = [list(x) for x in all_data]


# A vocabulary is constructed and the word2vec model is learnt using the preprocessed data. The word2vec model provides a semantic word representation for every word in the vocabulary.

# In[27]:


wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec,
                         size=embed_size_word2vec, window=context_window_word2vec)
vocabulary = wordvec_model.wv.vocab
vocab_size = len(vocabulary)


# The ten times chronological cross validation split is performed as follows:

# In[28]:


totalLength = len(all_data)
splitLength = int(totalLength / (numCV + 1))

for i in range(1, numCV+1):
    train_data = all_data[:i*splitLength-1]
    test_data = all_data[i*splitLength:(i+1)*splitLength-1]
    train_owner = all_owner[:i*splitLength-1]
    test_owner = all_owner[i*splitLength:(i+1)*splitLength-1]
    train_y = all_y[:i*splitLength-1]
    test_y = all_y[i*splitLength:(i+1)*splitLength-1]


# For the ith cross validation set, remove all the words that is not present in the vocabulary

# In[29]:


i = 1  # Denotes the cross validation set number
updated_train_data = []
updated_train_data_length = []
updated_train_owner = []
updated_train_y = []
final_test_data = []
final_test_owner = []
final_test_y = []
for j, item in enumerate(train_data):
    current_train_filter = [word for word in item if word in vocabulary]
    if len(current_train_filter) >= min_sentence_length:
        updated_train_data.append(current_train_filter)
        updated_train_owner.append(train_owner[j])
        updated_train_y.append(train_y[j])

for j, item in enumerate(test_data):
    current_test_filter = [word for word in item if word in vocabulary]
    if len(current_test_filter) >= min_sentence_length:
        final_test_data.append(current_test_filter)
        final_test_owner.append(test_owner[j])
        final_test_y.append(test_y[j])


# For the ith cross validation set, remove those classes from the test set, for whom the train data is not available.

# In[30]:


i = 1  # Denotes the cross validation set number
# Remove data from test set that is not there in train set
train_owner_unique = set(updated_train_owner)
test_owner_unique = set(final_test_owner)
train_y_unique = set(updated_train_y)
test_y_unique = set(final_test_y)
unwanted_owner = list(test_owner_unique - train_owner_unique)
unwanted_y = list(test_y_unique - train_y_unique)
updated_test_data = []
updated_test_owner = []
updated_test_y = []
updated_test_data_length = []
for j in range(len(final_test_owner)):
    if final_test_owner[j] not in unwanted_owner:
        updated_test_data.append(final_test_data[j])
        updated_test_owner.append(final_test_owner[j])
        updated_test_y.append(final_test_y[j])

unique_train_label = list(set(updated_train_owner))
unique_train_ys = list(set(updated_train_y))
classes = np.array(unique_train_ys)


# In[31]:


X_train = np.empty(shape=[len(updated_train_data),
                          max_sentence_len, embed_size_word2vec], dtype='float32')
Y_train = np.empty(shape=[len(updated_train_y), 1], dtype='int32')
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
for j, curr_row in enumerate(updated_train_data):
    sequence_cnt = 0
    for item in curr_row:
        if item in vocabulary:
            X_train[j, sequence_cnt, :] = wordvec_model[item]
            sequence_cnt = sequence_cnt + 1
            if sequence_cnt == max_sentence_len-1:
                break
    for k in range(sequence_cnt, max_sentence_len):
        X_train[j, k, :] = np.zeros((1, embed_size_word2vec))
    Y_train[j, 0] = unique_train_ys.index(updated_train_y[j])

X_test = np.empty(shape=[len(updated_test_data),
                         max_sentence_len, embed_size_word2vec], dtype='float32')
Y_test = np.empty(shape=[len(updated_test_y), 1], dtype='int32')
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
for j, curr_row in enumerate(updated_test_data):
    sequence_cnt = 0
    for item in curr_row:
        if item in vocabulary:
            X_test[j, sequence_cnt, :] = wordvec_model[item]
            sequence_cnt = sequence_cnt + 1
            if sequence_cnt == max_sentence_len-1:
                break
    for k in range(sequence_cnt, max_sentence_len):
        X_test[j, k, :] = np.zeros((1, embed_size_word2vec))
    Y_test[j, 0] = unique_train_ys.index(updated_test_y[j])

y_train = np_utils.to_categorical(Y_train, len(unique_train_ys))
y_test = np_utils.to_categorical(Y_test, len(unique_train_ys))


# In[84]:


# From https://github.com/tensorflow/tensorflow/issues/34697#issuecomment-623332746
# Hotfix function

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


# Run the function
make_keras_picklable()

# In[71]:


def make_safe(x):
    return K.clip(x, K.common._EPSILON, 1.0 - K.common._EPSILON)


class ProbabilityTensor(Wrapper):
    """ function for turning 3d tensor to 2d probability matrix, which is the set of a_i's """

    def __init__(self, dense_function=None, *args, **kwargs):
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        #layer = TimeDistributed(dense_function) or TimeDistributed(Dense(1, name='ptensor_func'))
        layer = TimeDistributed(Dense(1, name='ptensor_func'))
        super(ProbabilityTensor, self).__init__(layer, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if K.backend() == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis.')

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ProbabilityTensor, self).build()

    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,n
        #       s.t. \sum_n n = 1
        if isinstance(input_shape, (list, tuple)) and not isinstance(input_shape[0], int):
            input_shape = input_shape[0]

        return (input_shape[0], input_shape[1])

    def squash_mask(self, mask):
        if K.ndim(mask) == 2:
            return mask
        elif K.ndim(mask) == 3:
            return K.any(mask, axis=-1)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        return self.squash_mask(mask)

    def call(self, x, mask=None):
        energy = K.squeeze(self.layer(x), 2)
        p_matrix = K.softmax(energy)
        if mask is not None:
            mask = self.squash_mask(mask)
            p_matrix = make_safe(p_matrix * mask)
            p_matrix = (p_matrix / K.sum(p_matrix,
                                         axis=-1, keepdims=True))*mask
        return p_matrix

    def get_config(self):
        config = {}
        base_config = super(ProbabilityTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SoftAttentionConcat(ProbabilityTensor):
    '''This will create the context vector and then concatenate it with the last output of the LSTM'''

    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,f where f is weighted features summed across n
        return (input_shape[0], 2*input_shape[2])

    def compute_mask(self, x, mask=None):
        if mask is None or mask.ndim == 2:
            return None
        else:
            raise Exception("Unexpected situation")

    def call(self, x, mask=None):
        # b,n,f -> b,f via b,n broadcasted
        p_vectors = K.expand_dims(
            super(SoftAttentionConcat, self).call(x, mask), 2)
        expanded_p = K.repeat_elements(p_vectors, K.int_shape(x)[2], axis=2)
        context = K.sum(expanded_p * x, axis=1)
        last_out = x[:, -1, :]
        return K.concatenate([context, last_out])


# In[97]:


inp = Input(shape=(max_sentence_len, embed_size_word2vec))
#sequence_embed = Embedding(vocab_size, embed_size_word2vec, input_length=max_sentence_len)(inp)

forwards_1 = Bidirectional(
    LSTM(1024, return_sequences=True, recurrent_dropout=0.2))(inp)
attention_1 = SoftAttentionConcat()(forwards_1)
after_dp_forward_5 = BatchNormalization()(attention_1)

after_merge = Dense(1000, activation='relu')(after_dp_forward_5)
after_dp = Dropout(0.4)(after_merge)
output = Dense(2, activation='softmax')(after_dp)
model = Model(inputs=inp, outputs=output)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


# In[98]:


model.summary()


# Train the deep learning model and test using the classifier as follows:

# In[96]:


y_train = np.argmax(y_train, axis=-1)
y_test = np.argmax(y_test, axis=-1)

y_train = np.where(y_train < 2, 0, np.where(y_train < 6, 1, 2))
y_test = np.where(y_test < 2, 0, np.where(y_test < 6, 1, 2))

y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)


# In[99]:


early_stopping = EarlyStopping(monitor='loss', patience=5)
hist = model.fit(X_train, y_train, batch_size=batch_size,
                 epochs=200, callbacks=[early_stopping])

predict = model.predict(X_test)
predict_classes = np.argmax(predict, axis=-1)
y_test = np.argmax(y_test, axis=-1)

print('Predictions:', predict_classes)
print('Actual:', y_test)

acc = sum(predict_classes == y_test) / len(y_test)
print('Accuracy =', round(acc * 100, 2), '\b%')

# Top-2 accuracy


def top_2(pred_probs, y_test):
    """
    Returns top-2 accuracy.

    :param pred_probs - Prediction probabilities in shape (n_samples, n_classes)
    :param y_test - Actual target labels of 1-D shape.
    """
    best_n = np.argsort(pred_probs, axis=-1)[:, -2:]
    correct = 0
    total = len(y_test)

    for i, pred in enumerate(best_n):
        if y_test[i] in pred:
            correct += 1
    print('Top-2 accuracy =', round(correct / total, 3))


top_2(predict, y_test)

train_result = hist.history
print(train_result)

f = open('deeptriage.pkl', 'wb')
pickle.dump(model, f)
f.close()


# In[ ]:
