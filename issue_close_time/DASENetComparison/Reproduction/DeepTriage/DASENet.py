#!/usr/bin/env python
# coding: utf-8

# ## Data

# In[1]:


import pickle
from keras.utils import to_categorical
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, LSTM, Dense, BatchNormalization, merge, Input, LeakyReLU, Flatten, Reshape
from gensim.models.fasttext import FastText
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from raise_utils.data import Data
import gc
import sys
import time

# In[2]:


df = pd.read_csv(f'./Bug-Related-Activity-Logs/{sys.argv[1]}.csv')
df.drop(['Unnamed: 0', 'bugID'], axis=1, inplace=True)

_df = df[['s1', 's2', 's3', 's4', 's5', 's6', 's8', 'y']]
_df['s70'] = df['s7'].apply(lambda x: eval(x)[0])
_df['s71'] = df['s7'].apply(lambda x: eval(x)[1])
_df['s72'] = df['s7'].apply(lambda x: eval(x)[2])

_df['s90'] = df['s9'].apply(lambda x: eval(x)[0])
_df['s91'] = df['s9'].apply(lambda x: eval(x)[1])

if len(df['s9'][0]) == 3:
    _df['s92'] = df['s9'].apply(lambda x: eval(x)[2])

x = _df.drop('y', axis=1)
y = _df['y']

data = Data(*train_test_split(x, y))
data.y_train = data.y_train < (5 if sys.argv[1] == 'chromium' else 4 if sys.argv[1] == 'firefox' else 6)
data.y_test = data.y_test < (5 if sys.argv[1] == 'chromium' else 4 if sys.argv[1] == 'firefox' else 6)

# ## Word embeddings
print('Generating word embeddings...', end='')

x_user_raw = [eval(x) for x in df['user_comments']]
x_user_raw_train, x_user_raw_test = train_test_split(x_user_raw)

user_model = FastText(sentences=x_user_raw_train, sg=1, size=200)

x_sys_raw = [eval(x) for x in df['system_records']]
x_sys_raw_train, x_sys_raw_test = train_test_split(x_sys_raw)

system_model = FastText(sentences=x_sys_raw_train, sg=1, size=200)

maxlen = 50
x_user_train = [t[:50] if len(
    t) >= 50 else t + (['end'] * (50 - len(t))) for t in x_user_raw_train]

x_sys_train = [t[:50] if len(t) >= 50 else t +
               (['end'] * (50 - len(t))) for t in x_sys_raw_train]

x_user_train = [[user_model[word] for word in arr] for arr in x_user_train]
x_user_train = np.array(x_user_train)

x_sys_train = np.array([[system_model[word] for word in arr]
                        for arr in x_sys_train])

x_user_test = [t[:50] if len(t) >= 50 else t +
               (['end'] * (50 - len(t))) for t in x_user_raw_test]

x_sys_test = [t[:50] if len(t) >= 50 else t +
              (['end'] * (50 - len(t))) for t in x_sys_raw_test]

x_user_test = np.array([[user_model[word] for word in arr] for arr in x_user_test])
x_sys_test = np.array([[system_model[word] for word in arr] for arr in x_sys_test])

print('done!', flush=True)

del x_sys_raw_train, x_sys_raw_test, x_user_raw_train, x_user_raw_test
gc.collect()

user_activity_input = Input(shape=x_user_train.shape[1:])

user_activity_stream_hidden_layer = Bidirectional(
    LSTM(128, return_sequences=True, input_shape=(
        x_user_train.shape[1], x_user_train.shape[2]))
)(user_activity_input)

user_activity_stream = Bidirectional(
    LSTM(256, return_sequences=True, input_shape=(x_user_train.shape[1], 256))
)(user_activity_stream_hidden_layer)


# In[36]:


sys_activity_input = Input(shape=x_sys_train.shape[1:])

sys_activity_stream_hidden_layer = Bidirectional(
    LSTM(128, return_sequences=True, input_shape=x_sys_train.shape[1:])
)(sys_activity_input)

sys_activity_stream = Bidirectional(
    LSTM(32, return_sequences=True, input_shape=(x_sys_train.shape[1], 256))
)(sys_activity_stream_hidden_layer)


# In[37]:


meta_input = Input(shape=data.x_train.shape[1:])

meta_hidden_layer = Dense(50)(meta_input)
meta_hidden_activation = LeakyReLU()(meta_hidden_layer)

meta = Dense(30)(meta_hidden_activation)
meta = LeakyReLU()(meta)


# ## Concatenate

# In[38]:


flatten1 = Flatten()(user_activity_stream)
flatten2 = Flatten()(sys_activity_stream)

concat_layer = merge.Concatenate(axis=-1)([flatten1, flatten2, meta])


# ## Merging MLP layer

# In[39]:


mlp_layer1 = Dense(380, input_shape=())(concat_layer)
mlp_activ1 = LeakyReLU()(mlp_layer1)

mlp_layer2 = Dense(450)(mlp_activ1)
mlp_activ2 = LeakyReLU()(mlp_layer2)

mlp_layer3 = Dense(260)(mlp_activ2)
mlp_activ3 = LeakyReLU()(mlp_layer3)

mlp_layer4 = Dense(200)(mlp_activ3)
mlp_activ4 = LeakyReLU()(mlp_layer4)

mlp_out = Dense(200)(mlp_activ4)
mlp_out = LeakyReLU()(mlp_out)


# ## Bin-sequence embedding

# In[40]:


# from https://gist.github.com/skeeet/b639eea7e3fc51dd03e9b69c06b2fdf1
def make_residual_lstm_layers(input, rnn_width, rnn_depth):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    x = input
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1
        x_rnn = LSTM(rnn_width, return_sequences=return_sequences)(input)
        x_rnn = BatchNormalization()(x_rnn)
        if return_sequences:
            # residual block
            x = merge.Add()([x, x_rnn])
        else:
            # last layer does not return sequences and cannot be residual
            x = x_rnn
    return x


# In[41]:


reshape = Reshape((1, 200))(mlp_out)


# In[42]:


dasenet = make_residual_lstm_layers(reshape, 200, rnn_depth=8)


# In[43]:


dasenet.shape


# In[44]:


n_classes = 2
final_layer = Dense(n_classes, activation='softmax')(dasenet)


# In[45]:


# In[46]:


model = Model(inputs=[user_activity_input,
                      sys_activity_input, meta_input], outputs=[final_layer])


# In[47]:


print(model.summary())


# ## Fit

# In[48]:


# In[49]:

start = time.time()
model.compile(optimizer='adam', loss='categorical_crossentropy')


# In[50]:


data.y_train = to_categorical(data.y_train)


# In[51]:


data.y_test = to_categorical(data.y_test)


# In[52]:


x_user_train.shape, x_sys_train.shape, data.x_train.shape


# In[54]:


model.fit([x_user_train, x_sys_train, data.x_train], data.y_train, epochs=100)

end = time.time()

with open(f'model-{sys.argv[1]}.pkl', 'wb') as f:
    pickle.dump(model, f)

#with open('model.pkl', 'rb') as f:
#    model = pickle.load(f)
preds = model.predict([x_user_test, x_sys_test, data.x_test])

preds = np.argmax(preds, axis=-1)

with open(f'model-{sys.argv[1]}-outs.pkl', 'wb') as f:
    pickle.dump({
        'preds': preds,
        'y_test': np.argmax(data.y_test, axis=-1)
    }, f)

print('Completed in', (end - start), 'seconds')
