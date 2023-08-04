import pandas as pd
import re
import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import tensorflow as tf

csv_path = './data/arXiv-DataFrame.csv'
df = pd.read_csv(csv_path)


title = df['Title']
genre = df['Primary Category']
genre_dict = {}
for i in genre:
    if i not in genre_dict.values():
        genre_dict[i] = len(genre_dict)
print(genre_dict)

X_data = []
Y_data = []
for i in title:
    X_data.append(i)
for i in genre:
    Y_data.append(genre_dict[i])

print(X_data[:10])
print(Y_data[:10])

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_data)
word_indexs = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(X_data)    # 해당 단어를 가리키는 숫자를 단어 사전(tokenizer)에서 가져온다.
input_sequence = pad_sequences(sequences, padding='post')  # post padding 사용 (0을 뒤에 추가)

input_sequence = np.array(input_sequence)
Y_data = np.array(Y_data)
Y_data = keras.utils.to_categorical(Y_data, num_classes=len(genre_dict))

max_length = max(len(sequence) for sequence in sequences)
# 패딩 적용
padded_sequences = pad_sequences(sequences, maxlen=max_length)

x_train ,x_test, y_train, y_test = train_test_split(padded_sequences, Y_data,test_size=0.2, random_state=0)


batch_size = 100    # 데이터 묶음 크기
num_epochs = 1000   # 반복 수
vocab_size = len(tokenizer.word_index)+1    # 단어 사전의 크기
emb_size = 120      # 단어를 밀집 벡터로 변환할 때 백터의 차원 크기를 결정
hidden_dim = 256    # 은닉층의 개수
output_dim = 6


class genre_division_model(keras.Model):
    def __init__(self, vocab_size, emb_size, hidden_dim, output_dim):
        super(genre_division_model, self).__init__(name='subclassing')
        self.embedding = keras.layers.Embedding(vocab_size, emb_size, name='embedding')
        self.hidden = keras.layers.Dense(hidden_dim, 'relu', name='hidden')
        self.outputs = keras.layers.Dense(output_dim, 'sigmoid', name='outputs')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.reduce_mean(x, axis=1)
        x = self.hidden(x)
        x = self.outputs(x)

        return x


division = genre_division_model(vocab_size, emb_size, hidden_dim, output_dim)

division.build(input_shape=(1, 17,))
division.summary()

earlystop_callback = EarlyStopping(
    monitor='val_loss',  # 모니터링할 지표
    min_delta=0.0001,    # 개선되는 것으로 판단하기 위한 최소 변화량
    patience=3,          # 개선이 없는 에포크를 얼마나 기다릴 것인가
    verbose=1,           # 로그를 출력
    restore_best_weights=True  # 가장 좋은 가중치를 복원
)

division.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = division.fit(x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=num_epochs,
            batch_size=batch_size,
            callbacks=[earlystop_callback])

print(f'{division.evaluate(x_test, y_test)[1]:.4f}')