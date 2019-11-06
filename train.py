import pandas as pd
from numpy import stack, mean, std
from keras.utils import to_categorical
from keras.models import Sequential, load_model #Model
from keras.layers import Dense, Flatten, Dropout #Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate

n_timesteps = 10

train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

def transform(df):
    for i in ['x', 'y', 'z']:
        df[i] = df[i] - df[i].shift(1)
    df = df.iloc[1:,:]
    pre_x, x = df[['x', 'y', 'z']].values, []
    for i in range(len(pre_x)-n_timesteps):
        x.append(pre_x[i:i+n_timesteps])
    x = stack(x)
    y = to_categorical(df.iloc[10:,-1])
    print(x.shape, y.shape)
    return x, y

trainX, trainY = transform(train_df)
testX, testY = transform(test_df)
n_features, n_outputs = trainX.shape[2], trainY.shape[1]

def evaluate_model(trainX, trainY, testX, testY):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(testX, testY, batch_size=32, verbose=0)
    # print(f'Accuracy: {accuracy}')
    model.save("model.h5")
    return model, accuracy

# model, accuracy = evaluate_model(trainX, trainY, testX, testY)
model = load_model("model.h5")
pred = model.predict_classes(testX).tolist()
pred = [0 for i in range(11)] + pred
test_df['pred_state'] = pred
test_df['correct_pred'] = test_df['state'] == test_df['pred_state']
print(test_df.to_string())# print(test_df.iloc[10:,-1].to_string())

# TO DO
# Comment code, put together github repo, and markdown file for Tom
# Convert to .nn with below link
# https://github.com/Unity-Technologies/ml-agents/blob/3d7c4b8d3c1ad17070308b4e06bb57d4a80f9a0c/ml-agents/mlagents/trainers/tensorflow_to_barracuda.py

# scores = []
# for i in range(10):
#     _, score = evaluate_model(trainX, trainY, testX, testY)
#     score = score * 100.0
#     print('>#%d: %.3f' % (i+1, score))
#     scores.append(score)
# print(scores)
# m, s = mean(scores), std(scores)
# print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
