import numpy
import tokenization
from tokenization import tokenize_word
import tensorflow as tf
import numpy as np
from data_collection import collect_conversation
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import data_collection
import tokenization
import train_model
from train_model import plot_model_accurate

# Preprocessing
text_generation_dataset = collect_conversation()    # collect dataset
dataset_for_hotel = text_generation_dataset[0]
dataset_for_restaurant = text_generation_dataset[1]

# data collect and setting part
# label the message need to response or not
# if system response is '', then the label will be 0 (means no need to response)
user_input_message, system_output_message = [], []
user_input_response_label, turn_stage_with_messages = [], []
for conversation in dataset_for_hotel:
    user_input_message.append(conversation['hotel_user_transcript'])
    system_output_message.append(conversation['hotel_system_transcript'])
    turn_stage_with_messages.append({"turn_id": conversation['hotel_turn_id'],
                                     "user_input": conversation['hotel_user_transcript'],
                                     "system_output": conversation['hotel_system_transcript']
                                     })
    if conversation['hotel_system_transcript'] == '':
        user_input_response_label.append(0)     # 0 means no need to response
    else:
        user_input_response_label.append(1)     # 1 means need to response
for conversation in dataset_for_restaurant:
    user_input_message.append(conversation['restaurant_user_transcript'])
    system_output_message.append(conversation['restaurant_system_transcript'])
    turn_stage_with_messages.append({"turn_id": conversation['restaurant_turn_id'],
                                     "user_input": conversation['restaurant_user_transcript'],
                                     "system_output": conversation['restaurant_system_transcript']
                                     })
    if conversation['restaurant_system_transcript'] == '':
        user_input_response_label.append(0)     # 0 means no need to response
    else:
        user_input_response_label.append(1)     # 1 means need to response


# split data for train and test (train response or not)
user_input_response_train = user_input_message[:8000]
user_input_response_train_label = user_input_response_label[:8000]
user_input_response_test = user_input_message[8000:]
user_input_response_test_label = user_input_response_label[8000:]

# Dictionary    !!! Tokenizer !!!
tokenizer_for_response, padded_tokenized_user_response_train, padded_tokenized_user_response_test \
    = tokenization.tokenize_message(user_input_response_train, user_input_response_test)
# train model
response_train = [padded_tokenized_user_response_train, user_input_response_train_label]
response_test = [padded_tokenized_user_response_test, user_input_response_test_label]
# train_model.train_response_or_not_models(tokenizer_for_response, response_train, response_test)


# stage check model (using functional structure)
stage_user_message, stage_system_message, stage_label = [], [], []
for turn in turn_stage_with_messages:
    stage_user_message.append(turn['user_input'])
    stage_system_message.append(turn['system_output'])
    if turn['turn_id'] == 0:
        turn_stage_label = [1, 0, 0]
    elif turn['system_output'] == '':
        turn_stage_label = [0, 0, 1]
    else:
        turn_stage_label = [0, 1, 0]
    stage_label.append(turn_stage_label)

user_input_stage_train = stage_user_message[:8000]
user_input_stage_train_label = stage_label[:8000]
user_input_stage_test = stage_user_message[8000:]
user_input_stage_test_label = stage_label[8000:]

# add 'start=(index3)' and 'end'=(index2) in the corresponding position
stage_system_message_with_tag = []
for sentence in stage_system_message:
    if sentence != '':
        stage_system_message_with_tag.append('start ' + sentence + ' end') # no /start, /end since not in all sentence
system_input_stage_train = stage_system_message_with_tag[:6000]
system_input_stage_test = stage_system_message_with_tag[6000:]

# 2 Dictionary
# user dictionary
tokenizer_for_stage_for_user, padded_tokenized_user_stage_train, padded_tokenized_user_stage_test \
    = tokenization.tokenize_message(user_input_stage_train, user_input_stage_test)
tokenizer_for_stage_for_system, padded_tokenized_system_stage_train, padded_tokenized_system_stage_test \
    = tokenization.tokenize_message_for_response(system_input_stage_train, system_input_stage_test)

# # train model stage   (check message need to response or not)
# stage_train = [padded_tokenized_user_stage_train, user_input_stage_train_label, user_input_response_train_label]
# stage_test = [padded_tokenized_user_stage_test, user_input_stage_test_label, user_input_response_test_label]
# train_model.message_stage_model(tokenizer_for_stage_for_user, stage_train, stage_test)

# train response model
# for system
all_sentence_predict_pair, user_input_index = [], 0
for sentence in system_input_stage_train:
    all_round_message, previous_message = [], ''
    sentence_in_list = sentence.split()
    for word in sentence_in_list:
        if word == 0:
            break
        tokenized_transcript = tokenizer_for_stage_for_system.texts_to_sequences([previous_message])
        tokenized_transcript_word = tokenizer_for_stage_for_system.texts_to_sequences([word])
        padded_tokenized_transcript = pad_sequences(tokenized_transcript, maxlen=35, padding='post', truncating='post')
        padded_tokenized_keyword = pad_sequences(tokenized_transcript_word, maxlen=1, padding='post', truncating='post')
        all_round_message.append({'user_input': padded_tokenized_user_stage_train[user_input_index],
                                  'message': padded_tokenized_transcript[0], 'predict_word': tokenization.one_hot(padded_tokenized_keyword[0])})
        previous_message = previous_message + word + ' '
    user_input_index += 1
    all_sentence_predict_pair.append(all_round_message)
user_input_index = 0
for sentence in system_input_stage_test:
    all_round_message, previous_message = [], ''
    sentence_in_list = sentence.split()
    for word in sentence_in_list:
        if word == 0:
            break
        tokenized_transcript = tokenizer_for_stage_for_system.texts_to_sequences([previous_message])
        tokenized_transcript_word = tokenizer_for_stage_for_system.texts_to_sequences([word])
        padded_tokenized_transcript = pad_sequences(tokenized_transcript, maxlen=35, padding='post', truncating='post')
        padded_tokenized_keyword = pad_sequences(tokenized_transcript_word, maxlen=1, padding='post', truncating='post')
        all_round_message.append({'user_input': padded_tokenized_user_stage_test[user_input_index],
                                  'message': padded_tokenized_transcript[0], 'predict_word': tokenization.one_hot(padded_tokenized_keyword[0])})
        previous_message = previous_message + word + ' '
    all_sentence_predict_pair.append(all_round_message)
print(len(all_sentence_predict_pair))
# collect all pair as array
data_dic_train = all_sentence_predict_pair[:5000]
data_dic_test = all_sentence_predict_pair[5000:]
user_input_pair_train, system_message_train, predict_word_train = [], [], []
for sentence in data_dic_train:
    for turn in sentence:
        user_input_pair_train.append(turn['user_input'])
        system_message_train.append(turn['message'])
        predict_word_train.append(turn['predict_word'])
user_input_pair_test, system_message_test, predict_word_test = [], [], []
for sentence in data_dic_test:
    for turn in sentence:
        user_input_pair_test.append(turn['user_input'])
        system_message_test.append(turn['message'])
        predict_word_test.append(turn['predict_word'])

# train model
# size: user_input = 25, generated_input = 35, output_probobility = 2514
print("Train message generate model")
vocab_size_for_user_input = len(tokenizer_for_stage_for_user.word_index)+1
vocab_size_for_system_response = len(tokenizer_for_stage_for_system.word_index)+1
user_message_input = tf.keras.Input(shape=(25,), name="user_input")
system_accrue_message_input = tf.keras.Input(shape=(35,), name="system_accrue_message_input")
user_message_features = tf.keras.layers.Embedding(vocab_size_for_user_input, 256)(user_message_input)
user_message_features = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(user_message_features)
system_message_features = tf.keras.layers.Embedding(vocab_size_for_system_response, 256)(system_accrue_message_input)
system_message_features = tf.keras.layers.LSTM(32)(system_message_features)
combine_layer = tf.keras.layers.concatenate([user_message_features, system_message_features])
control_range_layer = tf.keras.layers.Dense(64, activation='linear')(combine_layer)
predict_output = tf.keras.layers.Dense(100, name="predict_output", activation='relu')(control_range_layer)
response_model = tf.keras.Model(inputs=[user_message_input, system_accrue_message_input],
                                outputs=[predict_output], )
response_model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['acc'])
print(response_model.summary())
tf.keras.utils.plot_model(response_model, 'stage_model.png', show_shapes=True)
#
# # problem start at here
history = response_model.fit({"user_input": np.array(user_input_pair_train), "system_accrue_message_input": np.array(system_message_train)},
                             np.array(predict_word_train), epochs=1,
                             validation_data=({"user_input": np.array(user_input_pair_test), "system_accrue_message_input": np.array(system_message_test)},
                                       np.array(predict_word_test)), verbose=2)
plot_model_accurate(history)    # plot graph
response_model.save('system_response.h5')

# test the model
user_input = ["am looking for a place to to stay that has cheap price range it should be in a type of hotel"
              ,"no, i just need to make sure it's cheap. oh, and i need parking"
              ,"yes, please. 6 people 3 nights starting on tuesday."
              ]
user_input_sentence = tokenizer_for_stage_for_user.texts_to_sequences(user_input)
system_output = ['okay, do you have'
                 ,"i found 1 cheap hotel for you that includes"
                 ,"i am sorry but i wasn't able to book that for you for"
                 ]
system_output_sentence = tokenizer_for_stage_for_system.texts_to_sequences(system_output)

# input layer data text
padded_user_input = pad_sequences(user_input_sentence, maxlen=25, padding='post', truncating='post')
padded_predicted_sentence = pad_sequences(system_output_sentence, maxlen=35, padding='post', truncating='post')
print(response_model.predict({"user_input": padded_user_input, "system_accrue_message_input": padded_predicted_sentence}))
print(tokenizer_for_stage_for_system.word_index)
