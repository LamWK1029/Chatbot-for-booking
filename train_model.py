import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import data_collection
import tokenization


def train_domain_models(tokenizer, domain_model_train, domain_model_test):
    print("Train domain model")
    # decode data
    domain_message_train, domain_label_train = domain_model_train[0], domain_model_train[1]
    print(domain_message_train)
    print(domain_label_train)
    print()

    domain_message_test, domain_label_test = domain_model_test[0], domain_model_test[1]
    # model structure
    domain_define_model = tf.keras.Sequential()
    print(len(domain_message_train))
    print(domain_message_train.shape)

    vocab_size = len(tokenizer.word_index)+1   # word embedding
    embedding_dim = 8
    message_word_number = 25    # max message length
    domain_define_model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=message_word_number))
    domain_define_model.add(tf.keras.layers.Flatten())  # output_shape = 200
    domain_define_model.add(tf.keras.layers.Dense(128, activation='relu'))
    domain_define_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # output
    # end of structure
    domain_define_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(domain_define_model.summary())
    tf.keras.utils.plot_model(domain_define_model, 'domain_define_model.png', show_shapes=True)

    history = domain_define_model.fit(domain_message_train, np.array(domain_label_train), epochs=6
                                      , validation_data=(domain_message_test, np.array(domain_label_test)), verbose=2)
    plot_model_accurate(history)    # plot graph
    # domain_define_model.save('domain.h5')


def train_intent_models(tokenizer, intent_model_train, intent_model_test):
    print("Train intent model")
    # decode data
    intent_message_train, intent_label_train = intent_model_train[0], intent_model_train[1]
    intent_message_test, intent_label_test = intent_model_test[0], intent_model_test[1]

    # model structure
    intent_classification_model = tf.keras.Sequential()

    vocab_size = len(tokenizer.word_index) + 1  # word embedding
    embedding_dim = 32
    message_word_number = 25  # max message length
    intent_classification_model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=message_word_number))
    intent_classification_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)))  # long term dependence in both side
    intent_classification_model.add(tf.keras.layers.Dense(32, activation='linear'))
    intent_classification_model.add(tf.keras.layers.Dense(16, activation='relu'))
    intent_classification_model.add(tf.keras.layers.Dense(8, activation='relu'))
    intent_classification_model.add(tf.keras.layers.Dense(4, activation='sigmoid'))  # output
    # end of structure
    intent_classification_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(intent_classification_model.summary())
    tf.keras.utils.plot_model(intent_classification_model, 'intent_classification_model.png', show_shapes=True)


    history = intent_classification_model.fit(intent_message_train, np.array(intent_label_train), epochs=7
                                              , validation_data=(intent_message_test, np.array(intent_label_test)), verbose=2)
    plot_model_accurate(history)    # plot graph
    # intent_classification_model.save('intent.h5')


def tag_model(tag_tokenizer_index, tag_model_train, tag_model_test):
    print("Train tag model")
    tag_words_train, tag_label_train_label = tag_model_train[0], tag_model_train[1]
    tag_words_test, tag_label_test_label = tag_model_test[0], tag_model_test[1]

    # model structure
    tag_word_model = tf.keras.Sequential()
    vocab_size = len(tag_tokenizer_index) + 1  # word embedding
    embedding_dim = 8
    message_word_number = 12  # max message length
    tag_word_model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=message_word_number))
    tag_word_model.add(tf.keras.layers.Flatten())
    tag_word_model.add(tf.keras.layers.Dense(64, activation='linear'))
    tag_word_model.add(tf.keras.layers.Dense(32, activation='linear'))
    tag_word_model.add(tf.keras.layers.Dense(8, activation='relu'))
    tag_word_model.add(tf.keras.layers.Dense(2, activation='relu'))
    tag_word_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # output
    # end of structure
    tag_word_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(tag_word_model.summary())
    tf.keras.utils.plot_model(tag_word_model, 'tag_word_model.png', show_shapes=True)

    history = tag_word_model.fit(np.array(tag_words_train), np.array(tag_label_train_label), epochs=18
                                              , validation_data=(np.array(tag_words_test), np.array(tag_label_test_label)), verbose=2)
    plot_model_accurate(history)  # plot graph
    tf.keras.utils.plot_model(tag_word_model, 'tag_word_model.png', show_shapes=True)
    tag_word_model.save('tag.h5')


def train_classify_model_for_hotel(tokenizer, classify_model_train, classify_model_test):
    print("Train classify model for hotel")
    train_data, train_label = classify_model_train[0], classify_model_train[1]
    test_data, test_label = classify_model_test[0], classify_model_test[1]
    print(len(train_data), len(train_label))
    print(len(test_data), len(test_label ))

    # model structure
    classify_hotel_model = tf.keras.Sequential()
    vocab_size = len(tokenizer) + 1  # word embedding
    embedding_dim = 8
    message_word_number = 12
    classify_hotel_model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=message_word_number))
    classify_hotel_model.add(tf.keras.layers.SimpleRNN(64, return_sequences=False))  # long term dependence in both side
    classify_hotel_model.add(tf.keras.layers.Dense(32, activation='linear'))
    classify_hotel_model.add(tf.keras.layers.Dense(16, activation='relu'))
    classify_hotel_model.add(tf.keras.layers.Dense(3, activation='sigmoid'))  # output
    # end of structure
    classify_hotel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(classify_hotel_model.summary())
    tf.keras.utils.plot_model(classify_hotel_model, 'classify_hotel_model.png', show_shapes=True)

    history = classify_hotel_model.fit(np.array(train_data), np.array(train_label), epochs=10
                                           , validation_data=(np.array(test_data), np.array(test_label)), verbose=2)
    plot_model_accurate(history)  # plot graph
    # classify_hotel_model.save('models/classify_for_hotel.h5')


def classify_restaurant(tokenizer, classify_model_train, classify_model_test):
    print("Train classify model for restaurant")
    train_data, train_label = classify_model_train[0], classify_model_train[1]
    test_data, test_label = classify_model_test[0], classify_model_test[1]
    print(len(train_data), len(train_label))
    print(len(test_data), len(test_label ))

    # model structure
    classify_hotel_model = tf.keras.Sequential()
    vocab_size = len(tokenizer) + 1  # word embedding
    embedding_dim = 8
    message_word_number = 12
    classify_hotel_model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=message_word_number))
    classify_hotel_model.add(tf.keras.layers.SimpleRNN(64, return_sequences=False))  # long term dependence in both side
    classify_hotel_model.add(tf.keras.layers.Dense(32, activation='linear'))
    classify_hotel_model.add(tf.keras.layers.Dense(16, activation='relu'))
    classify_hotel_model.add(tf.keras.layers.Dense(4, activation='sigmoid'))  # output
    # end of structure
    classify_hotel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(classify_hotel_model.summary())
    tf.keras.utils.plot_model(classify_hotel_model, 'classify_hotel_model.png', show_shapes=True)

    history = classify_hotel_model.fit(np.array(train_data), np.array(train_label), epochs=10
                                           , validation_data=(np.array(test_data), np.array(test_label)), verbose=2)
    plot_model_accurate(history)  # plot graph
    # classify_hotel_model.save('models/classify_for_restaurant.h5')


def plot_model_accurate(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# Model for generating text for response
# model setup
def train_response_or_not_models(tokenizer, response_or_not_model_train, response_or_not_model_test):
    print("Train response or not model")
    # decode data
    response_message_train, response_label_train = response_or_not_model_train[0], response_or_not_model_train[1]
    response_message_test, response_label_test = response_or_not_model_test[0], response_or_not_model_test[1]

    # model structure
    response_or_not_model = tf.keras.Sequential()
    print(len(response_message_train))
    print(response_message_train.shape)
    vocab_size = len(tokenizer.word_index)+1   # word embedding
    embedding_dim = 8
    message_word_number = 25    # max message length
    # response_or_not_model.add(tf.keras.layers.Flatten())  # output_shape = 200
    response_or_not_model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=message_word_number))
    response_or_not_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False)))
    response_or_not_model.add(tf.keras.layers.Dense(64, activation='linear'))
    response_or_not_model.add(tf.keras.layers.Dense(32, activation='relu'))
    response_or_not_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # output
    # end of structure
    response_or_not_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(response_or_not_model.summary())
    tf.keras.utils.plot_model(response_or_not_model, 'response_or_not_model.png', show_shapes=True)

    history = response_or_not_model.fit(response_message_train, np.array(response_label_train), epochs=5,
                                    validation_data=(response_message_test, np.array(response_label_test))
                                    , verbose=2)
    plot_model_accurate(history)    # plot graph
    # response_or_not_model.save('response_or_not.h5')

    # test the model    0 = not reply, 1 = reply
    sentence = ["hello, I would like to book a hotel", "No, thank you. Goodbye", "I have four people", "help"]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=25, padding='post', truncating='post')
    print(response_or_not_model.predict(padded))


def message_stage_model(tokenizer, response_or_not_model_train, response_or_not_model_test):
    print("Train message stage model")
    user_message_input_train, stage_label_train, response_label_train = np.array(response_or_not_model_train[0]), \
                                                                       np.array(response_or_not_model_train[1]), \
                                                                       np.array(response_or_not_model_train[2])
    user_message_input_test, stage_label_test, response_label_test = np.array(response_or_not_model_test[0]), \
                                                                       np.array(response_or_not_model_test[1]), \
                                                                       np.array(response_or_not_model_test[2])
    print(user_message_input_train.shape)
    print(response_label_train.shape)
    vocab_size = len(tokenizer.word_index)+1

    user_message_input = tf.keras.Input(shape=(25,), name="user_input")
    system_response_or_not_input = tf.keras.Input(shape=(1,), name="response_or_not_input")
    user_message_features = tf.keras.layers.Embedding(vocab_size, 8)(user_message_input)
    user_message_features = tf.keras.layers.LSTM(32)(user_message_features)
    combine_layer = tf.keras.layers.concatenate([user_message_features, system_response_or_not_input])
    control_range_layer = tf.keras.layers.Dense(8, activation='linear')(combine_layer)
    stage_output = tf.keras.layers.Dense(3, name="stage_output", activation='relu')(control_range_layer)
    stage_model = tf.keras.Model(
        inputs=[user_message_input, system_response_or_not_input],
        outputs=[stage_output],
    )
    stage_model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['acc'])
    print(stage_model.summary())
    tf.keras.utils.plot_model(stage_model, 'stage_model.png', show_shapes=True)

    # problem start at here
    history = stage_model.fit({"user_input": user_message_input_train, "response_or_not_input": response_label_train},
                              np.array(stage_label_train), epochs=10,
                              validation_data=({"user_input": user_message_input_test,
                                                "response_or_not_input": response_label_test},
                                               np.array(stage_label_test)), verbose=2)
    # plot_model_accurate(history)    # plot graph
    # stage_model.save('stage.h5')

    # test the model    0 = not reply, 1 = reply
    sentence = ["hello, I would like to book a hotel", "ys, 4 people", "Thanks, bye"]
    label = np.array([1, 1, 0])
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=25, padding='post', truncating='post')
    print(
        stage_model.predict({"user_input": padded, "response_or_not_input": label})
    )


# train new text classification model
def categories_model(tokenizer, categories_train_dataset, categories_test_dataset):
    padded_tokenized_message_train, categories_train_label = categories_train_dataset[0], categories_train_dataset[1]
    padded_tokenized_message_test, categories_test_label = categories_test_dataset[0], categories_test_dataset[1]

    # model structure
    categories_model = tf.keras.Sequential()
    print("shape: ", padded_tokenized_message_train.shape)
    vocab_size = len(tokenizer.word_index) + 1  # word embedding
    embedding_dim = 32
    message_word_number = 25  # max message length
    categories_model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=message_word_number))
    categories_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)))
    categories_model.add(tf.keras.layers.Dense(32, activation='linear'))
    categories_model.add(tf.keras.layers.Dense(16, activation='linear'))
    categories_model.add(tf.keras.layers.Dense(8, activation='relu'))
    categories_model.add(tf.keras.layers.Dense(4, activation='sigmoid'))  # output
    # end of structure
    categories_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(categories_model.summary())
    tf.keras.utils.plot_model(categories_model, 'categories_model.png', show_shapes=True)

    # problem start at here
    history = categories_model.fit(padded_tokenized_message_train, np.array(categories_train_label), epochs=7,
                                   validation_data=(padded_tokenized_message_test, np.array(categories_test_label)),
                                   verbose=2)
    plot_model_accurate(history)  # plot graph
    categories_model.save('categories_model.h5')

    # [price-range, name, time, food]
    # test the model    0 = not reply, 1 = reply
    sentence = ["Book a cheap hotel for me", "Book a hotel at Tseung Kwan O", "Book a table for chinese food at 12:00"]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=25, padding='post', truncating='post')
    print(categories_model.predict(padded))