from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import numpy


def tokenize_message(message_train, message_test):
    tokenizer = Tokenizer(num_words=3000, oov_token="<OOV>")   # dictionary with max 3000 word
    tokenizer.fit_on_texts(message_train)
    # word_index = tokenizer.word_index     # show the dictionary
    # print(word_index)
    message_train_in_number = tokenizer.texts_to_sequences(message_train)
    padded_tokenized_message_train = pad_sequences(message_train_in_number, padding='post', truncating='post', maxlen=25)

    message_test_in_number = tokenizer.texts_to_sequences(message_test)
    padded_tokenized_message_test = pad_sequences(message_test_in_number, padding='post', truncating='post', maxlen=25)
    return [tokenizer, padded_tokenized_message_train, padded_tokenized_message_test]


def tokenize_message_for_response(message_train, message_test):
    tokenizer = Tokenizer(600, oov_token="<OOV>")   # dictionary with max 3000 word
    tokenizer.fit_on_texts(message_train)
    word_index = tokenizer.word_index     # show the dictionary
    # print(word_index)
    message_train_in_number = tokenizer.texts_to_sequences(message_train)
    padded_tokenized_message_train = pad_sequences(message_train_in_number, padding='post', truncating='post', maxlen=35)

    message_test_in_number = tokenizer.texts_to_sequences(message_test)
    padded_tokenized_message_test = pad_sequences(message_test_in_number, padding='post', truncating='post', maxlen=35)
    return [tokenizer, padded_tokenized_message_train, padded_tokenized_message_test]



def tokenize_word(message_train, message_test):
    tokenizer = Tokenizer(num_words=3000, oov_token="<OOV>")   # dictionary with max 3000 word
    tokenizer.fit_on_texts(message_train)
    # word_index = tokenizer.word_index     # show the dictionary
    # print(word_index)

    message_train_in_number = tokenizer.texts_to_sequences(message_train)
    padded_tokenized_message_train = pad_sequences(message_train_in_number, padding='post', truncating='post', maxlen=1)

    message_test_in_number = tokenizer.texts_to_sequences(message_test)
    padded_tokenized_message_test = pad_sequences(message_test_in_number, padding='post', truncating='post', maxlen=1)
    # return [tokenizer, padded_tokenized_message_train, padded_tokenized_message_test]


def tokenize_char_level(message_train, message_test):
    # set type of char: <OOV>, a-z, A-Z, 0-9 (fit_on_text)
    tokenizer_index = {'<OOV>': 1}  # index 0 will be used to padded
    index_number = 2
    for letter in string.ascii_lowercase:
        tokenizer_index[letter] = index_number
        index_number += 1
    for letter in string.ascii_uppercase:
        tokenizer_index[letter] = index_number
        index_number += 1
    for number in range(10):
        tokenizer_index[str(number)] = index_number
        index_number += 1

    # change the char to number (fit on text) + pad_sequences(padded in 64) + numpy array
    message_train_in_number = []
    for word in message_train:  # can change to reduce the size of numpy array
        one_word = []
        for character in word:
            if character in tokenizer_index:
                one_word.append(tokenizer_index[character])
            else:
                one_word.append(tokenizer_index['<OOV>'])
        message_train_in_number.append(one_word)
    padded_tokenized_message_train = pad_sequences(message_train_in_number, padding='post', truncating='post', maxlen=12)

    # change the char to number (fit on text) + pad_sequences(padded in 64) + numpy array
    message_test_in_number = []
    for word in message_test:  # can change to reduce the size of numpy array
        one_word = []
        for character in word:
            if character in tokenizer_index:
                one_word.append(tokenizer_index[character])
            else:
                one_word.append(tokenizer_index['<OOV>'])
        message_test_in_number.append(one_word)
    padded_tokenized_message_test = pad_sequences(message_test_in_number, padding='post', truncating='post', maxlen=12)

    return [tokenizer_index, padded_tokenized_message_train, padded_tokenized_message_test]


def sentence_fit_on_text(tokenizer_index, message):
    # change the char to number (fit on text) + pad_sequences(padded in 64) + numpy array
    message_train_in_number = []
    message_in_word = message.split()
    for word in message_in_word:  # can change to reduce the size of numpy array
        one_word = []
        for character in word:
            if character in tokenizer_index:
                one_word.append(tokenizer_index[character])
            else:
                one_word.append(tokenizer_index['<OOV>'])
        message_train_in_number.append(one_word)
    return message_train_in_number


def sentence_fit_on_text_in_list(tokenizer_index, message_in_word):
    # change the char to number (fit on text) + pad_sequences(padded in 64) + numpy array
    message_train_in_number = []
    for word in message_in_word:  # can change to reduce the size of numpy array
        one_word = []
        for character in word:
            if character in tokenizer_index:
                one_word.append(tokenizer_index[character])
            else:
                one_word.append(tokenizer_index['<OOV>'])
        message_train_in_number.append(one_word)
    return message_train_in_number


def one_hot(word_index):
    index = int(word_index[0])
    # one-hot encoding
    one_hot_of_response_dic = []
    for _ in range(600):   # 600 is the number of dictionary size
        one_hot_of_response_dic.append(0)
    one_hot_of_response_dic[index] = 1
    return numpy.array(one_hot_of_response_dic)

