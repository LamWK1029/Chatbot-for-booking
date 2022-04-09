from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tokenization


def domain_using_ml(tokenizer, use_input):
    model = keras.models.load_model('models/domain.h5')
    sentence = [use_input]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=25, padding='post', truncating='post')
    return model.predict(padded)


def intent_using_ml(tokenizer, use_input):
    model = keras.models.load_model('models/intent.h5')
    sentence = [use_input]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=25, padding='post', truncating='post')
    return model.predict(padded)


def tag_using_ml(tag_tokenizer_index, user_input):
    model = keras.models.load_model('models/tag.h5')
    sentence = user_input
    words_for_test = tokenization.sentence_fit_on_text(tag_tokenizer_index, sentence)
    padded = pad_sequences(words_for_test, maxlen=12, padding='post', truncating='post')
    return model.predict(padded)


def classify_using_ml_hotel(tag_tokenizer_index, user_input):
    model = keras.models.load_model('models/classify_for_hotel.h5')
    words_for_test = tokenization.sentence_fit_on_text_in_list(tag_tokenizer_index, user_input)
    padded = pad_sequences(words_for_test, maxlen=12, padding='post', truncating='post')
    return model.predict(padded)


def classify_using_ml_restaurant(tag_tokenizer_index, user_input):
    model = keras.models.load_model('models/classify_for_restaurant.h5')
    words_for_test = tokenization.sentence_fit_on_text_in_list(tag_tokenizer_index, user_input)
    padded = pad_sequences(words_for_test, maxlen=12, padding='post', truncating='post')
    return model.predict(padded)


