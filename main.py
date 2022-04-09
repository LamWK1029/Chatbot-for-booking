import data_collection
import tokenization
import train_model
import use_model

# print(data_collection.raw_domain_collect())
# print(data_collection.raw_intent_collect())
# print(data_collection.raw_collect_tag_data())
# print(data_collection.raw_key_word_collect())

# # domain model
# domain_tokenizer, domain_model_train, domain_model_test = data_collection.collect_domain_model_data()
# train_model.train_domain_models(domain_tokenizer, domain_model_train, domain_model_test)
# # # end of domain model
#
# # # intent model
# intent_tokenizer, intent_model_train, intent_model_test = data_collection.collect_intent_data()
# train_model.train_intent_models(intent_tokenizer, intent_model_train, intent_model_test)
# #
# # must in char level
# # tag model
tag_tokenizer_index, tag_model_train, tag_model_test = data_collection.collect_tag_data()
train_model.tag_model(tag_tokenizer_index, tag_model_train, tag_model_test)
#
# # # classify model for hotel
# classify_tokenizer, classify_model_train, classify_model_test = data_collection.collect_classify_data_for_hotel()
# train_model.train_classify_model_for_hotel(classify_tokenizer, classify_model_train, classify_model_test)
# #
# # # classify model for restaurant
# classify_tokenizer, classify_model_train, classify_model_test = data_collection.collect_classify_data_for_restaurant()
# train_model.classify_restaurant(classify_tokenizer, classify_model_train, classify_model_test)