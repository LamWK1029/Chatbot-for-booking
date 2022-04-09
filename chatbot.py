import use_model
import data_collection
from search_target import get_hotel


# loop begin
user_input = input("Message: ") # read user input

# use model01 to distinct hotel or restaurant
domain_tokenizer, domain_model_train, domain_model_test = data_collection.collect_domain_model_data()   # need to optimize
domain = use_model.domain_using_ml(domain_tokenizer, user_input)    # user domain = 0(hotel), 1(restaurant)


if domain < 0.5:    # user would like to book a hotel
    # variables for hotel information
    hotel_price, hotel_location, hotel_name = [], [], []

    perfect_fit, good_fit, ok_fit = [], [], []
    print("System: User required to book hotel")

    # repeat per round
    while True:
        # check the word used to search using ML
        intent_one_hot = use_model.intent_using_ml(domain_tokenizer, user_input)
        # print("Intent:", intent_one_hot)
        intent = max(intent_one_hot[0][0], intent_one_hot[0][1], intent_one_hot[0][2])
        intent_index = list(intent_one_hot[0]).index(intent)   # user intent = [price, location, name]

        # tag the important word using ML
        classify_tokenizer, classify_model_train, classify_model_test = data_collection.collect_classify_data_for_restaurant()
        important_word_one_hot = use_model.tag_using_ml(classify_tokenizer, user_input)    # if > 0.8 means word meaningful
        # print("Tag:", important_word_one_hot)
        important_word_index, tagged_word = [], []
        for input_index in range(len(important_word_one_hot)):
            if important_word_one_hot[input_index] >= 0.9:      # collect important word to a list
                important_word_index.append(input_index)
        user_input_list = user_input.split()
        for word_index in range(len(user_input_list)):
            if word_index in important_word_index:
                tagged_word.append(user_input_list[word_index])  # the tagged word

        # classify the important word using ML
        classify_word_one_hot_list = use_model.classify_using_ml_hotel(classify_tokenizer, tagged_word)
        # set a for loop and append the word by type
        word_type_index = 0
        for word_one_hot in classify_word_one_hot_list:
            # check max value and it's index
            word_type = max(word_one_hot[0], word_one_hot[1], word_one_hot[2])
            type_index = list(word_one_hot).index(word_type)  # classify_word_one_hot = [price, location, name]
            if type_index == 0:
                hotel_price.append(tagged_word[word_type_index])
            elif type_index == 1:
                hotel_location.append(tagged_word[word_type_index])
            else:
                hotel_name.append(tagged_word[word_type_index])
            word_type_index += 1    # move the tagged index
        print("System collected: Hotel Price-", hotel_price,
              "Hotel Location-",hotel_location,"Hotel Name-",hotel_name,"\n")

        # search by price, Location, Name
        fit_price_hotel, fit_location_hotel, fit_name_hotel = [], [], []
        if hotel_price:
            for price in hotel_price:
                fit_price_hotel = get_hotel("Price", price)
        if hotel_location:
            for location in hotel_location:
                fit_location_hotel = get_hotel("Premises Address", location)
        if hotel_name:
            for name in hotel_location:
                fit_name_hotel = get_hotel("Premises Name", name)

        # Check the price, location, name in the same list or not
        if intent_index == 0: # search by price
            for target_hotel in fit_price_hotel:    # checked by set method
                if (target_hotel in fit_location_hotel) and (target_hotel in fit_name_hotel):
                    perfect_fit.append(target_hotel)
                elif (target_hotel in fit_location_hotel) or (target_hotel in fit_name_hotel):
                    good_fit.append(target_hotel)
                else:
                    ok_fit.append(target_hotel)
        if intent_index == 1:  # search by location
            for target_hotel in fit_location_hotel:  # checked by set method
                if (target_hotel in fit_price_hotel) and (target_hotel in fit_name_hotel):
                    perfect_fit.append(target_hotel)
                elif (target_hotel in fit_price_hotel) or (target_hotel in fit_name_hotel):
                    good_fit.append(target_hotel)
                else:
                    ok_fit.append(target_hotel)
        if intent_index == 2:  # search by name
            for target_hotel in fit_location_hotel:  # checked by set method
                if (target_hotel in fit_price_hotel) and (target_hotel in fit_location_hotel):
                    perfect_fit.append(target_hotel)
                elif (target_hotel in fit_price_hotel) or (target_hotel in fit_location_hotel):
                    good_fit.append(target_hotel)
                else:
                    ok_fit.append(target_hotel)

        # show 3 hotels to the customer if here is 2 column filled
        if (hotel_price and hotel_location) or (hotel_price and hotel_name) or (hotel_location and hotel_name):
            show_hotel = []
            while len(show_hotel) != 3:
                if perfect_fit:
                    show_hotel.append(perfect_fit.pop())
                elif good_fit:
                    show_hotel.append(good_fit.pop())
                elif ok_fit:
                    show_hotel.append(ok_fit.pop())
                elif fit_price_hotel:
                    show_hotel.append(fit_price_hotel.pop())
                elif fit_location_hotel:
                    show_hotel.append(fit_location_hotel.pop())
                elif fit_name_hotel:
                    show_hotel.append(fit_name_hotel.pop())
                elif not fit_price_hotel and not fit_location_hotel and not fit_name_hotel:
                    print("No hotel searched")
                    break
            break
        print("We found too many hotels that fit you need.\nCan you provide more information? ")
        user_input = input("Message: ")  # read user input


    # Chatbot Responses
    print("We found hotel that fit for you", show_hotel)
    selected_hotel = int(input("Please select the hotel you want, so we can book it for you.\nSelect:"))
    if selected_hotel > 0 and selected_hotel <= 3:
        print(show_hotel[selected_hotel-1], "be selected.")








else:
    print("System: User required to book restaurant")
    # check the word used to search using ML
    intent_one_hot = use_model.intent_using_ml(domain_tokenizer, user_input)
    intent = max(intent_one_hot[0])
    intent_index = list(intent_one_hot[0]).index(intent)  # user intent = [price, location, name, food]

    # tag the important word using ML
    classify_tokenizer, classify_model_train, classify_model_test = data_collection.collect_classify_data_for_restaurant()
    important_word_one_hot = use_model.tag_using_ml(classify_tokenizer, user_input)  # if > 0.8 means word meaningful

    important_word_index, tagged_word = [], []
    for input_index in range(len(important_word_one_hot)):
        if important_word_one_hot[input_index] >= 0.9:  # collect important word to a list
            important_word_index.append(input_index)
    user_input_list = user_input.split()
    for word_index in range(len(user_input_list)):
        if word_index in important_word_index:
            tagged_word.append(user_input_list[word_index])  # the tagged word

    # classify the important word using ML
    classify_word_one_hot_list = use_model.classify_using_ml_restaurant(classify_tokenizer, tagged_word)
    # set a for loop and append the word by type
    restaurant_price, restaurant_location, restaurant_name, food_type = [], [], [], []
    word_type_index = 0
    for word_one_hot in classify_word_one_hot_list:
        # check max value and it's index
        word_type = max(word_one_hot)
        type_index = list(word_one_hot).index(word_type)  # classify_word_one_hot = [price, location, name]
        if type_index == 0:
            restaurant_price.append(tagged_word[word_type_index])
        elif type_index == 1:
            restaurant_location.append(tagged_word[word_type_index])
        elif type_index == 2:
            restaurant_name.append(tagged_word[word_type_index])
        else:
            food_type.append(tagged_word[word_type_index])
        word_type_index += 1  # move the tagged index

    print("Restaurant Price: ", restaurant_price)
    print("Restaurant Location: ", restaurant_location)
    print("Restaurant Name: ", restaurant_name)
    print("Food Type: ", food_type)

