import pandas as pd

lc = pd.read_csv("../../Downloads/fyp_sem2 (1)/fyp_sem2/csv_file/Licensed_hotels.csv")


def get_hotel(col, req):
    lc['Premises Phone No.'] = lc['Premises Phone No.'].apply(lambda x: str(x))
    hotels = []
    counter = 0
    if col == "Price":
        for pricing in lc["Price"]:
            if req in pricing:
                hotels.append(lc['Premises Name'][counter])
            counter += 1
    elif col == "Licence No.":
        for licence_num in lc["Licence No."]:
            if req in licence_num:
                hotels.append(lc['Premises Name'][counter])
            counter += 1
    elif col == "Premises Name":
        for hotel_name in lc["Premises Name"]:
            if req in hotel_name:
                hotels.append(hotel_name)
    elif col == "Premises Address":
        for address in lc["Premises Address"]:
            if req in address:
                hotels.append(lc['Premises Name'][counter])
            counter += 1
    elif col == "Premises Phone No.":
        for tel in lc['Premises Phone No.']:
            if req in tel:
                hotels.append(lc['Premises Name'][counter])
            counter += 1

    if len(hotels) > 0:
        return hotels
    else:
        return []
