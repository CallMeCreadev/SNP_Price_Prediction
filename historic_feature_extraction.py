from datetime import datetime, timedelta
from WebScraper_Home_Page.ratios import Ratios
from WebScraper_Home_Page.sectors import Sectors
from WebScraper_Home_Page.bonds import Bonds
from temporary_data_storage import TemporaryDataStorage
from utils.utils import Utils


def ratios_ft_ext():
    # Your extraction logic here
    ratios_instance = Ratios()
    data_list = []
    #Gather data for past 7 years
    for day in range(2555):
        print(f"Data pulled for day {day}")
        shifted_date = datetime.now() - timedelta(days=day)
        data = ratios_instance.get_spy_ratios(shifted_date)
        data_list.append(data)
    return data_list

def bonds_ft_ext():
    # Your extraction logic here
    bonds_instance = Bonds()
    data_list = []
    # Gather data for past 7 years
    for day in range(2555):
        print(f"Data pulled for day {day}")
        shifted_date = datetime.now() - timedelta(days=day)
        raw_data = Utils.process_bond_sector_data(bonds_instance.get_bonds_dictionary(shifted_date))
        clean = Utils.replace_second_num_lod(raw_data)
        data_list.append(clean)

    return data_list
def sectors_ft_ext():
    # Your extraction logic here
    sectors_instance = Sectors()
    data_list = []
    # Gather data for past 7 years
    for day in range(2555):
        print(f"Data pulled for day {day}")
        shifted_date = datetime.now() - timedelta(days=day)
        raw_data = Utils.process_bond_sector_data(sectors_instance.get_sectors_dictionary(shifted_date))
        clean = Utils.replace_second_num_lod(raw_data)
        data_list.append(clean)

    return data_list

# You can add more extraction functions as needed

def perform_extraction(extraction):
    # Map extraction numbers to functions
    extraction_functions = {
        'ratios_ext': ratios_ft_ext,
        'bonds_ext': bonds_ft_ext,
        'sectors_ext': sectors_ft_ext,
        # Add more mappings as necessary
    }

    # Check if the extraction number is valid
    if extraction in extraction_functions:
        # Get the corresponding function and call it
        data = extraction_functions[extraction]()
        filename = f"data_{extraction}.pkl"
        storage = TemporaryDataStorage(filename)
        storage.save_data(data)
        print(f"Saved data from extraction {extraction} to {filename}")
    else:
        print(f"No extraction function defined for {extraction}.")

# Directly specify the extraction number here
if __name__ == "__main__":
    extraction = 'ratios_ext'
    perform_extraction(extraction)







