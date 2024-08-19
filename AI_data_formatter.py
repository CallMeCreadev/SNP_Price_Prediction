from temporary_data_storage import TemporaryDataStorage


class DataFormatter:
    normalization_params = {}

    @staticmethod
    def convert_prediction_into_prices(prediction, days_ago):
        realdata = TemporaryDataStorage("market_data.pkl")
        all_stock_data = realdata.load_data()
        target_features = all_stock_data['closing_prices']
        current_stock_data = target_features['SPY']
        ma_20d_l = DataFormatter.calculate_20_day_moving_average(current_stock_data)

        # Get the last 7-day moving average value
        ma_20d = ma_20d_l[(-days_ago)]

        # Initialize the result list
        result = []

        # Iterate through each sublist in the prediction
        for sublist in prediction:
            # Check if the current item is a list to handle nested lists
            if isinstance(sublist, list):
                # Multiply each item in the sublist by the last ma_7d value
                result.append([item * ma_20d for item in sublist])
            else:
                # If the item is not a list, just multiply the item itself
                result.append(sublist * ma_20d)

        return result


    @staticmethod
    def calculate_7_day_moving_average(prices):
        moving_averages = []
        if len(prices) < 200:
            return moving_averages  # Return an empty list if there aren't enough values
        for i in range(6, len(prices)):
            window = prices[i - 6:i + 1]  # Select the window of 20 days
            window_average = sum(window) / 7  # Calculate the average for the window
            moving_averages.append(window_average)

        return moving_averages[193:]

    @staticmethod
    def calculate_20_day_moving_average(prices):
        moving_averages = []
        if len(prices) < 200:
            return moving_averages  # Return an empty list if there aren't enough values
        for i in range(19, len(prices)):
            window = prices[i - 19:i + 1]  # Select the window of 20 days
            window_average = sum(window) / 20  # Calculate the average for the window
            moving_averages.append(window_average)

        return moving_averages[180:]

    @staticmethod
    def calculate_50_day_moving_average(prices):
        moving_averages = []
        if len(prices) < 200:
            return moving_averages  # Return an empty list if there aren't enough values
        for i in range(49, len(prices)):
            window = prices[i - 49:i + 1]  # Select the window of 20 days
            window_average = sum(window) / 50  # Calculate the average for the window
            moving_averages.append(window_average)

        return moving_averages[150:]

    @staticmethod
    def calculate_100_day_moving_average(prices):
        moving_averages = []
        if len(prices) < 200:
            return moving_averages  # Return an empty list if there aren't enough values
        for i in range(99, len(prices)):
            window = prices[i - 99:i + 1]  # Select the window of 20 days
            window_average = sum(window) / 100  # Calculate the average for the window
            moving_averages.append(window_average)

        return moving_averages[100:]

    @staticmethod
    def calculate_200_day_moving_average(prices):
        moving_averages = []
        if len(prices) < 200:
            return moving_averages  # Return an empty list if there aren't enough values
        for i in range(199, len(prices)):
            window = prices[i - 199:i + 1]  # Select the window of 20 days
            window_average = sum(window) / 200  # Calculate the average for the window
            moving_averages.append(window_average)

        return moving_averages

    @staticmethod
    def prune(lst):
        list_from_200 = lst[199:] if len(lst) > 199 else []

        return list_from_200


    @staticmethod
    def divide_lists(list_a, list_b):
        if len(list_a) != len(list_b):
            raise ValueError("The lists must have the same length.")

        # Avoid division by zero by checking if any element in list_b is 0
        if any(b == 0 for b in list_b):
            raise ZeroDivisionError("Cannot divide by zero in list_b.")

        return [a / b for a, b in zip(list_a, list_b)]

    @staticmethod
    def transform_features(features):
        return [(x - 1) * 10 for x in features]

    @staticmethod
    def normalize_data(data, feature_name, feature, variable_index):
        mean = sum(data) / len(data)
        std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
        key = f"{feature_name}_{feature}_{variable_index}"
        DataFormatter.normalization_params[key] = {'mean': mean, 'std': std}
        normalized_data = [(x - mean) / std if std else 0 for x in data]
        return normalized_data

    @staticmethod
    def denormalize_data(normalized_data, feature, feature_name):
        denormalized_data = []
        for index, value in enumerate(normalized_data):
            key = f"{feature_name}_{feature}_{index}"
            params = DataFormatter.normalization_params.get(key, None)
            if params is None:
                raise ValueError(f"Normalization parameters for {key} not found.")
            mean, std = params['mean'], params['std']
            denormalized_value = value * std + mean
            denormalized_data.append(denormalized_value)
        return denormalized_data

    @staticmethod
    def inverse_transform_features(transformed_features):
        return [(y / 10) + 1 for y in transformed_features]


    @staticmethod
    def build_feature_tuple(list1, list2, list3, list4, list5):
        # Check if all lists have the same length
        if not all(len(lst) == len(list1) for lst in [list2, list3, list4, list5]):
            raise ValueError("All lists must have the same length.")

        # Combine elements from all six lists into tuples
        return list(zip(list1, list2, list3, list4, list5))

    @staticmethod
    def build_target_tuple(list1, list2, list3, list4, list5):
        # Check if all lists have the same length
        if not all(len(lst) == len(list1) for lst in [list2, list3, list4, list5]):
            raise ValueError("All lists must have the same length.")

        # Combine elements from all five lists into tuples
        return list(zip(list1, list2, list3, list4, list5))


    @staticmethod
    def construct_moving_avg_feature_tuple(data, feature_name, callback, sequence_size, feature):
        data_7d = DataFormatter.calculate_7_day_moving_average(data)
        data_20d = DataFormatter.calculate_20_day_moving_average(data)
        data_50d = DataFormatter.calculate_50_day_moving_average(data)
        data_100d = DataFormatter.calculate_100_day_moving_average(data)
        data_200d = DataFormatter.calculate_200_day_moving_average(data)

        data_pruned = DataFormatter.prune(data)
        coefficients_7d = DataFormatter.divide_lists(data_pruned, data_7d)
        coefficients_20d = DataFormatter.divide_lists(data_pruned, data_20d)
        coefficients_50d = DataFormatter.divide_lists(data_pruned, data_50d)
        coefficients_100d = DataFormatter.divide_lists(data_pruned, data_100d)
        coefficients_200d = DataFormatter.divide_lists(data_pruned, data_200d)

        feature_7d = DataFormatter.normalize_data(coefficients_7d, feature_name, feature, 0)
        feature_20d = DataFormatter.normalize_data(coefficients_20d, feature_name, feature, 1)
        feature_50d = DataFormatter.normalize_data(coefficients_50d, feature_name, feature, 2)
        feature_100d = DataFormatter.normalize_data(coefficients_100d, feature_name, feature, 3)
        feature_200d = DataFormatter.normalize_data(coefficients_200d, feature_name, feature, 4)

        prebuild = DataFormatter.build_feature_tuple(feature_7d, feature_20d, feature_50d, feature_100d,
                                                     feature_200d)
        if callback == -1:
            trimmed = prebuild[:-20]
            return trimmed
        else:
            recent = prebuild[(-sequence_size-callback-1):(-callback-1)]
            return recent


    @staticmethod
    def construct_target_tuple(tgt_data, target_name, feature):
        pf7, pf20, pf50, pf100, pf200 = DataFormatter.prep_target_future_prices(tgt_data)
        ma_7d, ma_20d, ma_50d, ma_100d, ma_200d = DataFormatter.prep_target_current_averages(tgt_data)

        target1_coef = DataFormatter.divide_lists(pf7, ma_7d)
        target1 = DataFormatter.normalize_data(target1_coef, target_name, feature, 0)

        target2_coef = DataFormatter.divide_lists(pf20, ma_20d)
        target2 = DataFormatter.normalize_data(target2_coef, target_name, feature, 1)

        target3_coef = DataFormatter.divide_lists(pf50, ma_50d)
        target3 = DataFormatter.normalize_data(target3_coef, target_name, feature, 2)

        target4_coef = DataFormatter.divide_lists(pf100, ma_100d)
        target4 = DataFormatter.normalize_data(target4_coef, target_name, feature, 3)

        target5_coef = DataFormatter.divide_lists(pf200, ma_200d)
        target5 = DataFormatter.normalize_data(target5_coef, target_name, feature, 4)

        return DataFormatter.build_target_tuple(target1, target2, target3, target4, target5)



    @staticmethod
    def prep_target_future_prices(original_list):
        #get future moving averages 20 days out
        data_7d = DataFormatter.calculate_7_day_moving_average(original_list)
        data_20d = DataFormatter.calculate_20_day_moving_average(original_list)
        data_50d = DataFormatter.calculate_50_day_moving_average(original_list)
        data_100d = DataFormatter.calculate_100_day_moving_average(original_list)
        data_200d = DataFormatter.calculate_200_day_moving_average(original_list)


        future_7d_prices = data_7d[20:]
        future_20d_prices = data_20d[20:]
        future_50d_prices = data_50d[20:]
        future_100d_prices = data_100d[20:]
        future_200d_prices = data_200d[20:]

        return future_7d_prices, future_20d_prices, future_50d_prices, future_100d_prices, future_200d_prices


    @staticmethod
    def prep_target_current_averages(original_list):
        data_7d = DataFormatter.calculate_7_day_moving_average(original_list)
        data_20d = DataFormatter.calculate_20_day_moving_average(original_list)
        data_50d = DataFormatter.calculate_50_day_moving_average(original_list)
        data_100d = DataFormatter.calculate_100_day_moving_average(original_list)
        data_200d = DataFormatter.calculate_200_day_moving_average(original_list)
        target_ma_7d = data_7d[:-20]
        target_ma_20d = data_20d[:-20]
        target_ma_50d = data_50d[:-20]
        target_ma_100d = data_100d[:-20]
        target_ma_200d = data_200d[:-20]
        return target_ma_7d, target_ma_20d, target_ma_50d, target_ma_100d, target_ma_200d

    @staticmethod
    def construct_complete_feature_dictionary(callback, sequence_size):
        tempstorage = TemporaryDataStorage("market_data.pkl")
        market_data = tempstorage.load_data()

        stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'HYG', 'TLT', 'LQD', 'XLK', 'XLY', 'XLI', 'XLF', 'XLE',
           'XLB', 'IYR', 'XLV', 'XLP', 'XLU', '^RUT', '^IXIC', '^DJI',  'SHY',
           'IEF', 'IEI', 'SPY', 'NVDA', 'SMH', 'AMZN', 'XHB', 'GLD', '^VIX']

        #feature_types = ['closing_prices', 'opening_prices', 'high_prices', 'low_prices', 'volumes']
        feature_types = ['closing_prices', 'volumes']

        # Dictionary to hold the feature tuples
        feature_tuples = {}

        for feature in feature_types:
            feature_data_set = market_data[feature]
            for symbol in stock_symbols:
                feature_data = feature_data_set[symbol]
                #if symbol != 'DX-Y.NYB' and symbol != 'HG=F':
                    #feature_data = feature_data[2:]
                #if symbol == 'DX-Y.NYB':
                    #feature_data = feature_data[3:]
                #if symbol == 'HG=F':
                    #feature_data = feature_data[1:]
                if not (feature == 'volumes' and (symbol == 'DX-Y.NYB' or symbol == '^VIX')):
                    feature_tuples[(symbol, feature)] = DataFormatter.construct_moving_avg_feature_tuple(feature_data, symbol, callback, sequence_size, feature)
        return feature_tuples

    @staticmethod
    def construct_complete_targets():
        tempstorage2 = TemporaryDataStorage("market_data.pkl")
        stock_data = tempstorage2.load_data()
        spy_stock_data = stock_data['closing_prices']
        SPY_data = spy_stock_data['SPY']
        target_tuple_spy = DataFormatter.construct_target_tuple(SPY_data, 'TGT_SPY', 'closing_prices')
        return target_tuple_spy


    @staticmethod
    def are_numbers_within_1_percent(num1, num2):
        """
        Check if num1 and num2 are within 1% of each other.

        :param num1: First number
        :param num2: Second number
        :return: Boolean indicating whether the numbers are within 1% of each other
        """
        if num1 == 0 and num2 == 0:
            return True  # If both numbers are 0, they are considered to be within 1% of each other

        larger = max(abs(num1), abs(num2))
        difference = abs(num1 - num2)

        # Check if the percentage difference is less than or equal to 1%
        if (difference / larger) <= 0.02:
            return 1
        else:
            return 0


    @staticmethod
    def validate_predictions(prediction, days_ago):
        realdata = TemporaryDataStorage("market_data.pkl")
        all_stock_data = realdata.load_data()
        target_features = all_stock_data['closing_prices']
        current_stock_data = target_features['SPY']
        ma_20d_l = DataFormatter.calculate_20_day_moving_average(current_stock_data)

        # Get the last 7-day moving average value
        real_ma_20d = ma_20d_l[(-days_ago)]
        print("Real:", real_ma_20d)
        prediction_ma_20d = ma_20d_l[(-days_ago)-20]
        print("Predicted:", prediction_ma_20d)

        # Initialize the result list
        result = []

        # Iterate through each sublist in the prediction
        for sublist in prediction:
            # Check if the current item is a list to handle nested lists
            if isinstance(sublist, list):
                # Multiply each item in the sublist by the last ma_7d value
                result.append([item * prediction_ma_20d for item in sublist])
            else:
                # If the item is not a list, just multiply the item itself
                result.append(sublist * prediction_ma_20d)

        return DataFormatter.are_numbers_within_1_percent(prediction_ma_20d, real_ma_20d)


if __name__ == "__main__":

    tempstorage3 = TemporaryDataStorage("market_data.pkl")
    market_data = tempstorage3.load_data()
    closing_prices = market_data['closing_prices']
    opening_prices = market_data['opening_prices']
    high_prices = market_data['high_prices']
    low_prices = market_data['low_prices']
    volumes = market_data['volumes']

    AAPL_closings = closing_prices['AAPL']
    SPY_closings = closing_prices['SPY']
    Copper_closings = closing_prices['HG=F']
    dollar_closings = closing_prices['DX-Y.NYB']
    print('AAPL_closings', len(AAPL_closings))
    print('SPY_closings', len(SPY_closings))
    print('Copper_closings', len(Copper_closings))
    print('dollar_closings', len(dollar_closings))






