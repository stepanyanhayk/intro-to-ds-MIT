"""
@author: Hayk Stepanyan

Created on July 20, 2020
"""

import pylab
import re
import numpy

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""


class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """

    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]


def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y) ** 2).sum()
    var_x = ((x - x.mean()) ** 2).sum()
    SE = pylab.sqrt(EE / (len(x) - 2) / var_x)
    return SE / model[0]


"""
End helper code
"""


def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    coefficients = []
    for degree in degs:
        model = pylab.polyfit(x, y, degree)
        coefficients.append(pylab.array(model))
    return coefficients


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    estimates_error = ((y - estimated)**2).sum()
    mean = y.sum() / len(y)
    variability = ((y - mean)**2).sum()

    return float(1 - estimates_error / variability)


def evaluate_models_on_training(x, y, models):
    """
    For each regression         temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    plot_count = 0
    for model in models:
        plot_count += 1
        pylab.figure(plot_count)
        degree = len(model) - 1
        pylab.plot(x, y, 'bo', data='Data')
        pylab.xlabel("Years")
        pylab.ylabel("Temperature")

        estYVals = pylab.polyval(model, x)
        error = r_squared(y, estYVals)
        pylab.plot(x, estYVals, 'r-')

        if degree == 1:
            pylab.title(label='Degree=' + str(degree) + '    R2=' + str(round(error, 5)) + '    SE/Slope=' + str(
                se_over_slope(x, y, estYVals, model)))
        else:
            pylab.title(label='Degree=' + str(degree) + '    R2=' + str(round(error, 5)))


def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    temperatures = []
    for year in years:
        city_avg_sum = 0
        for city in multi_cities:
            year_temp = climate.get_yearly_temp(city, year)
            city_avg_sum += year_temp.sum() / len(year_temp)
        temperatures.append(city_avg_sum / len(multi_cities))
    return pylab.array(temperatures)


def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    moving_averages = []
    y = pylab.array(y)
    for i in range(len(y)):
        if i + 1 < window_length:
            mean = y[range(0, i + 1)].sum() / len(range(0, i + 1))
            moving_averages.append(mean)
        else:
            mean = y[range(i + 1 - window_length, i + 1)].sum() / len(range(i + 1 - window_length, i + 1))
            moving_averages.append(mean)

    return moving_averages


def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    estimated_error = ((y - estimated)**2).sum()
    return (estimated_error / len(y))**0.5


def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    st_devs = []
    for year in years:
        if year % 4 == 0:
            daily_temperatures = pylab.zeros(366)
        else:
            daily_temperatures = pylab.zeros(365)
        for city in multi_cities:
            daily_temperatures += climate.get_yearly_temp(city, year)

        mean_temperatures = daily_temperatures / len(multi_cities)
        st_devs.append(numpy.std(mean_temperatures))
    return st_devs


def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    plot_count = 0
    for model in models:
        plot_count += 1
        pylab.figure(plot_count)
        degree = len(model) - 1
        pylab.plot(x, y, 'bo', data='Data')
        pylab.xlabel("Years")
        pylab.ylabel("Temperature")

        estYVals = pylab.polyval(model, x)
        error = rmse(y, estYVals)
        pylab.plot(x, estYVals, 'r-')

        pylab.title(label='Degree=' + str(degree) + '    RMSE=' + str(round(error, 5)))


if __name__ == '__main__':
    # Part A.4
    # # 4.I
    # training_sample = Climate("data.csv")
    # temperatures = []
    # for year in TRAINING_INTERVAL:
    #     temperatures.append(training_sample.get_daily_temp("NEW YORK", 1, 10, year))
    # temperatures = pylab.array(temperatures)
    # models = generate_models(pylab.array(TRAINING_INTERVAL), temperatures, [1])
    # evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), temperatures, models)
    # pylab.show()

    # # 4.II
    # training_sample = Climate("data.csv")
    # temperatures = []
    # for year in TRAINING_INTERVAL:
    #     temperatures.append(training_sample.get_yearly_temp("NEW YORK", year))
    # temperatures = pylab.array(temperatures)
    #
    # mean_temperatures = []
    # for year_temp in temperatures:
    #     mean_temperatures.append(year_temp.sum() / len(year_temp))
    # mean_temperatures = pylab.array(mean_temperatures)
    # models = generate_models(pylab.array(TRAINING_INTERVAL), mean_temperatures, [1])
    # evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), mean_temperatures, models)
    # pylab.show()

    # # Part B (National Yearly Temperatures)
    # training_sample = Climate("data.csv")
    # temperatures = pylab.array(gen_cities_avg(training_sample, CITIES, TRAINING_INTERVAL))
    # models = generate_models(pylab.array(TRAINING_INTERVAL),temperatures, [1])
    # evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), temperatures, models)
    # pylab.show()

    # # Part C (Moving Average)
    # training_sample = Climate("data.csv")
    # yVals = pylab.array(gen_cities_avg(training_sample, CITIES, TRAINING_INTERVAL))
    # temperatures = pylab.array(moving_average(yVals, 5))
    # models = generate_models(pylab.array(TRAINING_INTERVAL), temperatures, [1])
    # evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), temperatures, models)
    # pylab.show()

    # # Part D.2 (Predicting)
    # Generating new models
    # training_sample = Climate("data.csv")
    # yVals = pylab.array(gen_cities_avg(training_sample, CITIES, TRAINING_INTERVAL))
    # temperatures = pylab.array(moving_average(yVals, 5))
    # models = generate_models(pylab.array(TRAINING_INTERVAL), temperatures, [1, 2, 20])
    # # evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), temperatures, models)
    # # pylab.show()
    #
    # # 2.II
    # test_data = pylab.array(gen_cities_avg(training_sample, CITIES, TESTING_INTERVAL))
    # test_data_temp = pylab.array(moving_average(test_data, 5))
    # evaluate_models_on_testing(TESTING_INTERVAL, test_data_temp, models)
    # pylab.show()


    # # Part E
    # training_sample = Climate("data.csv")
    # standard_deviations = gen_std_devs(training_sample, CITIES, TRAINING_INTERVAL)
    # moving_averages = pylab.array(moving_average(standard_deviations, 5))
    # models = generate_models(pylab.array(TRAINING_INTERVAL), moving_averages, [1])
    # evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), moving_averages, models)
    # pylab.show()
