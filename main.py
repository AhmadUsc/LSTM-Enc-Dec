# LSTM for regression
import gc
from datetime import datetime
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler





# convert Dataframe to list without bracket
def without_bracket(list_XY):
    new_list = []
    for item in (list_XY):
        new_list.append(item[0])
    return new_list


# parse date time
def parser(x):
    return datetime.strptime(x, '%m/%d/%y %H:%M')

    # convert an array of values into a dataset matrix


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


epoch_list = []
batch_list = []
forecast_days = []


# loading dataset
datasetName = 'data/sample.csv'
l = len(datasetName)
datasetName_title = datasetName[:l - 4]

dataframe = read_csv('./' + datasetName, usecols=[1], engine='python')

global dataset_gl
dataset_gl = dataframe.values
dataset_gl = dataset_gl.astype('float32')



dataset_actual = dataset_gl

epoch_ = 50
batch_ = 16
look_back = 12
numberOfDays_forecast = 7



# timestamp to be forecasted
totaltimestamp = 156

# get list of time
listOfTime = []
with open('./' + datasetName) as f:
    for row in f:
        listOfTime.append(row.split(',')[0])
del listOfTime[0]



# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset_gl)

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:train_size + totaltimestamp, :]


# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit our data to the LSTM network
model = Sequential()
model.add(LSTM(128, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=["accuracy"])
model.fit(trainX, trainY, epochs=epoch_, batch_size=batch_, verbose=0)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

path_plot = './plots/'
path_result = './results/'

# getting actual values part
actual_start = train_size + look_back + 2
actual_end = train_size + totaltimestamp - look_back

# get name of dataset
edited_path = datasetName_title.split('/')
dataset_name = edited_path[1]

# get time of each run in minutes
# time_sec = "{:.2f}".format(((time.time() - start_time) / 60))

# getting the timestamps
timestamp = listOfTime[train_size + look_back:train_size + totaltimestamp - 1]

# save forecast output
df_test = pd.DataFrame(testPredict[:])
lst_test = df_test.values.tolist()
# print('lst_test', len(lst_test))

df_actual = pd.DataFrame(dataset_actual[train_size + look_back:train_size + totaltimestamp - 1])
lst_actual = df_actual.values.tolist()

new_actual = without_bracket(lst_actual)
new_test = without_bracket(lst_test)

df_forecast = pd.DataFrame({'time': timestamp, 'actual': new_actual, 'predicted': new_test})
df_forecast.to_csv(path_result + dataset_name + "-Epochs" + str(epoch_) + "-Batch" + str(batch_) + "-days" + str(
    numberOfDays_forecast) + '_result_forecast.csv', index=False)


# plotting the points
plt.scatter(new_actual, new_test, label="stars", color="blue",
            marker="*", s=30)

title = dataset_name + '-Epochs:' + str(epoch_) + '-Batch:' + str(batch_) + '-days:' + str(numberOfDays_forecast)

# naming the x axis
plt.xlabel('x - actual')
# naming the y axis
plt.ylabel('y - predicted')
# naming the title
plt.title(title)
# save forecast plot
plt.savefig(path_plot + dataset_name + '-Epochs' + str(epoch_) + 'Batch' + str(batch_) + 'days' + str(
    numberOfDays_forecast) + '.png')

plt.clf()
plt.close('all')


gc.collect()

