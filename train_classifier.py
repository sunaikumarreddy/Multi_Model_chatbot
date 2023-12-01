import pickle


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#original----------------------------
data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
#---------------------------------


with open('data.pickle', 'rb') as f:
    loaded_data = pickle.load(f)

# max_length = max(len(item) for item in loaded_data['data'])
# data = np.array([item + [0] * (max_length - len(item)) for item in loaded_data['data']])

data = np.array([item[:42] for item in loaded_data['data']])

# data = np.asarray(loaded_data['data'])
# data = np.ravel(loaded_data['labels'])
# data = data.reshape(-1, 1)

# print(type(loaded_data['data']))
# print(type(loaded_data['data'][0]))

# labels = np.asarray(loaded_data['labels'])



x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

print(len(x_train[0]))

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

