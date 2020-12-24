import extractingData
import preProcessData
import Nn_model

DURATION = 26# in seconds

# Извлечение данных(если данные готовы, следующую строку закомментировать)
#extractingData.extraction("data/Drunk/", "Drunk3", Create=1, Segments=13, duration=DURATION)
#extractingData.extraction("data/Sober/", "Sober3", Create=1, Segments=13, duration=DURATION)

# Пре-процессинг данных
dataset, label = preProcessData.preProcessData("img_data/Drunk3/", dataClass=0)
dataset, label = preProcessData.preProcessData("img_data/Sober3/", dataset, label, dataClass=1)
X_train, X_test, y_train, y_test = preProcessData.splitData(dataset, label)

# Создание нейронной модели
model = Nn_model.model_creation()

# Обучение и оценка модели
Nn_model.model_eval(model, X_train, X_test, y_train, y_test)