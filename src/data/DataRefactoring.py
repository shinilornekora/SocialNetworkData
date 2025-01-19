import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


def get_data():
    data = pd.read_csv('social_network_users.csv')

    data = data[['PostsCount', 'LikesPerPost', 'MessagesSent', 'PhotosPerDay', 'Topic']]

    topic_mapping = {
        "political": 0,
        "music": 1,
        "technologies": 2,
        "sport": 3,
        "traveling": 4,
        "food": 5
    }
    data['Topic'] = data['Topic'].map(topic_mapping)
    message_sent_mean = data['MessagesSent'].mean()
    photos_per_day_mean = data['PhotosPerDay'].mean()

    data['classes'] = data.apply(
        lambda row:
        (row['MessagesSent'] / message_sent_mean) *
        (row['PhotosPerDay'] / photos_per_day_mean) *
        (row['Topic'] / row['PostsCount']) *
        row['LikesPerPost'],
        axis=1
    )

    # classes_max = data['classes'].max()
    # data['classes'] = data['classes'].apply(lambda x: x/classes_max)

    # data = data.iloc[:300]
    data.to_csv('data_with_class.csv', index=False)

    print(data.head())

# get_data()


def get_classes_scaled():
    data = pd.read_csv('cleaned_data.csv')

    classes_max = data['classes'].max()
    data['classes'] = data['classes'].apply(lambda x: x/classes_max)

    data.to_csv('data_with_class_scaled.csv', index=False)

# get_classes_scaled()

def get_int_class():
    data = pd.read_csv('cleaned_data.csv')

    classes_mean = data['classes'].mean()
    print(classes_mean)
    classes_max = data['classes'].max()
    data['classes'] = data['classes'].apply(lambda x: 0 if x < classes_mean else 1)

    data.to_csv('data_with_class_int.csv', index=False)

get_int_class()


def analyze_data():
    data = pd.read_csv('cleaned_data.csv')
    classes_max = data['classes'].max()
    classes_min = data['classes'].min()
    print(classes_max)
    print(classes_min)

# analyze_data()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def z_score():
    data = pd.read_csv("data.csv")
    numerical_columns = ['PostsCount', 'LikesPerPost', 'MessagesSent', 'PhotosPerDay']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    data.to_csv("scaled_data.csv", index=False)
    print("Данные успешно масштабированы!")


def drop_emission():
    data = pd.read_csv("data_with_class.csv")

    data = drop_by_name(data, 'PostsCount')
    data = drop_by_name(data, 'LikesPerPost')
    data = drop_by_name(data, 'MessagesSent')
    data = drop_by_name(data, 'PhotosPerDay')

    # Сохраняем результат
    data.to_csv("cleaned_data.csv", index=False)
    print("Выбросы удалены!")
def drop_by_name(data, name):
    # Удаляем строки, где значения в столбце 'ColumnName' являются выбросами (например, за пределами 3 стандартных отклонений)
    mean = data[name].mean()
    std_dev = data[name].std()
    data = data[(data[name] >= mean - 1.5 * std_dev) & (data[name] <= mean + 1.5 * std_dev)]
    return data


def high_to_median():
    data = pd.read_csv("cleaned_data.csv")

    data = median_by_name(data, 'PostsCount')
    data = median_by_name(data, 'LikesPerPost')
    data = median_by_name(data, 'MessagesSent')
    data = median_by_name(data, 'PhotosPerDay')

    data.to_csv("median_replaced_data.csv", index=False)
    print("Крайние значения заменены на медиану!")


def median_by_name(data, name):
    count = 0
    median_value = data[name].median()
    print(median_value)

    def replace_and_count(x):
        nonlocal count
        if x > 0.95 or x < 0.1:
            count += 1
            return median_value
        else:
            return x

    data[name] = data[name].apply(replace_and_count)
    print(count)
    return data


def balance_data():
    # Загружаем данные
    data = pd.read_csv("cleaned_data.csv")
    # Разделяем на категории (например, target = 0 и target = 1)
    majority = data[data['target'] == 0]
    minority = data[data['target'] == 1]
    # Увеличиваем количество записей в меньшинстве
    minority_upsampled = resample(minority,
                                  replace=True,  # С выбором с заменой
                                  n_samples=len(majority),  # Уравниваем количество записей
                                  random_state=42)
    # Объединяем обратно
    balanced_data = pd.concat([majority, minority_upsampled])
    # Сохраняем результат
    balanced_data.to_csv("balanced_data.csv", index=False)
    print("Данные сбалансированы!")


# z_score()
# drop_emission()
# high_to_median()
# balance_data()
