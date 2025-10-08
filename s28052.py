import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

arr1 = []
arr2 = []
def generate_arr(start_range, stop_range):
    number_arr = random.randrange(50, 100)
    xpoints = list()
    ypoints = list()
    for num in range(number_arr):
        xpoints.append(random.randrange(start_range, stop_range))
        ypoints.append(random.randrange(start_range, stop_range))
    arr = [xpoints, ypoints]
    return arr


def generate_data():
    arr1 = generate_arr(1, 50)
    arr2 = generate_arr(100, 300)
    return arr1, arr2


def train_model():
    arr1, arr2 = generate_data()

    X1 = np.column_stack((arr1[0], arr1[1]))
    X2 = np.column_stack((arr2[0], arr2[1]))

    X = np.vstack((X1, X2))

    y1 = np.zeros(len(X1))
    y2 = np.ones(len(X2))

    y = np.concatenate((y1, y2))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    train_model()
