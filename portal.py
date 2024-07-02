import pandas as pd

# Correct the file path
url = r"C:\Users\venu2\Downloads\personalized-learning-system-main\personalized-learning-system-main\dataset\learning_style_dataset.csv"

# Read the CSV file
try:
    df = pd.read_csv(url)
    print(df.head())
except Exception as e:
    print(f"Failed to read the dataset: {e}")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

x = df[['quiz_score', 'time_spent']]
y = df['learning_style']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0, train_size=0.75)

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(cm)

ac = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {ac}")

precision = precision_score(Y_test, Y_pred, average="micro")
print(f"Precision: {precision}")

recall = recall_score(Y_test, Y_pred, average="micro")
print(f"Recall: {recall}")

f1 = f1_score(Y_test, Y_pred, average="micro")
print(f"F1 Score: {f1}")

cv_scores = cross_val_score(clf, x, y, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")



time_spent = int(input("Time spent by the User: "))
quiz_scores = int(input("Scores obtained by the User: "))

input_data = pd.DataFrame([[quiz_scores, time_spent]], columns=["quiz_score", "time_spent"])

result = clf.predict(input_data)

print("Prediction result:", result[0])


import pandas as pd

# Correct the file path
file_path = r"C:\Users\venu2\Downloads\personalized-learning-system-main\personalized-learning-system-main\dataset\content_database.content_collection_csv.csv"

# Read the CSV file
try:
    df = pd.read_csv(file_path)
    print(df.head())
except Exception as e:
    print(f"Failed to read the dataset: {e}")





learning_style_topics = {
    'visual': ['Percentages', 'Boats and streams'],
    'auditory': ['Age problems', 'Time, distance, speed'],
    'kinesthetic': ['Number series', 'Pipes and cisterns']
}

if isinstance(result, (list, np.ndarray)):
    result = result[0]

suggested_topics = learning_style_topics[result]
print(f"Suggested topics for {result} learners are: {suggested_topics[0]} & {suggested_topics[1]}")

chosen_topic = input(f"Which topic do you prefer? {suggested_topics[0]} or {suggested_topics[1]}: ")

if chosen_topic not in suggested_topics:
    print("Invalid topic chosen.")
else:
    resource_preference = input("Do you prefer YouTube or article?: ").lower()

    topic_row = df[df['topic'] == chosen_topic].iloc[0]
    if resource_preference == 'youtube':
        print(f"Here is the YouTube link for {chosen_topic}: {topic_row['Youtube']}")
    elif resource_preference == 'article':
        print(f"Here is the article link for {chosen_topic}: {topic_row['Article']}")
    else:
        print("Invalid preference chosen.")
