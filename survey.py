from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time 
model=RandomForestClassifier(n_estimators=300,
                             random_state=42,
                             min_samples_split=4,
                             max_features="sqrt",
                             max_depth=10)
data = pd.read_csv('dataset.csv',delimiter=",")
X=data[["Are the school meals sufficient?",
        "Is the school's location good?",
        "Is the school's social environment good?",
        "Is the school's management good?",
        "Are the teachers good?",
        "Is the school's fees good?"]]
y=data["Decision"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
condfusion_matrix_res=confusion_matrix(y_test,predictions)
classification_report_res=classification_report(y_test,predictions)
from sklearn.metrics import accuracy_score
accuracy_score_res=accuracy_score(y_test, predictions)
accuracy_score_res=accuracy_score_res*100

print(f"confusion matrix score:{condfusion_matrix_res}")
print(f"classification report score:{classification_report_res}")
print(f"accuracy score:%{accuracy_score_res:.2f}")
time.sleep(2)
while True:
    ask=input("do you want to join this survey or exit?(yes/no)").lower()
    if ask=="yes":
        def encode_response(response):
            return 1 if response == "yes" else 0

        new_data = [[
            encode_response(input("Are the school meals sufficient? (yes/no): ").lower()),
            encode_response(input("Is the school's location good? (yes/no): ").lower()),
            encode_response(input("Is the school's social environment good? (yes/no): ").lower()),
            encode_response(input("Is the school's management good? (yes/no): ").lower()),
            encode_response(input("Are the teachers good? (yes/no): ").lower()),
            encode_response(input("Is the school's fees good? (yes/no): ").lower())
        ]]

        new_data_df = pd.DataFrame(new_data, columns=[
            "Are the school meals sufficient?",
            "Is the school's location good?",
            "Is the school's social environment good?",
            "Is the school's management good?",
            "Are the teachers good?",
            "Is the school's fees good?"
        ])
        prediction_proba=model.predict_proba(new_data_df)
        prediction=model.predict(new_data_df)
        if prediction ==1:
            print(f"you will stay in the school and its proballity:{prediction_proba[0][1]:.2f}")
        else:
            print(f"you will leave the school and its proballity:{prediction_proba[0][0]:.2f}")
    elif ask=="no":
        break
