from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

df = pd.read_csv("./static/Data/diseases_data.csv")

label_encoder = LabelEncoder()
df["Category"] = label_encoder.fit_transform(df["Category"])

target = []
for item in df["Disease"]:
    item = item.lower()
    target.append(item)

df["Target_disease"] = target[:]

@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_name = request.form["user_name"]
        given_disease = request.form["disease"]

        if given_disease is None:
            return redirect(url_for("home"))
        
        else:
            disease = given_disease.lower()

            value = int(df[df['Target_disease']== disease].index[0])
            res_1 = df.at[value, 'Tenure']
            res_2 = df.at[value, 'Category']

            data=[res_1, res_2]

            tenure = ''
            cat = ''
            color_1 = 0
            color_2 = 0

            if data[0] == 1:
                tenure = "Chronic Medical Condition"
                color_1 = 1
            elif data[0] == 0:
                tenure = "Acute Medical Condition"
                color_1 = 0
            
            if data[1] == 1:
                cat = "Urgent Medical Assisstance required!"
                color_2 = 1
            elif data[1] == 0:
                cat = "Urgent Medical Assisstance NOT required"
                color_2 = 0

            data = [tenure, cat, user_name, given_disease, color_1, color_2]

            for item in data:
                if item is None:
                    return redirect(url_for("home"))
                else:
                    return render_template("result.html", data=data)

    else:
        return render_template("prediction_model.html")
    
if __name__ == "__main__":
    app.run(debug=True)


