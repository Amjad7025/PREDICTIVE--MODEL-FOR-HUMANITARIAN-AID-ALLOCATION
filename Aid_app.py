from flask import Flask, request, render_template
import pickle
import numpy as np

# Load pickle files
with open("SCALE_MODEL.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("LABEL_ENCODER.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("AID_ALLOCATION.pkl", "rb") as f:
    model = pickle.load(f)


# with open("features.pkl.pkl", "rb") as f:
#     model = pickle.load(f)

app = Flask(__name__)

# mapping
urgency_level_map={"Medium":1,'Low':0,'Critical':3,'High':2}

accessibility_map={'Difficult':3,'Moderate':2,'Easy':0,'Normal':1}

disaster_type_map={'Cyclone':0,'Flood':4,'Landslide':2,'Earthquake':1,'Drought':3,'volcano_erruption':5,}

severity_map={'low':0,'normal':1,'medium':2,'high':3,'extream':4}



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    
    severity=severity_map[request.form['severity']]
    population_affected_rate=int(request.form['population_affected_rate'])
    economic_loss=int(request.form['economic_loss'])
    infrastructure_damage=int(request.form['infrastructure_damage'])
    disaster_type=disaster_type_map[request.form['disaster_type']]
    urgency_level=urgency_level_map[request.form['urgency_level']]
    accessibility=accessibility_map[request.form['accessibility']]



    # Convert to numpy array
    input_data = np.array([[severity,population_affected_rate,economic_loss,infrastructure_damage,disaster_type,urgency_level,accessibility]])




    prediction = model.predict(input_data)[0]

    if prediction <= 0:
        return render_template("index.html", result=f"Aid not Required")
    else:
        return render_template("index.html", result=f"Aid Required Value: {prediction:.2f}")






if __name__ == "__main__":
    app.run(debug=True)