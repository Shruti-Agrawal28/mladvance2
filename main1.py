import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col
import os

os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-11"

# Create a Spark session
spark = SparkSession.builder.appName("LiverDiseaseClassification").getOrCreate()

# Read the dataset into a DataFrame
data = spark.read.csv("indian_liver_patient.csv", header=True, inferSchema=True)

# Handle missing values
data = data.na.drop()

# Prepare the data for modeling
assembler = VectorAssembler(
    inputCols=[
        "Age",
        "GenderIndex",
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphotase",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Protiens",
        "Albumin",
        "Albumin_and_Globulin_Ratio"
    ],
    outputCol="features"
)


# Define a function to make predictions on user input
def predict_liver_disease(input_df):
    gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")
    input_df = gender_indexer.fit(input_df).transform(input_df)
    input_df = assembler.transform(input_df)
    input_df = input_df.withColumn("label", col("Dataset").cast("double"))
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
    model = rf.fit(input_df)

    # Make predictions
    predictions = model.transform(input_df)

    # Extract the predicted label
    prediction = predictions.select("prediction").first()[0]

    return prediction


# Define custom CSS styles
styles = """
    .container {
        width: 500px;
        margin: auto;
    }
    .input-label {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .input-field {
        font-size: 16px;
        padding: 5px;
        margin-bottom: 10px;
        width: 100%;
    }
    .submit-button {
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
    }
    .result {
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
        white-space: pre-line;
    }
"""

# Define the Streamlit app
def main():
    # Set the title and sidebar
    st.title("Liver Disease Classification")
    st.sidebar.title("User Input")

    # Get row details and prediction for a specific index
    index = st.sidebar.number_input("Enter an index to retrieve row details", min_value=0, max_value=data.count()-1, step=1)
    if st.sidebar.button("Get Row Details"):
        # Get the row at the specified index
        row = data.take(index + 1)[-1]
        # Make prediction
        prediction = predict_liver_disease(data)

        # Save the result to a file
        result_file = "result.txt"
        with open(result_file, "w") as f:
            f.write("Row Details:\n")
            f.write(str(row) + "\n")
            f.write("Prediction:\n")
            f.write("The model predicts that the patient has liver disease." if prediction == 1.0 else "The model predicts that the patient does not have liver disease.")

        # Display the result in Streamlit
        with open(result_file, "r") as f:
            result = f.read()
            st.text_area("Result", result)

if __name__ == "__main__":
    main()
