import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col
import os
os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-11"
import h2o
h2o.init()

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

def get_row_and_prediction(row_number):
    # Get the row at the specified index
    row = data.take(row_number + 1)[-1]
    # Make prediction
    prediction = predict_liver_disease(data)

    # Display the row and prediction
    st.write("Row Details:")
    st.write(row)
    st.write("Prediction:")
    if prediction == 1.0:
        st.write("The model predicts that the patient has liver disease.")
    else:
        st.write("The model predicts that the patient does not have liver disease.")


# Define the Streamlit app
def main():
    # Set the title and sidebar
    st.title("Liver Disease Classification")
    st.sidebar.title("User Input")

    # Get row details and prediction for a specific index
    index = st.sidebar.number_input("Enter an index to retrieve row details", min_value=0, max_value=data.count()-1, step=1)
    if st.sidebar.button("Get Row Details"):
        get_row_and_prediction(index)

if __name__ == "__main__":
    main()

