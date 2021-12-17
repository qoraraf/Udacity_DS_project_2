
# Disaster Response Pipeline Web-Application

### Introduction:
This project is a part of the Data Scientist nanodegree at Udacity with partnership of Misk & Sdaia. Its goal is to deploy an NLP model to a web application that is then used to filter incoming diasters' help request messages to ease the organizations work.<br>
The application trains any new data and classifies incoming messages

### Files:
1. data: 
    - DisasterResponse.db, this is a database where data is fetched. It is produced by "process_data.py"
    - ETL Pipeline Preparation.ipynb
    - ML Pipeline Preparation.ipynb
    - disaster_categories.csv, contains the categories of the messages
    - disaster_messages.csv, contains the messages
    - process_data.py, pipeline that cleans data and stores in database

3. models:
    - classifier.pkl, this is the fitted model constructed by the "train_classifier.py"
    - train_classifier.py, the ML classifier pipeline py file: it uses the sqlite database to train data within it and returns a model that classifies messages

5. app:
    - run.py, runs the web app
    - templates, folder contains html files to return the web app pages

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ (use your localhost url)

![webapp](https://user-images.githubusercontent.com/29784542/146616307-66055c6e-f762-4424-9957-11ec9c2e6aa1.PNG)
![classified](https://user-images.githubusercontent.com/29784542/146616472-595cf4e0-813c-42dc-b4ff-1abc07c1f6c5.PNG)
