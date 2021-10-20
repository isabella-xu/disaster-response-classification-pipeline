# disaster-response-classification-pipeline
This project is part of the Udacity Data Scientist Nanodegree Program.

## Project motivation
After a natural disaster, people will send out messages to ask for help through various channels such as social media. For example, I need food; I am trapped under the rubble. However, the government does not have enough time to read all the messages and send them to various departments. Therefore, this project is target to provide a easy portal for emergency workers to input new message and get classification results based on categories.

This data used in this proejct is thousands of real messages provided by [Figure Eight](https://appen.com/) (acquired by Appen), sent during natural disasters either via social media or directly to disaster response organizations.

## Files Description
This project contains three parts:
1. ETL pipeline: extract thousands of data from the source csv files, clean and transform the data, save the data to a db file.
2. ML pipeline: a machine leanring pipeline uses NLTK, scikit-learn, GridSearchCV to predict classifcation for 36 message categories(multi-output classification).
3. Flask Web application: provide a portal to enter new message and get the classification results in several categories. This web app also disply visuallization of the data.
```
.
├── app     
│   ├── run.py                           # Flask file that runs app
│   └── templates   
│       ├── go.html                      # Classification result page of web app
│       └── master.html                  # Main page of web app    
├── data                   
│   ├── disaster_categories.csv          # Dataset including all the categories  
│   ├── disaster_messages.csv            # Dataset including all the messages
│   └── process_data.py                  # Data cleaning
├── models
│   └── train_classifier.py              # Train ML model           
└── README.md
```

## Dependencies for installation
+ Python (>=3.7)
+ sys
+ pandas 
+ numpy 
+ nltk
+ sqlalchemy
+ scikit-learn 
+ plotly
+ joblib
+ flask
+ pickle

## Deployment
```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db  

python train_classifier.py ../data/DisasterResponse.db classifier.pkl

python run.py

# Note: by default, it is set to run at localhost: http://127.0.0.1:3001/ .
```


## Licensing, Authors, and Acknowledgements
### Author: 
Isabella Xu<br/>

### Acknowledgements
This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. Please refer to Udacity Terms of Service for further information.
