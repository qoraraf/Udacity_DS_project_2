import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("messagesTab", engine)
    category_names = list(df.columns)
    X = df['message']
    Y = df.drop(columns = ['id', 'message' , 'original' ,'genre'])
    return X, Y, category_names



def tokenize(text):
    #Convert to lowercase
    text = text.lower()
    #Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)    
    #Tokenize words
    tokens = word_tokenize(text)
    #stop words
    stop_words = stopwords.words("english")
    #normalize word tokens
    normalizer = PorterStemmer()
    normalized = [normalizer.stem(word) for word in tokens if word not in stop_words]
    return normalized


def build_model():
    """
    Build a model to classify disaster messages
    Return:
    cv(list of str): classification model
    """
    # create a pipeline:
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    # define gridSearch parameters
    parameters = {
                'tfidf__use_idf': (True, False), 
                'clf__estimator__n_estimators': [50, 100],
                'clf__estimator__learning_rate': [2,3] 
                }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and print classification report for each category.
    Args:
    model: the classification model
    X_test: disaster test messages
    Y_test: disaster test targets
    category_names: categories
    """
    
    y_pred = model.predict(X_test)

    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_pred[:, i]),"\n\n")
    # print accuracy score 
    print('Accuracy Score: {:.3f}'.upper().format(np.mean(Y_test.values == y_pred)),"\n\n")


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()