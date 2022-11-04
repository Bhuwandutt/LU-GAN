import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

CSV_DIR = '/Users/bhuwandutt/Documents/GitHub/LU-GAN/csv/'

dataframe = pd.read_csv(CSV_DIR + "/" + 'indiana_reports.csv')
dataframe = dataframe.drop(columns=['MeSH', 'image', 'Problems', 'comparison'], axis=1)
# print(dataframe.head)



def checkNan(datafram: pd.DataFrame):
    NaN = datafram.isnull().sum()
    i = 0
    for column in datafram.columns:
        print(f"Total Nan Values in {column} columns - {NaN[i]}")
        i += 1


def decontracted(phrase):  # Performs text de-contraction of words like won't to will not

    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase


def preprocess_text(data):

    # extracts the information data from the xml file and does text preprocessing on them
    # here info can be 1 value in this list ["COMPARISON","INDICATION","FINDINGS","IMPRESSION"]

    preprocessed = []

    for sentence in tqdm(data.values):

        sentence = BeautifulSoup(sentence, 'lxml').get_text()

        regex = r"\d."
        sentence = re.sub(regex, "", sentence)  # removing all values like "1." and "2." etc

        regex = r"X+"
        sentence = re.sub(regex, "", sentence)  # removing words like XXXX

        regex = r"[^.a-zA-Z]"
        sentence = re.sub(regex, " ", sentence)  # removing all special characters except for full stop

        regex = r"http\S+"
        sentence = re.sub(regex, "", sentence)
        sentence = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?$%^&*'/+\[\]_]+", "", sentence)
        sentence = re.sub('&', 'and', sentence)
        sentence = re.sub('@', 'at', sentence)
        sentence = re.sub('0', 'zero', sentence)
        sentence = re.sub('1', 'one', sentence)
        sentence = re.sub('2', 'two', sentence)
        sentence = re.sub('3', 'three', sentence)
        sentence = re.sub('4', 'four', sentence)
        sentence = re.sub('5', 'five', sentence)
        sentence = re.sub('6', 'six', sentence)
        sentence = re.sub('7', 'seven', sentence)
        sentence = re.sub('8', 'eight', sentence)
        sentence = re.sub('9', 'nine', sentence)
        sentence = re.sub('year old', "", sentence)  # Occur multiple times in Indication feature but not necessary
        sentence = re.sub('yearold', "", sentence)
        sentence = decontracted(sentence)  # perform decontraction
        sentence = sentence.strip().lower()  # Strips the begining and end of the string of spaces and converts all
        # into lowercase
        sentence = " ".join(sentence.split())  # removes unwanted spaces
        if sentence == "":  # if the resulting sentence is an empty string return null value
            sentence = np.nan
        preprocessed.append(sentence)

    return preprocessed

# Replacing the nan values


column_list = list(dataframe)


# dataframe['comparison'] = dataframe['comparison'].fillna('No Comparison')
dataframe['indication'] = dataframe['indication'].fillna('No Indication')
dataframe['findings'] = dataframe['findings'].fillna('No Findings')
dataframe['impression'] = dataframe['impression'].fillna('No Impression')


# dataframe['MeSH'] = preprocess_text(dataframe['MeSH'])
# dataframe['comparison'] = preprocess_text(dataframe['comparison'])
dataframe['indication'] = preprocess_text(dataframe['indication'])
dataframe['findings'] = preprocess_text(dataframe['findings'])
dataframe['impression'] = preprocess_text(dataframe['impression'])

checkNan(datafram=dataframe)
# dataframe['image'] = preprocess_text(dataframe['image'])
# dataframe['Problems'] = preprocess_text(dataframe['Problems'])

dataframe.replace("", "No Value", inplace=True)
dataframe['indication_count'] = dataframe['indication'].astype(str).str.split().apply(lambda x: 0 if x == None else len(x))
dataframe['findings_count'] = dataframe['findings'].astype(str).str.split().apply(lambda x: 0 if x == None else len(x))
dataframe['impression_count'] = dataframe['impression'].astype(str).str.split().apply(lambda x: 0 if x == None else len(x))


print(dataframe.head())

dataframe['indication'] = dataframe['indication'].fillna('No Indication')
dataframe['findings'] = dataframe['findings'].fillna('No Findings')
dataframe['impression'] = dataframe['impression'].fillna('No Impression')

checkNan(datafram=dataframe)

df = dataframe.sample(frac=1).reset_index(drop=True)
df.to_csv(path_or_buf=CSV_DIR+'/indiana_reports_cleaned.csv', index=False)
n_rows = df.shape[0]
df_train = df.iloc[0: math.floor(n_rows * 0.03), :]
df_val = df.iloc[math.floor(n_rows * 0.03):math.floor(n_rows * 0.05), :]
df_test = df.iloc[math.floor(n_rows * 0.05):math.floor(n_rows * 0.06), :]
print(dataframe.shape)


df_train.to_csv(path_or_buf=CSV_DIR+'/indiana_reports_train.csv', index=False)
df_val.to_csv(path_or_buf=CSV_DIR+'/indiana_reports_val.csv', index=False)
df_test.to_csv(path_or_buf=CSV_DIR+'/indiana_reports_test.csv', index=False)
