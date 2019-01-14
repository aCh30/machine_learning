import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import re
import nltk
from sklearn import model_selection

data= pd.read_csv('amazon_cells_labelled.txt',header=None, delimiter='\t')
data.columns= ['Review','Sentiment']
print(data.head()) 
print(data['Review'][0])
print(data.shape)

## Download text data sets, including stop words
#nltk.download_shell()

# Import the stop word list
from nltk.corpus import stopwords 

#View the list of English-language stop words
print(stopwords.words("english"))


#Function to clean the review
def user_review(raw_review):

# Function to convert a raw review to a string of words
# The input is a single string (a raw user review), and 
# the output is a single string (a preprocessed user review)

	#
    # 1. Remove HTML
    review_text=BeautifulSoup(raw_review).get_text()
	
	#
    # 2. Remove non-letters
    letters_only= re.sub("[^a-zA-Z]", " ", review_text)
	
	#
    # 3. Convert to lower case, split into individual words
    word=letters_only.lower().split()
	
	#
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
	stops = set(stopwords.words("english"))
	
	# 
    # 5. Remove stop words
    meaningful_words =[w for w in word if w not in stops]
	
	#
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return(" ".join(meaningful_words))

# Initialize an empty list to hold the clean reviews
clean_review=[]

# Get the number of reviews based on the dataframe column size
num_review= data["Review"].size

# Loop over each review; create an index i that goes from 0 to the length
# of the user review list
for i in range(0,num_review):
    if ((i+1)%100)==0:
        print("Review %d of %d"%(i+1,num_review))
	# Call our function for each one, and add the result to the list of
    # clean reviews
    clean_review.append(user_review(data["Review"][i]))
    
    
	
train_review, test_review, train_sentiment, test_Sentiment= model_selection.train_test_split(clean_review, data["Sentiment"], test_size=0.2)

print("Creating the bag of words...\n")

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool. 
vectorizer= CountVectorizer(analyzer='word', preprocessor=None, tokenizer= None, stop_words=None, max_features=5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_features= vectorizer.fit_transform(train_review)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_features= train_features.toarray()
print("Train Feature Shape \n", train_features.shape)

vocab= vectorizer.get_feature_names()

# Sum up the counts of each vocabulary word
dist = np.sum(train_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for i,j  in zip(vocab,dist):
    print(i, j)
    

print("Training the random forest...")

# Initialize a Random Forest classifier with 100 trees	
forest= RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
forest.fit(train_features, train_sentiment)

# Get a bag of words for the test set, and convert to a numpy array. 
#Note: When using the Bag of Words for the test set, we only call "transform", not "fit_transform" as we did for the training set
#Else it might result into a risk of overfitting.
test_review_feature= vectorizer.transform(test_review)
test_review_feature= test_review_feature.toarray()
print("Test Feature Shape \n", test_review_feature.shape)

# Use the random forest to make sentiment label predictions
result= forest.predict(test_review_feature)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame(data = {"id": test_review, "sentiment": result})

# Use pandas to write the comma-separated output file
output.to_csv("Result.csv", index= False, quoting= 3)