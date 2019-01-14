# **Online_product_user_review**

The code predicts the sentiments of the user review for products to be Good(1) or Bad(0) using Natural Language Processing

## **Overview**

NLP (Natural Language Processing) is a set of techniques for approaching text problems. The code deals with loading and cleaning of the user review data for products available on an e-commerce site, then applying a simple Bag of Words model to get surprisingly accurate predictions of whether a review is thumbs-up or thumbs-down.

It uses Random Forest algorithm to fit the model using the training dataset and then predicts the sentiments based on user review on the test dataset.


## Dependencies

- Numpy
- Scikit-learn
- Pandas
- BeautifulSoup4
- re (Regular Expressions)
- nltk (Natural Language Toolkit)


Python 2 and 3 both work for this. Use [pip](https://pip.pypa.io/en/stable/) to install any dependencies.

## Usage

Run python3 User_Review.py to see the results. The sentiment prediction on the test data will be saved in a file generated called Result.csv.