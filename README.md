# SYND-iNNOVATE
Repository for Syndicate Hackathon

# Alpha A.I
Team members:
Pruthvi Raj R.G,Om Shri Prasath,Adithya Swaroop S,Nikhil Mattapally

# Voice based customer grievance redressal system

# OVERVIEW
In this submission we are proposing a voice based customer grievance redressal system which is
able to help with basic requests on its own and able to register and track other requests with and
help them reach their respective departments efficiently.

# GOALS
1. Able to converse with the customer and extract important information from the voice.(Like
account number etc..)
2. Provide basic info like bank balance/transactions and do operations like blocking their
account.
3. Register and track the complaints with provision of having centralised monitoring. That is
it should be able to route customer queries to the respective departments.

# Technology stack
BERT, ANDROID SDK, VOLLEY ,PYTORCH, FLASK ,PYTHON

# Working method

# Speech recognition
Our first step is to convert voice input given to the system to text.This is done by using google
voice API. Advantages of using google API is we will be able to scale our model to all major
Indian languages.

# Intent classification
The next step in our system is to understand the text for its intent(if the text means a bank
balance query or some other question).This we are doing it using BERT(Bidirectional Encoder
Representations from Transformers) which is a state of the art general purpose NLP architecture
by Google. First we trained our intent classifier using a set of training examples and later used
the model to classify the text obtained from Google voice API.

we are planning to use mysql for database management (for details of customers).
# Hosting the program
Finally host our programs on server which is created using flask, which is connected to an
Android app for the initial phase testing.
