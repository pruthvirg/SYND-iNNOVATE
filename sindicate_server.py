
#Importing necessary packages
import requests
from flask import Flask, render_template, request
from flask import jsonify
import re
import os
import pandas as pd
import numpy as np

database = pd.read_csv('./sample_data.csv')
global state
state = 0
global accno

bert_url = "http://127.0.0.1:5000/bert"

def final_function(query):
  global state
  global accno
  squery=query.split()
  if query == 'quit':
    state=0
  if state == 0:
    if query[0]=='0':#hi
      return "Heyoo! I'm Syndica. I can help you to get info on your account balance or your recent transactions or to block/unblock your debit card"
    elif query[0]=='1':
      return 'bye'#todoo
    elif query[0]=='2':#balance
      state=1
    elif query[0]=='3':
      state=2
    elif query[0]=='4':
      state=3
    else:
      return query
  if state == 1:
    for each in squery[1:]:
      if each.isdigit() and (len(each)==4 or len(each)==11):
        if int(each) in database['numbers'].values:
          state=0
          return 'Your current balance is ' +str(database[database['numbers']==int(each)]['balance'].values[0])
        else:
          state=11
          return 'Please say your last four digits of account number again'
    state=11
    return 'Please tell me your last four digits of account number'
  elif state ==11:
    if squery[1].isdigit() and (len(squery[1])==4 or len(squery[1])==11):
      if int(squery[1]) in database['numbers'].values:
        state = 0
        return 'Your balance is ' +str(database[database['numbers']==int(squery[1])]['balance'].values[0])
      else:
        return 'Account number not found'
    else:
      return 'Please say a valid account number'
  elif state == 2:
    for each in squery[1:]:
      if each.isdigit() and (len(each)==4 or len(each)==11):
        if int(each) in database['numbers'].values:
          state=0
          return 'Your recent transactions are'+database[database['numbers']==int(each)]['transactions'].values[0]
        else:
          state=12
          return 'Please say your last four digits of account number again'
    state=12
    return 'Please tell me your last four digits of account number again'
  elif state ==12:
    if squery[1].isdigit() and (len(squery[1])==4 or len(squery[1])==11):
      if int(squery[1]) in database['numbers'].values:
        state = 0
        return 'Your recent transactions are ' +database[database['numbers']==int(squery[1])]['transactions'].values[0]
      else:
        return 'Account number not found'
    else:
      return 'Please tell me a valid account number'
  elif state == 3:
    state =13
    return 'Are you sure you want to block your account ?!'
  elif state == 13:
    if query == 'yes':
      state =23
      return 'Tell me last four digits of your acccount number'
    elif query == 'no':
      state = 0
      return 'Thank you'
    else:
      return query
  elif state == 23:
    if squery[1].isdigit() and (len(squery[1])==4 or len(squery[1])==11):
      if int(squery[1]) in database['numbers'].values:
        state = 33
        accno=int(squery[1])
        return 'Are you sure you want to block your account?! '
      else:
        return 'Account not found'
    else:
      return 'Please tell a valid account number'
  elif state == 33:
    if query == 'yes':
      state =0
      if database.loc[database['numbers']==accno,'blocked'].values==False:
        database.loc[database['numbers']==accno,'blocked']=True
        return 'Your account has been blocked'
      elif database.loc[database['numbers']==accno,'blocked'].values==True:
        return 'Your account is already blocked'
    elif query == 'no':
      state = 0
      return 'You have chose to abort'

def get_intent(query):
	PARAMS = {"text":query}
	r = requests.get(url = bert_url,params = PARAMS)
	return r.content

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/get_action')
def final_call_app():
	query = request.args.get('query')
	query_concat = get_intent(query).decode("utf-8") +' '+query
	if query_concat in ["\"yes\"\n yes","\"no\"\n no","0\n quit"]:
	  if "yes" in query_concat:
	    query_concat = 'yes'
	  if "no" in query_concat:
	    query_concat = 'no'
	  if "quit" in query_concat:
	    query_concat = 'quit'
	return jsonify(final_function(query_concat))
	
