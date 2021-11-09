# from google.colab import drive
# drive.mount('/content/gdrive')



import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string



import mysql.connector



def readfromsql(tablename):
        # Setup MySQL connection
        db = mysql.connector.connect(
            host="localhost",              # your host, usually localhost
            user="root",            # your username
            password="Atharva1!",        # your password
            database="fakenews1db"     # name of the data base
        )   

        # You must create a Cursor object. It will let you execute all the queries you need
        cur = db.cursor()

        # Use all the SQL you like
        cur.execute("SELECT * FROM "+tablename)


        # Put it all to a data frame
        sql_data = pd.DataFrame(cur.fetchall())
        sql_data.columns = cur.column_names

        # sql_data = sql_data.iloc[0:0]
        # sql_data = pd.DataFrame(cur.fetchall())
        # sql_data.columns = cur.column_names

        # Close the session
        db.close()
        
        return sql_data
    

def writetosql(user_text,result):
#establishing the connection
    conn = mysql.connector.connect(
                host="localhost",              # your host, usually localhost
                user="root",            # your username
                password="Atharva1!",        # your password
                database="fakenews1db"     # name of the data base
            ) 

    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    # Preparing SQL query to INSERT a record into the database.
    sql = "INSERT INTO testing2 ( " + " text, result) " + "VALUES ('" + user_text +"',  '" + result +"' )"

    #try:
        # Executing the SQL command
    cursor.execute(sql)

    # Commit your changes in the database
    conn.commit()

    #except:
        # Rolling back in case of error
    #conn.rollback()

    # Closing the connection
    conn.close()    
    
df_fake = readfromsql("fake2")    
df_true = readfromsql("true1")

# df_fake = pd.read_csv("Fake3.csv")
# df_true = pd.read_csv("True.csv")

df_fake.head(5)

df_true.head(5)

df_fake["class"] = 0
df_true["class"] = 1

df_fake.shape, df_true.shape

#((23451, 5), (21417, 5))

df_fake_manual_testing = df_fake.tail(10)
for i in range(23450,23440,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)


df_fake.shape, df_true.shape


df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1


df_fake_manual_testing.head(10)


df_true_manual_testing.head(10)


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")


df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(10)


df_marge.columns


df = df_marge.drop(["title", "subject","date"], axis = 1)


df.isnull().sum()


df = df.sample(frac = 1)

df.head()


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)


df.columns

df.head()

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


df["text"] = df["text"].apply(wordopt)


x = df["text"]
y = df["class"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)



pred_lr=LR.predict(xv_test)


LRscore= LR.score(xv_test, y_test)

print("LRscore is")
print(LRscore)


print(classification_report(y_test, pred_lr))


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)


DT.score(xv_test, y_test)

DTscore= DT.score(xv_test, y_test)

print("DTscore is")
print(DTscore)

print(classification_report(y_test, pred_dt))


# from sklearn.ensemble import GradientBoostingClassifier


# GBC = GradientBoostingClassifier(random_state=0)
# GBC.fit(xv_train, y_train)

# pred_gbc = GBC.predict(xv_test)

# GBC.score(xv_test, y_test)

# print(classification_report(y_test, pred_gbc))


# from sklearn.ensemble import RandomForestClassifier

# RFC = RandomForestClassifier(random_state=0)
# RFC.fit(xv_train, y_train)

# pred_rfc = RFC.predict(xv_test)

# RFC.score(xv_test, y_test)

# print(classification_report(y_test, pred_rfc))


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    #pred_GBC = GBC.predict(new_xv_test)
    #pred_RFC = RFC.predict(new_xv_test)
     
     
    writetosql(news,output_lable(pred_LR[0]))      
    
    return "LR Prediction: {} DT Prediction: {} ".format(
                     output_lable(pred_LR[0]), 
                     output_lable(pred_DT[0])
                    #, 
                    #output_lable(pred_GBC[0])
 
)
    
    
print("training done")    



#for i in range(5) : 
# news = str(input())
# print(manual_testing(news))





# #!pip install flask
# #!pip install flask-ngrok


print("starting flask")


from flask import Flask, redirect, url_for, request , render_template
app = Flask(_name_)
# flask_ngrok.run_with_ngrok(app)

@app.route("/" , methods=['POST' , 'GET'])
def newstest() : 
  if request.method== 'POST' : 
    enterednews = request.form['news']
    print(enterednews)
    result = manual_testing(enterednews)
    print(result)
    return render_template("index.html" , news_result = result)
  else : 
   return render_template("index.html" , news_result = "")
  
if _name_ == '_main_':
   app.run()