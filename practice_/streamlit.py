# to run --> streamlit run streamlit.py
import streamlit as st

st.title("Intro")
st.write("This is a learning place")
st.write("Yes it is running")

name =st.text_input("Your name : ")

if(name):
    st.write("Welcome let's Build !")
    
age =st.slider("Your age : ",0,100)

edu = st.radio("Status : " , ("Studnet", "UG", "PG"))

skills = st.selectbox("Skills : " , ("python", "OpenCV" , "CNNs"))

if st.button("Show"):
    st.write(f"Here's your data : {edu} {skills} {age} {name}")
    
    
    
all_clicks=[]

data ={'name' : name, 'edu' : edu , 'age' : age , 'skills' : skills}

all_clicks.append(data)

if st.button("Show all data"):
    st.write(all_clicks)