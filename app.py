import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data



def main():
	"""Simple Login App"""

	st.title("Sign Language WebApp")

# Set the background image
	background_image = "bsign.jpg"
	st.image(background_image,use_column_width=True)
# Display the image
	# st.image(background_image, use_column_width=True)

	# st.image("sign.jpg")

	menu = ["Home","SignUp","Login"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")

	elif choice == "Login":
		st.subheader("Login Section")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:

				st.success("Logged In as {}".format(username))

				task = st.selectbox("Task",["Signs","Prediction","Profiles"])
				if task == "Signs":
						st.title('Images of Sign Language')
						col1, col2, col3, col4, col5 ,col6, col7, col8,col9, col10 = st.columns(10)

						with col1:
							st.header("0")
							st.image("Dataset/train/0/IMG_1158.JPG")
							st.image("Dataset/train/0/IMG_1169.JPG")
							st.image("Dataset/train/0/IMG_1322.JPG")
							st.image("Dataset/train/0/IMG_5203.JPG")
							st.image("Dataset/train/0/IMG_5350.JPG")
       
         
						with col2:
							st.header("1")
							st.image("Dataset/train/1/IMG_1159.JPG")
							st.image("Dataset/train/1/IMG_1313.JPG")
							st.image("Dataset/train/1/IMG_4070.JPG")
							st.image("Dataset/train/1/IMG_5646.JPG")
							st.image("Dataset/train/1/IMG_5573.JPG")
            

						with col3:
							st.header("2")
							st.image("Dataset/train/2/IMG_4071.JPG")
							st.image("Dataset/train/2/IMG_4091.JPG")
							st.image("Dataset/train/2/IMG_1231.JPG")
							st.image("Dataset/train/2/IMG_1130.JPG")
							st.image("Dataset/train/2/IMG_5215.JPG")
       
						with col4:
							st.header("3")
							st.image("Dataset/train/3/IMG_1325.JPG")
							st.image("Dataset/train/3/IMG_4092.JPG")
							st.image("Dataset/train/3/IMG_1325.JPG")
							st.image("Dataset/train/3/IMG_4072.JPG")
							st.image("Dataset/train/3/IMG_4174.JPG")
        
						with col5:
							st.header("4")
							st.image("Dataset/train/4/IMG_5187.JPG")
							st.image("Dataset/train/4/IMG_5136.JPG")
							st.image("Dataset/train/4/IMG_4900.JPG")
							st.image("Dataset/train/4/IMG_4940.JPG")
							st.image("Dataset/train/4/IMG_5115.JPG")
        
						with col6:
							st.header("5")
							st.image("Dataset/train/5/IMG_4699.JPG")
							st.image("Dataset/train/5/IMG_4911.JPG")
							st.image("Dataset/train/5/IMG_4931.JPG")
							st.image("Dataset/train/5/IMG_4699.JPG")
							st.image("Dataset/train/5/IMG_4821.JPG")
        
						with col7:
							st.header("6")
							st.image("Dataset/train/6/IMG_1318.JPG")
							st.image("Dataset/train/6/IMG_4045.JPG")
							st.image("Dataset/train/6/IMG_1185.JPG")
							st.image("Dataset/train/6/IMG_1235.JPG")
							st.image("Dataset/train/6/IMG_1297.JPG")
        
						with col8:
							st.header("7")
							st.image("Dataset/train/7/IMG_4251.JPG")
							st.image("Dataset/train/7/IMG_4313.JPG")
							st.image("Dataset/train/7/IMG_1155.JPG")
							st.image("Dataset/train/7/IMG_4410.JPG")
							st.image("Dataset/train/7/IMG_4443.JPG")
       
						with col9:
							st.header("8")
							st.image("Dataset/train/8/IMG_4964.JPG")
							st.image("Dataset/train/8/IMG_5061.JPG")
							st.image("Dataset/train/8/IMG_5130.JPG")
							st.image("Dataset/train/8/IMG_5150.JPG")
							st.image("Dataset/train/8/IMG_5221.JPG")
       
						with col10:
							st.header("9")
							st.image("Dataset/train/9/IMG_5506.JPG")
							st.image("Dataset/train/9/IMG_5581.JPG")
							st.image("Dataset/train/9/IMG_5664.JPG")
							st.image("Dataset/train/9/IMG_5714.JPG")
							st.image("Dataset/train/9/IMG_5674.JPG")
       
            

				elif task == "Prediction":
					st.subheader("Prediction")


					mpHands = mp.solutions.hands
					hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
					mpDraw = mp.solutions.drawing_utils

					model = load_model('mp_hand_gesture')

					f = open('gesture.names', 'r')
					classNames = f.read().split('\n')
					f.close()
					print(classNames)


					cap = cv2.VideoCapture(0)
					# cap =st.camera_input("Take a picture")

					while True:
						_, frame = cap.read()

						x, y, c = frame.shape

						frame = cv2.flip(frame, 1)
						framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

						result = hands.process(framergb)

						
						className = ''

						if result.multi_hand_landmarks:
							landmarks = []
							for handslms in result.multi_hand_landmarks:
								for lm in handslms.landmark:
									# print(id, lm)
									lmx = int(lm.x * x)
									lmy = int(lm.y * y)

									landmarks.append([lmx, lmy])

								mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

								prediction = model.predict([landmarks])
								classID = np.argmax(prediction)
								className = classNames[classID]

						cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
									1, (0,0,255), 2, cv2.LINE_AA)

						cv2.imshow("Output", frame) 

						if cv2.waitKey(1) == ord('q'):
							break

					cap.release()

					cv2.destroyAllWindows()
     
     
        
        
                    
                   
				elif task == "Profiles":
					st.subheader("User Profiles")
					user_result = view_all_users()
					clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
					st.dataframe(clean_db)
			else:
				st.warning("Incorrect Username/Password")





	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")



if __name__ == '__main__':
	main()


