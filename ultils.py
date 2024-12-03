import csv
import uuid
import cv2
import os
from flask import Flask,request,render_template,jsonify
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import shutil
from flask_mail import Mail, Message

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

mail = Mail(app)

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### get name and id of all users
def getusers():
    nameUsers = []
    idUsers = []
    emailUsers = [] 
    l = len(os.listdir('static/faces'))
    print(os.listdir('static/faces'))
    if l == 0: 
        print('No user')
    for user in os.listdir('static/faces'):
        print(user)
        nameUsers.append(user.split('_')[0])
        idUsers.append(user.split('_')[1])
        emailUsers.append(user.split('_')[2])
    return nameUsers, idUsers, l, emailUsers

#### delete user
def delUser(userid, username, email):
    for user in os.listdir('static/faces'):
        if user.split('_')[1] == userid:
            shutil.rmtree(f'static/faces/{username}_{userid}_{email}', ignore_errors=True)

#### extract the face from an image
def extract_faces(img):
    if img is not None and img.size > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        if len(face_points):  # Check if face_points is not empty
            return face_points
    return None

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    print(userlist)
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
            
    if len(faces) == 0:
        return
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    file_path = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.isfile(file_path):
        # Tạo file CSV nếu chưa tồn tại
        with open(file_path, 'w') as f:
            f.write('Name,ID,Check_in_time,Check_out_time,Total_time\n')
        return [], [], [], [], [], 0

    try:
        # Đọc file CSV
        df = pd.read_csv(file_path)

        # Kiểm tra và xử lý các cột nếu thiếu giá trị
        required_columns = ['Name', 'ID', 'Check_in_time', 'Check_out_time', 'Total_time']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''  # Thêm cột nếu thiếu và để giá trị mặc định là rỗng

        # Thay thế giá trị NaN bằng chuỗi rỗng
        df.fillna('', inplace=True)

        # Trích xuất dữ liệu
        names = df['Name']
        rolls = df['ID']
        inTimes = df['Check_in_time']
        outTimes = df['Check_out_time']
        totalTimes = df['Total_time']
        l = len(df)

        return names, rolls, inTimes, outTimes, totalTimes, l

    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return [], [], [], [], [], 0


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    file_path = f'Attendance/Attendance-{datetoday}.csv'

    # Nếu file không tồn tại, tạo file mới với tiêu đề
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'ID', 'Check_in_time', 'Check_out_time', 'Total_time'])

    # Đọc dữ liệu hiện có
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        existing_ids = [row['ID'] for row in reader]


    # Kiểm tra xem user đã có trong file hay chưa
    if userid not in existing_ids:
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([username, userid, current_time, '', ''])  # Check_out_time & Total_time trống
    else:
        # Nếu `userid` tồn tại, cập nhật Check_out_time và Total_time
        df = pd.read_csv(file_path)
        row_index = df[df['ID'] == userid].index[0]

        if pd.isna(df.loc[row_index, 'Check_out_time']):
            df.loc[row_index, 'Check_out_time'] = current_time

            in_time = datetime.strptime(df.loc[row_index, 'Check_in_time'], '%H:%M:%S')
            out_time = datetime.strptime(df.loc[row_index, 'Check_out_time'], '%H:%M:%S')

            total_time = out_time - in_time
            df.loc[row_index, 'Total_time'] = str(total_time)

            # Ghi đè lại file
            df.to_csv(file_path, index=False)

#Get check in and out time of user
def getUserTime(userid):
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    row_index = 0

    for i in range(0, df['ID'].count()):
        if str(df['ID'][i]) == userid:
            row_index = i
            break
            
    return str(df['Check_in_time'][row_index]), str(df['Check_out_time'][row_index])

#Check existed userID
def checkUserID(newuserid):
    listID = os.listdir('static/faces')
    for i in range(0, len(listID)):
        if listID[i].split('_')[1] == newuserid:
            return True
    return False

def send_email(to_email, username):
    """
    Gửi email thông báo cho nhân viên sau khi thêm user thành công.
    """
    subject = "Thông Báo: Đăng Ký Thành Công"
    body = f""" <html> 
        <body> 
            <div style="font-family: Arial, sans-serif; line-height: 1.6;"> 
                <h2 style="color: #4CAF50;">Xin chào {username},</h2> 
                <p>Bạn đã được thêm thành công vào hệ thống chấm công.</p> 
                <p>Chào mừng bạn đến với đội ngũ của chúng tôi!</p> <p>Trân trọng,</p> 
                <p><strong>Đội ngũ quản lý</strong></p> 
                <hr style="border: 0; border-top: 1px solid #eee;"> 
                <p style="font-size: 0.9em; color: #555;">Nếu bạn có bất kỳ câu hỏi nào, vui lòng liên hệ với chúng tôi qua email này.</p> 
            </div> 
        </body> 
    </html> """

    try:
        msg = Message(subject, recipients=[to_email], html=body)
        mail.send(msg)
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")
