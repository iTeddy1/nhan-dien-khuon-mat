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

# Cấu hình mail server
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'duytrung.ng1@gmail.com'  # Email của bạn
app.config['MAIL_PASSWORD'] = 'nics wtpn qhcq dvbo'  # Mật khẩu ứng dụng
app.config['MAIL_DEFAULT_SENDER'] = 'duytrung.ng1@gmail.com'

mail = Mail(app)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,ID,Check_in_time,Check_out_time,Total_time')
if not os.path.isdir(f'Attendance/Attendance_faces-{datetoday}'):
    os.makedirs(f'Attendance/Attendance_faces-{datetoday}')


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
        msg = Message(subject, recipients=[to_email], body=body)
        mail.send(msg)
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names,rolls,inTimes,outTimes,totalTimes,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,inTimes=inTimes,outTimes=outTimes,totalTimes=totalTimes,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

@app.route('/listUsers')
def users():
    names, rolls, l, emails = getusers()
    return render_template('ListUser.html', names= names, rolls=rolls, l=l, emails=emails)

@app.route('/deletetUser', methods=['POST'])
def deleteUser():
    userid = request.form['userid']
    username = request.form['username']
    useremail = request.form['useremail']
    delUser(userid, username, useremail)
    train_model()
    names, rolls, l, emails = getusers()
    return render_template('ListUser.html', names= names, rolls=rolls, l=l, emails=emails)

#### This function will run when we click on Check in / Check out Button
@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = cv2.VideoCapture(0)
    ret = True
    identified_person = None

    while ret:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            break  # Exit the loop if no frame is captured

        faces = extract_faces(frame)
        if faces is not None:  # Check if faces is not None
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27: # phim esc break chuong trinh
            break

    cap.release()
    cv2.destroyAllWindows()

    if identified_person:
        add_attendance(identified_person)
        username = identified_person.split('_')[0]
        userid = identified_person.split('_')[1]
    
    names,rolls,inTimes,outTimes,totalTimes,l = extract_attendance()    

    #Save attendance image
    username = identified_person.split('_')[0]
    userid = identified_person.split('_')[1]
    userimagefolder = f'Attendance/Attendance_faces-{datetoday}/'+username+'_'+str(userid)+'_'+datetoday2
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    inTime, outTime = getUserTime(userid)

    print(inTime, outTime)
    if inTime != 'nan':
        name = username+'_'+userid+'_'+'checkin'+'.jpg'
        if name not in os.listdir(userimagefolder):
            cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
    if outTime != 'nan':
        name = username+'_'+userid+'_'+'checkout'+'.jpg'
        if name not in os.listdir(userimagefolder):
            cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])

    return render_template('home.html',names=names,rolls=rolls,inTimes=inTimes,outTimes=outTimes,totalTimes=totalTimes,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuseremail = request.form['newuseremail']
    # Tự động tạo user ID bằng cách hash email
    newuserid = str(uuid.uuid4())[:8]  # Lấy 8 ký tự đầu tiên của UUID

    # Kiểm tra nếu User ID đã tồn tại
    if checkUserID(newuserid) == False:
        userimagefolder = f'static/faces/{newusername}_{newuserid}_{newuseremail}'
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        cap = cv2.VideoCapture(0)
        i, j = 0, 0
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            if faces is not None:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                    cv2.putText(frame, f'Images Captured: {i}/100', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    if j % 10 == 0:
                        name = newusername + '_' + str(i) + '.jpg'
                        cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                        i += 1
                    j += 1
                if j == 1000:
                    break
                cv2.imshow('Adding new User', frame)
                if cv2.waitKey(1) == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()

        # Gửi email thông báo thành công
        send_email(newuseremail, newusername)

        # Thêm người dùng vào bảng Attendance
        add_attendance(f"{newusername}_{newuserid}")

        # Trả về giao diện
        names, rolls, inTimes, outTimes, totalTimes, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes, totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    else:
        names, rolls, inTimes, outTimes, totalTimes, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes, totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='User ID has existed. Please type other ID.')

#### Our main function which runs the Flask App
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port='6969')



