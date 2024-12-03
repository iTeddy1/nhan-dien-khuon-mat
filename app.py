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
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['ID']
    inTimes = df['Check_in_time']
    outTimes = df['Check_out_time']
    totalTimes = df['Total_time']
    l = len(df)
    return names,rolls,inTimes,outTimes,totalTimes,l

#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

    if int(userid) not in list(df['ID']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time},'',''')
    else:
        row_index = 0

        for i in range(0, df['ID'].count()):
            if str(df['ID'][i]) == userid:
                row_index = i
                break

        if str(df['Check_out_time'][row_index]) == 'nan':
            df.loc[row_index, 'Check_out_time'] = current_time

            inTime = (datetime.strptime(df['Check_in_time'][row_index], '%H:%M:%S'))
            outTime = (datetime.strptime(df['Check_out_time'][row_index], '%H:%M:%S'))

            totalTime = outTime - inTime

            df.loc[row_index, 'Total_time'] = totalTime

            df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

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
    body = f"""
    Xin chào {username},

    Bạn đã được thêm thành công vào hệ thống chấm công.

    Trân trọng,
    Đội ngũ quản lý.
    """

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
        useremail = identified_person.split('_')[2]


    
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
    newuserid = request.form['newuserid']
    newuseremail = request.form['newuseremail']

    if checkUserID(newuserid) == False:
        userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid) + '_' + newuseremail
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

        # Gửi email
        send_email(newuseremail, newusername)

        names, rolls, inTimes, outTimes, totalTimes, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes, totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    else:
        names, rolls, inTimes, outTimes, totalTimes, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, inTimes=inTimes, outTimes=outTimes, totalTimes=totalTimes, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='User ID has existed. Please type other ID.')

#### Our main function which runs the Flask App
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port='6969')



