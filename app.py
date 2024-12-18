import uuid
import cv2
import os
from flask import Flask,request,render_template
from flask_mail import Mail, Message
from datetime import date
from ultils import evaluate_model, handle_checkin_checkout, save_attendance_image, totalreg, getusers, delUser, extract_faces, identify_face, train_model, add_attendance, extract_attendance, getUserTime, checkUserID

#### Defining Flask App
app = Flask(__name__)

# Cấu hình mail server
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'duytrung.ng1@gmail.com'  
app.config['MAIL_PASSWORD'] = 'qmwv uicf tpcu edmu'  
app.config['MAIL_DEFAULT_SENDER'] = 'duytrung.ng1@gmail.com'

mail = Mail(app)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('ID,Name,Check_in_time,Check_out_time,Total_time\n')
if not os.path.isdir(f'Attendance/Attendance_faces-{datetoday}'):
    os.makedirs(f'Attendance/Attendance_faces-{datetoday}')

################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names,rolls,inTimes,outTimes,totalTimes,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,inTimes=inTimes,outTimes=outTimes,totalTimes=totalTimes,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

@app.route('/listUsers')
def users():
    names, rolls, l, emails = getusers()
    return render_template('ListUser.html', names= names, rolls=rolls, l=l, emails=emails, totalreg=totalreg())

@app.route('/deleteUser', methods=['POST'])
def deleteUser():
    userid = request.form['userid']
    username = request.form['username']
    useremail = request.form['useremail']
    delUser(userid, username, useremail)
    train_model()
    names, rolls, l, emails = getusers()
    return render_template('ListUser.html', names= names, rolls=rolls, l=l, emails=emails, mess=f'User {username} deleted successfully.')

#### This function will run when we click on Check in / Check out Button
@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder.')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return render_template(
            'home.html',
            totalreg=totalreg(),
            datetoday2=datetoday2,
            mess='Unable to access the camera.'
        )
    
    ret = True
    identified_person = None    
    escKey = cv2.waitKey

    while ret:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            render_template('home.html',mess="Failed to grab frame")  # Exit the loop if no frame is captured

        faces = extract_faces(frame)
        if faces is not None:  # Check if faces is not None
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            # print(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

        cv2.imshow('Attendance', frame)

        if escKey(1) == 27:  # Exit on ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
 
    # identified_person: chuỗi có dạng name_id_email
    result_message = handle_checkin_checkout(identified_person)

    names,rolls,inTimes,outTimes,totalTimes,l = extract_attendance()    

    if result_message == "Bạn đã chấm công cho ngày hôm nay.":
        return render_template('home.html', 
                               names=names,rolls=rolls,inTimes=inTimes,outTimes=outTimes,totalTimes=totalTimes,l=l,totalreg=totalreg(),datetoday2=datetoday2,
                               mess=result_message)

    #Save attendance image
    username = identified_person.split('_')[0]
    userid = identified_person.split('_')[1]
    # userimagefolder = f'Attendance/Attendance_faces-{datetoday}/'+username+'_'+str(userid)+'_'+datetoday2
    # if not os.path.isdir(userimagefolder):
    #     os.makedirs(userimagefolder)
    inTime, outTime = getUserTime(userid)

    # print(inTime, outTime)
    if inTime != 'nan':
        save_attendance_image(username, userid, frame, x, y, w, h, 'checkin')
    if outTime != 'nan':
        save_attendance_image(username, userid, frame, x, y, w, h, 'checkout')

    return render_template('home.html',names=names,rolls=rolls,inTimes=inTimes,outTimes=outTimes,totalTimes=totalTimes,l=l,totalreg=totalreg(),datetoday2=datetoday2, mess=result_message) 

#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form.get('newusername', '').strip()
    newuseremail = request.form.get('newuseremail', '').strip()
    newuserid = str(uuid.uuid4())[:8]  # Lấy 8 ký tự đầu tiên của UUID

    # Kiểm tra nếu User ID đã tồn tại
    if checkUserID(newuserid) == False:
        user_image_folder = f'static/faces/{newusername}_{newuserid}_{newuseremail}'
        os.makedirs(user_image_folder, exist_ok=True)

        cap = cv2.VideoCapture(0)
        i, j = 0, 0
        while 1:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            faces = extract_faces(frame)
            if faces is not None:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                    cv2.putText(frame, f'Images Captured: {i}/100', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    if j % 10 == 0:
                        name = f'{newusername}_{i}.jpg'
                        cv2.imwrite(f'{user_image_folder}/{name}', frame[y:y + h, x:x + w])
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
        add_attendance(f"{newusername}_{newuserid}_{newuseremail}") # username_id_email
        # Trả về giao diện
        names, rolls, l, emails = getusers()
        return render_template('ListUser.html', names= names, rolls=rolls, l=l, emails=emails, mess=f'User {newusername} added successfully.')
    else:
        names, rolls, l, emails = getusers()
        return render_template('ListUser.html', names= names, rolls=rolls, l=l, emails=emails, mess=f'User {newusername} added failed.')

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

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
    app.run(host='0.0.0.0', port='6969')
