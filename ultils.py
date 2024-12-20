import csv
import cv2
import os
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
import shutil

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### Lấy thông tin người dùng
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

#### Xóa người dùng
def delUser(userid, username, email):
    # Xóa thư mục khuôn mặt của nhân viên
    face_dir = f'static/faces/{username}_{userid}_{email}'
    if os.path.exists(face_dir):
        shutil.rmtree(face_dir, ignore_errors=True)
    
    # Xóa thông tin nhân viên trong file CSV
    file_path = 'Attendance/Attendances.csv'
    if os.path.isfile(file_path):
        updated_rows = []
        
        # Đọc dữ liệu từ file CSV và giữ lại những dòng không khớp với thông tin nhân viên cần xóa
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Lưu tiêu đề
            for row in reader:
                if not (row[0] == userid and row[1] == username and row[2] == email):
                    updated_rows.append(row)
        
        # Ghi lại dữ liệu đã cập nhật vào file CSV (ghi đè)
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # Ghi lại tiêu đề
            writer.writerows(updated_rows)  # Ghi các dòng còn lại

#### extract the face from an image
def extract_faces(img):
    if img is not None and img.size > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        if len(face_points):  # Check if face_points is not empty
            if(is_live_face(img)):
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
    labels = np.array(labels)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    evaluate_model(faces, labels)
    joblib.dump(knn,'static/face_recognition_model.pkl') # Lưu mô hình đã huấn luyện

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    file_path = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.isfile(file_path):
        # Tạo file CSV nếu chưa tồn tại
        with open(file_path, 'w') as f:
            f.write('ID,Name,Check_in_time,Check_out_time,Total_time\n')
        return [], [], [], [], [], 0

    try:
        # Đọc file CSV
        df = pd.read_csv(file_path)

        # Kiểm tra và xử lý các cột nếu thiếu giá trị
        required_columns = ['ID', 'Name', 'Check_in_time', 'Check_out_time', 'Total_time']
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
    useremail = name.split('_')[2]
    current_time = datetime.now().strftime("%H:%M:%S-%d/%m/%Y")
    file_path = f'Attendance/Attendances.csv'

    # Nếu file không tồn tại, tạo file mới với tiêu đề
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Name' , 'Email', "Created at"])

    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([userid, username, useremail, current_time]) 

#Check existed userID
def checkUserID(newuserid):
    listID = os.listdir('static/faces')
    for i in range(0, len(listID)):
        if listID[i].split('_')[1] == newuserid:
            return True
    return False

def handle_checkin_checkout(identified_person):
    file_path = f'Attendance/Attendance-{datetoday}.csv'
    username = identified_person.split('_')[0]
    userid = identified_person.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    # If file doesn't exist, create it
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Name', 'Check_in_time', 'Check_out_time', 'Total_time\n'])

    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(file_path)

    if userid in df['ID'].values:
        idx = df[df['ID'] == userid].index[0]
        if pd.isna(df.at[idx, 'Check_in_time']) or df.at[idx, 'Check_in_time'] == '':
            # Người dùng chưa Check-in, tiến hành Check-in
            df.at[idx, 'Check_in_time'] = current_time
        elif pd.isna(df.at[idx, 'Check_out_time']) or df.at[idx, 'Check_out_time'] == '':
            # Người dùng đã Check-in, tiến hành Check-out
            df.at[idx, 'Check_out_time'] = current_time
            check_in_time = datetime.strptime(df.at[idx, 'Check_in_time'], '%H:%M:%S')
            total_time = (datetime.strptime(current_time, '%H:%M:%S') - check_in_time).seconds // 60
            df.at[idx, 'Total_time'] = f'{total_time} phút'
        else:
            # Người dùng đã Check-in và Check-out
            return "Bạn đã chấm công cho ngày hôm nay"
    else:
        # Thêm người mới vào danh sách Check-in
        new_row = {'ID': userid, 'Name': username, 'Check_in_time': current_time, 
                   'Check_out_time': '', 'Total_time': ''}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(file_path, index=False)
    return "Check-in/Check-out thành công!"

def save_attendance_image(user_name: str, user_id: str, frame, x, y, w, h, action: str):
    folder_path = f'Attendance/Attendance_faces-{datetoday}/{user_name}_{user_id}_{datetoday2}'
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    image_name = f'{user_name}_{user_id}_{action}.jpg'
    if image_name not in os.listdir(folder_path):
        cv2.imwrite(f'{folder_path}/{image_name}', frame[y:y+h, x:x+w])

#Get check in and out time of user
def getUserTime(userid):
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    row_index = 0

    for i in range(0, df['ID'].count()):
        if str(df['ID'][i]) == userid:
            row_index = i
            break
            
    return str(df['Check_in_time'][row_index]), str(df['Check_out_time'][row_index])

def evaluate_model(faces, labels):
    # Load the model
    model = joblib.load('static/face_recognition_model.pkl')
    
    # Perform predictions
    predictions = model.predict(faces)
    
    # Print evaluation metrics
    print("Accuracy:", accuracy_score(labels, predictions))
    print("Classification Report:\n", classification_report(labels, predictions))

def is_live_face(img):
    if img is None or img.size == 0:
        return False

    # Chuyển đổi sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return False  # Không có khuôn mặt nào được phát hiện

    # Khởi tạo Haar Cascade cho mắt
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    for (x, y, w, h) in faces:
        face_region = img[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        # Phát hiện mắt
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        if len(eyes) >= 2:
            return True  # Có ít nhất 2 mắt được phát hiện => Mặt sống

    return False  # Không phát hiện được mắt hoặc không đủ bằng chứng về sống động
