from flask import Flask, render_template, Response, redirect, url_for, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime, date
from PIL import ImageFont, ImageDraw, Image

app = Flask(__name__)

# --- CONFIGURATION (ตั้งค่าฐานข้อมูล) ---
# ใช้ SQLite เพราะเป็นไฟล์เดียวจบ ไม่ต้องลงโปรแกรมเพิ่ม
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///school_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ไฟล์เก็บรหัสใบหน้า (ที่สร้างจาก encode_faces.py)
encoding_file = 'encodings.pickle'
font_path = "font/Sarabun-Regular.ttf"

# --- DATABASE MODELS (โครงสร้างตารางข้อมูล) ---
class Student(db.Model):
    id = db.Column(db.String(20), primary_key=True)
    name_th = db.Column(db.String(100), nullable=False)
    name_en = db.Column(db.String(100), nullable=False)
    classroom = db.Column(db.String(20), nullable=False) # เช่น ม.5/1
    # สร้างความสัมพันธ์กับตาราง Attendance

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), db.ForeignKey('student.id'), nullable=False)
    date = db.Column(db.Date, default=date.today)
    time = db.Column(db.Time, default=datetime.now().time)
    status = db.Column(db.String(20), default='Present') # Present, Late, Absent
    subject_id = db.Column(db.String(20)) # รหัสวิชา (เผื่ออนาคต)

# สร้าง Database อัตโนมัติถ้ายังไม่มี
with app.app_context():
    db.create_all()

# --- GLOBAL VARIABLES ---
known_face_encodings = []
known_face_names = []

# --- HELPER FUNCTIONS ---

def load_encodings():
    """โหลดข้อมูล Encodings จากไฟล์ Pickle"""
    global known_face_encodings, known_face_names
    print("--- Loading Face Data ---")
    if os.path.exists(encoding_file):
        data = pickle.loads(open(encoding_file, "rb").read())
        known_face_encodings = data["encodings"]
        known_face_names = data["names"]
        print(f"Loaded {len(known_face_names)} faces.")
    else:
        print("Warning: Pickle file not found.")
        known_face_encodings, known_face_names = [], []

def put_thai_text(img, text, position, font_size=30, color=(0, 255, 0)):
    """วาดภาษาไทยบนภาพ"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def mark_attendance_db(student_id):
    """บันทึกเวลาเรียนลง Database (SQLite)"""
    with app.app_context():
        # 1. เช็กก่อนว่าวันนี้นักเรียนคนนี้เช็กไปหรือยัง
        today = date.today()
        existing = Attendance.query.filter_by(student_id=student_id, date=today).first()
        
        if existing:
            return # เช็กไปแล้ว ไม่ทำซ้ำ
            
        # 2. ถ้ายัง ให้บันทึกใหม่
        # ตรวจสอบเวลาสาย (ตัวอย่าง: สายหลัง 08:30)
        now = datetime.now().time()
        late_cutoff = datetime.strptime("08:30:00", "%H:%M:%S").time()
        status = 'Late' if now > late_cutoff else 'Present'
        
        new_record = Attendance(student_id=student_id, time=now, status=status)
        db.session.add(new_record)
        db.session.commit()
        
        # ดึงชื่อมาแสดงใน Console (ภาษาอังกฤษ)
        student = Student.query.get(student_id)
        name_display = student.name_en if student else student_id
        print(f"Recorded: {name_display} - {status} at {now.strftime('%H:%M:%S')}")

load_encodings()

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """หน้า Dashboard ดูสถิติ (ตัวอย่างง่ายๆ)"""
    # ดึงข้อมูลการเข้าเรียนวันนี้
    records = db.session.query(Attendance, Student).\
        join(Student, Attendance.student_id == Student.id).\
        filter(Attendance.date == date.today()).all()
    
    return render_template('dashboard.html', records=records) 
    # *คุณต้องสร้างไฟล์ dashboard.html เพิ่มใน templates*

@app.route('/update_db_faces')
def update_db_faces():
    """ปุ่มกดอัปเดต Face DB"""
    from encode_faces import create_encodings
    create_encodings() # เรียกฟังก์ชันรันใหม่
    load_encodings()   # โหลดเข้า RAM
    return redirect(url_for('index'))

# --- VIDEO PROCESSING ---

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # ย่อภาพเพื่อความเร็ว (สำคัญมากสำหรับมือถือ)
        scale = 0.25
        imgS = cv2.resize(frame, (0, 0), None, scale, scale)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # -----------------------------------------------------------
        # จุดตัดสินใจ: เลือกใช้โมเดล HOG หรือ CNN ตรงนี้
        # model='hog' -> เร็ว, ลื่น, เหมาะกับ Video Stream (แนะนำ)
        # model='cnn' -> ช้า, แม่นยำสูง, ต้องมี GPU (ถ้าไม่มีจะกระตุกมาก)
        # -----------------------------------------------------------
        facesCurFrame = face_recognition.face_locations(imgS, model='hog')
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(known_face_encodings, encodeFace, tolerance=0.5)
            faceDis = face_recognition.face_distance(known_face_encodings, encodeFace)
            
            matchIndex = np.argmin(faceDis)
            
            name_text = "Unknown"
            info_text = ""
            box_color = (0, 0, 255)

            if matches[matchIndex]:
                student_id = known_face_names[matchIndex]
                
                # ดึงข้อมูลจาก Database จริงๆ
                with app.app_context():
                    student = Student.query.get(student_id)
                    if student:
                        name_text = student.name_th
                        info_text = f"ห้อง: {student.classroom}"
                    else:
                        name_text = f"ID: {student_id}" # มีรูปแต่ไม่มีชื่อใน DB

                box_color = (0, 255, 0)
                mark_attendance_db(student_id)

            # วาดกราฟิก
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = int(y1/scale), int(x2/scale), int(y2/scale), int(x1/scale)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(frame, (x1, y2 - 60), (x2, y2), box_color, cv2.FILLED)
            
            frame = put_thai_text(frame, name_text, (x1 + 6, y2 - 55), 30, (255, 255, 255))
            if info_text:
                frame = put_thai_text(frame, info_text, (x1 + 6, y2 - 25), 20, (200, 200, 200))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- MAIN ---
if __name__ == "__main__":
    # host='0.0.0.0' คือคำสั่งเปิดให้มือถือเครื่องอื่นใน WiFi เดียวกันเข้าได้
    app.run(host='0.0.0.0', port=5000, debug=True)