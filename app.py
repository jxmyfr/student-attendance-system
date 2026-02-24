from flask import Flask, render_template, Response, redirect, url_for, request, jsonify
import base64
import shutil
from flask_sqlalchemy import SQLAlchemy
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
import pickle
import io
import csv
from flask import make_response
from datetime import datetime, date, timedelta
from PIL import ImageFont, ImageDraw, Image

app = Flask(__name__)

# --- CONFIGURATION (ตั้งค่าฐานข้อมูล) ---
# ใช้ SQLite เพราะเป็นไฟล์เดียวจบ ไม่ต้องลงโปรแกรมเพิ่ม
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///school_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ไฟล์เก็บรหัสใบหน้า (ที่สร้างจาก encode_faces.py)
encoding_file = 'encodings.pickle'
font_path = "C:\Windows\Fonts\IrisUPC.ttf" # ปรับตามระบบของคุณ (Windows, Mac, Linux)

# --- DATABASE MODELS (โครงสร้างตารางข้อมูล) ---

class Subject(db.Model):
    id = db.Column(db.String(20), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    teacher = db.Column(db.String(100))
    target_level = db.Column(db.String(10)) # เพิ่มคอลัมน์เก็บระดับชั้น เช่น "ม.1", "ม.4"
    
class SystemSetting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    late_grace_mins = db.Column(db.Integer, default=15) # เก็บเวลาผ่อนผัน (นาที)
    ai_tolerance = db.Column(db.Float, default=0.45)
    
class Student(db.Model):
    id = db.Column(db.String(20), primary_key=True)
    roll_number = db.Column(db.Integer)
    name_th = db.Column(db.String(100), nullable=False)
    name_en = db.Column(db.String(100))
    classroom = db.Column(db.String(50))
    # สร้างความสัมพันธ์กับตาราง Attendance

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), db.ForeignKey('student.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    subject = db.Column(db.String(100))  # <--- เพิ่มบรรทัดนี้เพื่อเก็บชื่อวิชา

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

def put_thai_text(img, text, position, font_size=30, color=(255, 255, 255)):
    """วาดภาษาไทยบนภาพ (อัปเกรดระบบค้นหาฟอนต์อัตโนมัติ)"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # ลิสต์รายชื่อฟอนต์ภาษาไทยที่มักจะมีใน Windows และในโปรเจค
    font_paths = [
        "font/Sarabun-Regular.ttf",      # หาในโฟลเดอร์โปรเจคก่อน
        "C:/Windows/Fonts/tahoma.ttf",   # ฟอนต์ Tahoma (Windows ทั่วไป)
        "C:/Windows/Fonts/LeelawUI.ttf", # ฟอนต์ Leelawadee UI (Windows 10/11)
        "C:/Windows/Fonts/angsana.ttc",  # ฟอนต์ Angsana
        "C:/Windows/Fonts/cordiau.ttf"   # ฟอนต์ Cordia
    ]

    font = None
    # วนลูปหาฟอนต์ ถ้าเจอตัวไหนก่อน ให้ใช้ตัวนั้นเลย
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                break # เจอและโหลดสำเร็จแล้ว ให้ออกจากลูป
            except IOError:
                continue

    # ถ้าหาไม่เจอเลยทุกไฟล์ ให้ใช้ Default (ซึ่งจะเป็นสี่เหลี่ยม)
    if font is None:
        print("Warning: ไม่พบไฟล์ฟอนต์ภาษาไทยเลยในเครื่อง! ข้อความจะเป็นสี่เหลี่ยม")
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

@app.route('/scan')
def index():
    """หน้าจอระบบสแกนใบหน้า (เปลี่ยนมาใช้ /scan แทน)"""
    return render_template('index.html')

import json
@app.route('/')
@app.route('/dashboard')
def dashboard():
    """หน้า Dashboard สรุปผลสถิติแบบ Drill-down"""
    students = Student.query.all()
    attendances = Attendance.query.all()

    # 1. นับสถิติการเข้าเรียนของนักเรียนแต่ละคน
    student_stats = {}
    for s in students:
        student_stats[s.id] = {
            'id': s.id,
            'name': s.name_th,
            'classroom': s.classroom or 'ไม่ระบุ',
            'roll': s.roll_number or 0,
            'total_classes': 0,
            'present': 0,
            'late': 0
        }

    for a in attendances:
        if a.student_id in student_stats:
            student_stats[a.student_id]['total_classes'] += 1
            if a.status == 'Present':
                student_stats[a.student_id]['present'] += 1
            elif a.status == 'Late':
                student_stats[a.student_id]['late'] += 1

    # 2. คำนวณเปอร์เซ็นต์การเข้าเรียนให้แต่ละคน
    student_list = []
    for uid, data in student_stats.items():
        if data['total_classes'] > 0:
            # สมมติว่ามาเรียนและมาสาย ถือว่าเข้าเรียน (คุณปรับแก้ลอจิกตรงนี้ได้)
            attendance_rate = ((data['present'] + data['late']) / data['total_classes']) * 100
        else:
            attendance_rate = 0.0 # ยังไม่มีประวัติ
            
        data['attendance_rate'] = round(attendance_rate, 2)
        student_list.append(data)

    # ส่งข้อมูลไปหน้าเว็บในรูปแบบ JSON text
    return render_template('dashboard.html', students_json=json.dumps(student_list))

@app.route('/update_db_faces')
def update_db_faces():
    """ปุ่มกดอัปเดต Face DB"""
    from encode_faces import create_encodings
    create_encodings() # เรียกฟังก์ชันรันใหม่
    load_encodings()   # โหลดเข้า RAM
    return redirect(url_for('index'))

@app.route('/reports')
def reports():
    """หน้าเว็บแสดงรายงานทั้งหมด"""
    # ดึงข้อมูลทั้งหมดจาก Database เรียงจากวันที่และเวลาล่าสุดขึ้นก่อน
    records = db.session.query(Attendance, Student).\
        join(Student, Attendance.student_id == Student.id).\
        order_by(Attendance.date.desc(), Attendance.time.desc()).all()
    
    return render_template('reports.html', records=records)

@app.route('/export_csv')
def export_csv():
    """ฟังก์ชันสำหรับดาวน์โหลดไฟล์ Excel (CSV)"""
    records = db.session.query(Attendance, Student).\
        join(Student, Attendance.student_id == Student.id).\
        order_by(Attendance.date.desc(), Attendance.time.desc()).all()
    
    # สร้างไฟล์ CSV ในหน่วยความจำ
    si = io.StringIO()
    cw = csv.writer(si)
    
    # เขียนหัวตาราง (Headers)
    cw.writerow(['วันที่ (Date)', 'เวลา (Time)', 'รหัสนักเรียน (ID)', 'ชื่อ-นามสกุล (Name)', 'ชั้นเรียน (Class)', 'สถานะ (Status)'])
    
    # วนลูปเขียนข้อมูลทีละบรรทัด
    for att, std in records:
        # แปลง status ภาษาอังกฤษ เป็นไทยเพื่อให้ครูอ่านง่าย
        status_th = "มาเรียน" if att.status == 'Present' else "มาสาย" if att.status == 'Late' else att.status
        
        cw.writerow([
            att.date.strftime('%d/%m/%Y'),
            att.time.strftime('%H:%M:%S'),
            std.id,
            std.name_th,
            std.classroom,
            status_th
        ])
        
    # ส่งไฟล์กลับไปให้เบราว์เซอร์ดาวน์โหลด
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=attendance_report.csv"
    # ใช้ utf-8-sig เพื่อให้โปรแกรม Excel เปิดแล้วภาษาไทยไม่เพี้ยน
    output.headers["Content-type"] = "text/csv; charset=utf-8-sig" 
    return output

@app.route('/students')
def students():
    """หน้าเว็บแสดงรายชื่อนักเรียนทั้งหมด (เรียงตามห้องและเลขที่)"""
    all_students = Student.query.order_by(Student.classroom, Student.roll_number).all()
    return render_template('students.html', students=all_students)

@app.route('/delete_student/<student_id>')
def delete_student_route(student_id):
    """ระบบลบนักเรียนออกแบบเบ็ดเสร็จ"""
    # 1. ลบประวัติเข้าเรียนและข้อมูลส่วนตัวออกจาก Database
    Attendance.query.filter_by(student_id=student_id).delete()
    student = Student.query.get(student_id)
    if student:
        db.session.delete(student)
        db.session.commit()
        
    # 2. ลบโฟลเดอร์รูปภาพในเครื่อง
    folder_path = os.path.join('images_db', str(student_id))
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path) # ลบโฟลเดอร์และไฟล์ข้างในทิ้งทั้งหมด
        
    # 3. อัปเดตไฟล์ AI (แปลงใบหน้าใหม่)
    from encode_faces import create_encodings
    create_encodings()
    
    # เสร็จแล้วให้รีเฟรชกลับมาหน้าเดิม
    return redirect(url_for('students'))

@app.route('/register_student', methods=['GET', 'POST'])
def register_student():
    """ระบบลงทะเบียนนักเรียนและถ่ายรูปผ่านเว็บ"""
    if request.method == 'GET':
        return render_template('add_student.html')

    if request.method == 'POST':
        # รับข้อมูล JSON จากหน้าเว็บ (Javascript)
        data = request.json
        student_id = data.get('student_id')
        name_th = data.get('name_th')
        name_en = data.get('name_en')
        classroom = data.get('classroom')
        images = data.get('images') # รายการรูปภาพแบบ Base64 (5 รูป)

        if not student_id or not images:
            return jsonify({"status": "error", "message": "ข้อมูลไม่ครบถ้วน"})

        # 1. บันทึกประวัติลง Database SQLite
        existing_student = Student.query.get(student_id)
        if not existing_student:
            new_student = Student(id=student_id, name_th=name_th, name_en=name_en, classroom=classroom)
            db.session.add(new_student)
            db.session.commit()

        # 2. สร้างโฟลเดอร์เก็บรูปภาพ
        folder_path = os.path.join('images_db', str(student_id))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 3. แปลงไฟล์รูปภาพ Base64 เป็นไฟล์ .jpg เซฟลงเครื่อง
        for i, img_data in enumerate(images):
            # ตัดส่วนหัว (Header) ของ Base64 ออกก่อน
            img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)
            file_path = os.path.join(folder_path, f"{student_id}_{i+1}.jpg")
            with open(file_path, "wb") as fh:
                fh.write(img_bytes)

        # 4. เรียกฟังก์ชัน AI ให้เรียนรู้ใบหน้าใหม่ทันที
        from encode_faces import create_encodings
        create_encodings()

        return jsonify({"status": "success", "message": "ลงทะเบียนและอัปเดต AI สำเร็จ!"})

@app.route('/api/classes')
def get_classes():
    """API สำหรับดึงรายชื่อชั้นเรียนทั้งหมดแบบไม่ซ้ำกัน"""
    # ค้นหา classroom ทั้งหมดที่พบล่าสุดและตัดตัวซ้ำออก
    classes = db.session.query(Student.classroom).filter(Student.classroom != None).distinct().all()
    class_list = [c[0] for c in classes]
    return jsonify(class_list)

@app.route('/api/students/<path:class_name>')   # <--- เติม path: ตรงนี้ครับ
def get_students_by_class(class_name):
    """API สำหรับดึงรายชื่อนักเรียนตามชั้นเรียนที่เลือก"""
    # อัปเกรดให้เรียงตามเลขที่ (roll_number) ตามที่เราเพิ่งทำไปเลยครับ
    students = Student.query.filter_by(classroom=class_name).order_by(Student.roll_number, Student.id).all()
    student_list = [{"id": s.id, "name_th": s.name_th} for s in students]
    return jsonify(student_list)

@app.route('/import_students', methods=['POST'])
def import_students():
    """API สำหรับรับไฟล์ Excel และบันทึกรายชื่อลง Database"""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "ไม่พบไฟล์ที่อัปโหลด"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "ไม่ได้เลือกไฟล์"})
        
    if file and file.filename.endswith('.xlsx'):
        try:
            # ใช้ pandas อ่านไฟล์ Excel
            df = pd.read_excel(file)
            
            # ตรวจสอบว่ามีคอลัมน์ที่ต้องการครบไหม (เพิ่ม Roll_Number)
            required_columns = ['ID', 'Roll_Number', 'Name_TH', 'Classroom']
            for col in required_columns:
                if col not in df.columns:
                    return jsonify({"status": "error", "message": f"ไฟล์ Excel ต้องมีคอลัมน์ชื่อ {col}"})

            success_count = 0
            for index, row in df.iterrows():
                student_id = str(row['ID']).strip()
                roll_number = int(row['Roll_Number']) if pd.notna(row['Roll_Number']) else None # <--- ดึงเลขที่
                name_th = str(row['Name_TH']).strip()
                name_en = str(row.get('Name_EN', '')).strip()
                classroom = str(row['Classroom']).strip()

                existing_student = Student.query.get(student_id)
                if not existing_student:
                    # เพิ่มตัวแปร roll_number เข้าไปตอนสร้างใหม่
                    new_student = Student(id=student_id, roll_number=roll_number, name_th=name_th, name_en=name_en, classroom=classroom)
                    db.session.add(new_student)
                    success_count += 1
                else:
                    # อัปเดตข้อมูลให้เป็นปัจจุบัน
                    existing_student.roll_number = roll_number
                    existing_student.name_th = name_th
                    existing_student.name_en = name_en
                    existing_student.classroom = classroom
                    success_count += 1
            
            db.session.commit()
            return jsonify({"status": "success", "message": f"นำเข้า/อัปเดตข้อมูลสำเร็จ {success_count} รายการ!"})
        except Exception as e:
            return jsonify({"status": "error", "message": f"เกิดข้อผิดพลาดในการอ่านไฟล์: {str(e)}"})
    else:
        return jsonify({"status": "error", "message": "ระบบรองรับเฉพาะไฟล์นามสกุล .xlsx เท่านั้น"})

import numpy as np
from datetime import datetime

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    """API รับรูปภาพ 1 รูปจากหน้าเว็บ มาให้ AI ทายชื่อ"""
    data = request.json
    image_data = data.get('image')
    
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    if not face_encodings:
        return jsonify({'status': 'not_found', 'message': 'ไม่พบใบหน้าในรูปภาพ'})
        
    # ดึงค่า AI จากการตั้งค่า
    setting = SystemSetting.query.first()
    tolerance_val = setting.ai_tolerance if setting else 0.45
    
    encodeFace = face_encodings[0]
    # ใช้ตัวแปร tolerance_val ตรงนี้
    matches = face_recognition.compare_faces(known_face_encodings, encodeFace, tolerance=tolerance_val)
        
    # แปลงไฟล์ภาพ Base64 ให้เป็นภาพที่ OpenCV และ Face_recognition อ่านได้
    img_data = image_data.split(',')[1]
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ให้ AI หาใบหน้าและสร้างชุดตัวเลข
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    
    if not face_encodings:
        return jsonify({'status': 'not_found', 'message': 'ไม่พบใบหน้าในรูปภาพ กรุณาถ่ายใหม่'})
        
    # เอาหน้าแรกที่เจอไปเทียบกับฐานข้อมูล
    encodeFace = face_encodings[0]
    matches = face_recognition.compare_faces(known_face_encodings, encodeFace, tolerance=0.45)
    faceDis = face_recognition.face_distance(known_face_encodings, encodeFace)
    
    if len(faceDis) > 0:
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            student_id = known_face_names[matchIndex]
            student = Student.query.get(student_id)
            if student:
                return jsonify({
                    'status': 'success',
                    'student_id': student.id,
                    'name_th': student.name_th,
                    'classroom': student.classroom,
                    'roll_number': student.roll_number
                })
                
    return jsonify({'status': 'unknown', 'message': 'ไม่รู้จักใบหน้านี้ (Unknown)'})

@app.route('/save_attendance', methods=['POST'])
def save_attendance():
    """API สำหรับบันทึกข้อมูลและคำนวณเวลาสายตามรายวิชา"""
    data = request.json
    student_id = data.get('student_id')
    subject = data.get('subject')
    start_time_str = data.get('start_time') # เวลาเริ่มคาบที่ครูกรอก เช่น '10:10'
    
    now = datetime.now()
    existing = Attendance.query.filter_by(student_id=student_id, date=now.date(), subject=subject).first()
    if existing:
        return jsonify({'status': 'warning', 'message': f'เช็กชื่อวิชา {subject} ไปแล้ว!'})

    # 1. ดึงค่าเวลาผ่อนผัน (เช่น 15 นาที)
    setting = SystemSetting.query.first()
    grace_mins = setting.late_grace_mins if setting else 15
    
    # 2. คำนวณเวลาเส้นตาย (เวลาเริ่มคาบ + เวลาผ่อนผัน)
    start_hour, start_min = map(int, start_time_str.split(':'))
    class_start_time = now.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
    late_limit_time = class_start_time + timedelta(minutes=grace_mins)
    
    # 3. ตัดสินว่าสายหรือไม่ (ถ้าเวลาปัจจุบัน เลยเวลาเส้นตายไปแล้ว = สาย)
    status = 'Late' if now > late_limit_time else 'Present'

    new_record = Attendance(student_id=student_id, date=now.date(), time=now.time(), status=status, subject=subject)
    db.session.add(new_record)
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': f'บันทึกสำเร็จ (สถานะ: {"มาสาย" if status == "Late" else "มาเรียนตรงเวลา"})'})

@app.route('/api/subjects')
def get_subjects():
    """API สำหรับดึงรายชื่อวิชาทั้งหมดไปใส่ใน Dropdown"""
    subjects = Subject.query.all()
    # ส่งข้อมูลกลับไปเป็น JSON (เช่น รหัสวิชา - ชื่อวิชา)
    return jsonify([{
        "id": s.id, 
        "name": s.name, 
        "teacher": s.teacher,
        "level": s.target_level # ส่งระดับชั้นกลับไปให้หน้าเว็บ
    } for s in subjects])

@app.route('/subjects', methods=['GET', 'POST'])
def manage_subjects():
    """หน้าเว็บจัดการข้อมูลรายวิชา (เพิ่ม/ลบ)"""
    if request.method == 'POST':
        sub_id = request.form.get('sub_id')
        sub_name = request.form.get('sub_name')
        sub_teacher = request.form.get('sub_teacher')
        # 1. เพิ่มบรรทัดรับค่าระดับชั้น (sub_level) จากฟอร์มหน้าเว็บ
        sub_level = request.form.get('sub_level') 
        
        if sub_id and sub_name:
            # 2. เพิ่มตัวแปร target_level=sub_level เข้าไปตอนสร้างวิชาใหม่
            new_sub = Subject(
                id=sub_id, 
                name=sub_name, 
                teacher=sub_teacher, 
                target_level=sub_level # <--- สำคัญมาก ต้องส่งค่านี้เข้าฐานข้อมูล
            )
            db.session.add(new_sub)
            db.session.commit()
            return redirect(url_for('manage_subjects'))

    all_subjects = Subject.query.all()
    return render_template('subjects.html', subjects=all_subjects) 
    # (หมายเหตุ: บรรทัดบน ตรวจสอบดูนะครับถ้าในโค้ดเดิมคุณใช้ subjects=all_subjects ให้ใช้ชื่อเดิมครับ)

@app.route('/delete_subject/<subject_id>')
def delete_subject(subject_id):
    """ฟังก์ชันลบรายวิชา"""
    sub = Subject.query.get(subject_id)
    if sub:
        db.session.delete(sub)
        db.session.commit()
    return redirect(url_for('manage_subjects'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """หน้าตั้งค่าระบบ"""
    setting = SystemSetting.query.first()
    if not setting:
        setting = SystemSetting(late_grace_mins=15, ai_tolerance=0.45)
        db.session.add(setting)
        db.session.commit()

    if request.method == 'POST':
        setting.late_grace_mins = int(request.form.get('late_grace_mins'))
        setting.ai_tolerance = float(request.form.get('ai_tolerance'))
        db.session.commit()
        return redirect(url_for('settings'))

    return render_template('settings.html', setting=setting)

@app.route('/attendance_management', methods=['GET', 'POST'])
def attendance_management():
    """หน้าจัดการสถานะการเข้าเรียนและลงชื่อเด็กที่ขาด/ลา"""
    today = datetime.now().date()
    # ดึงรายชื่อห้องทั้งหมดมาให้เลือก
    classes = db.session.query(Student.classroom).distinct().all()
    class_list = [c[0] for c in classes]
    
    selected_class = request.args.get('class_name')
    records = []
    missing_students = []

    if selected_class:
        # 1. รายชื่อคนที่เช็กชื่อแล้ว
        records = Attendance.query.filter_by(date=today).join(Student).filter(Student.classroom == selected_class).all()
        
        # 2. ค้นหาคนที่ "ยังไม่ได้เช็กชื่อ" ในวันนี้
        checked_ids = [r.student_id for r in records]
        missing_students = Student.query.filter(Student.classroom == selected_class, Student.id.not_in(checked_ids)).all()

    return render_template('attendance_data.html', 
                           class_list=class_list, 
                           records=records, 
                           missing_students=missing_students,
                           selected_class=selected_class)
    
@app.route('/add_missing_status', methods=['POST'])
def add_missing_status():
    """บันทึกสถานะ ขาด/ลา สำหรับคนที่ไม่ได้สแกนหน้า"""
    data = request.json
    now = datetime.now()
    # สร้าง Record ใหม่สำหรับสถานะ ขาด หรือ ลา
    new_record = Attendance(
        student_id=data['student_id'],
        date=now.date(),
        time=now.time(),
        status=data['status'], # Sick Leave, Personal Leave, Absent
        subject="รายวัน/ไม่ระบุ"
    )
    db.session.add(new_record)
    db.session.commit()
    return jsonify({"status": "success"})

@app.route('/update_status/<int:record_id>', methods=['POST'])
def update_status(record_id):
    """API สำหรับเปลี่ยนสถานะเป็น ลาป่วย/ลากิจ/ขาดเรียน"""
    new_status = request.json.get('status')
    record = Attendance.query.get(record_id)
    if record:
        record.status = new_status
        db.session.commit()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 404

# --- MAIN ---
if __name__ == "__main__":
    # host='0.0.0.0' คือคำสั่งเปิดให้มือถือเครื่องอื่นใน WiFi เดียวกันเข้าได้
    app.run(host='0.0.0.0', port=5000, debug=True)