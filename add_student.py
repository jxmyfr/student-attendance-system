import cv2
import os
from encode_faces import create_encodings

# [เพิ่ม] นำเข้า Database จากไฟล์ app.py
from app import app, db, Student 

base_dir = 'images_db'

def create_student_dataset():
    print("--- ระบบลงทะเบียนนักเรียนใหม่ ---")
    student_id = input("1. กรุณากรอกรหัสนักเรียน (ID): ").strip()
    if not student_id:
        print("Error: คุณไม่ได้กรอกรหัส!")
        return
        
    # [เพิ่ม] ถามข้อมูลส่วนตัวเพื่อบันทึกลงฐานข้อมูล SQLite
    name_th = input("2. กรุณากรอกชื่อ-นามสกุล (ภาษาไทย): ").strip()
    name_en = input("3. กรุณากรอกชื่อ-นามสกุล (ภาษาอังกฤษ): ").strip()
    classroom = input("4. กรุณากรอกชั้นเรียน (เช่น ม.5/1): ").strip()

    # บันทึกข้อมูลลง Database
    with app.app_context():
        # เช็กว่ามีรหัสนี้ในระบบหรือยัง
        existing_student = Student.query.get(student_id)
        if not existing_student:
            new_student = Student(id=student_id, name_th=name_th, name_en=name_en, classroom=classroom)
            db.session.add(new_student)
            db.session.commit()
            print(f"✅ บันทึกข้อมูล {name_th} ลงฐานข้อมูลสำเร็จ")
        else:
            print(f"ℹ️ รหัส {student_id} มีในฐานข้อมูลแล้ว จะทำการเพิ่มรูปภาพอย่างเดียว")

    # --- ส่วนของการถ่ายรูปเหมือนเดิม ---
    student_path = os.path.join(base_dir, student_id)
    if not os.path.exists(student_path):
        os.makedirs(student_path)
        print(f"สร้างโฟลเดอร์: {student_path}")

    cap = cv2.VideoCapture(0)
    count = 1
    existing_files = os.listdir(student_path)
    if existing_files:
        count = len(existing_files) + 1

    print("\n[กล้องเปิดแล้ว] จัดท่าทางให้เรียบร้อย (หน้าตรง, เอียงซ้าย-ขวาเล็กน้อย)")
    print("กด 's' เพื่อถ่ายรูป (แนะนำ 3-5 รูป) | กด 'q' เพื่อออก")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        cv2.putText(display_frame, f"ID: {student_id} | Pic: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Face Enrollment', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            file_path = os.path.join(student_path, f"{student_id}_{count}.jpg")
            cv2.imwrite(file_path, frame)
            print(f"📸 บันทึกรูปที่ {count} แล้ว")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print("\n🔄 กำลังแปลงโครงสร้างใบหน้า (Encoding) และอัปเดตระบบ...")
    create_encodings()
    print("🎉 ลงทะเบียนเสร็จสมบูรณ์!")

if __name__ == "__main__":
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    create_student_dataset()