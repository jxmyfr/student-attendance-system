import cv2
import face_recognition
import pickle
import os

# --- ตั้งค่าเส้นทางไฟล์ ---
dataset_path = 'images_db'      
encoding_file = 'encodings.pickle' 

def create_encodings():
    # เปลี่ยนเป็นภาษาอังกฤษเพื่อป้องกัน UnicodeEncodeError ใน Terminal
    print("--- Start Face Encoding Process ---")
    
    if os.path.exists(encoding_file):
        os.remove(encoding_file)
        print(f"Removed old {encoding_file}")

    knownEncodings = []
    knownNames = []
    
    # ตรวจสอบว่ามีโฟลเดอร์ images_db หรือไม่
    if not os.path.exists(dataset_path):
        print(f"Error: dataset_path '{dataset_path}' not found")
        return

    # 2. วนลูปอ่านข้อมูลใน images_db
    for root, dirs, files in os.walk(dataset_path):
        # ข้ามโฟลเดอร์หลัก images_db
        if root == dataset_path:
            continue
            
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                imagePath = os.path.join(root, file)
                
                # ดึง ID นักเรียนจากชื่อโฟลเดอร์ย่อย
                student_id = os.path.basename(root) 
                
                # เปลี่ยนเป็นภาษาอังกฤษ
                print(f"Processing: {student_id} -> {file}")

                # อ่านรูปภาพ
                image = cv2.imread(imagePath)
                if image is None:
                    continue
                    
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # ตรวจหาตำแหน่งใบหน้าและสร้างรหัสใบหน้า
                boxes = face_recognition.face_locations(rgb, model='hog')
                encodings = face_recognition.face_encodings(rgb, boxes)

                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(student_id)

    # 3. บันทึกข้อมูลลงไฟล์ Pickle
    print("--- Saving data to Pickle file ---")
    data = {"encodings": knownEncodings, "names": knownNames}
    
    with open(encoding_file, "wb") as f:
        f.write(pickle.dumps(data))
        
    print(f"Success! Processed {len(set(knownNames))} students.")

if __name__ == "__main__":
    create_encodings()