import cv2
import face_recognition
import pickle
import os

# --- ตั้งค่า ---
dataset_path = 'images_db'      
encoding_file = 'encodings.pickle' 

def create_encodings():
    print("Step 1: Walking through sub-directories...")
    
    knownEncodings = []
    knownNames = []
    
    # เดินลูปเข้าไปในทุกโฟลเดอร์ (os.walk)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                imagePath = os.path.join(root, file)
                
                # *** หัวใจสำคัญ: ดึง ID จากชื่อโฟลเดอร์แม่ ***
                # root จะเป็น path/to/64080502281 -> เราเอาแค่ตัวท้ายสุด
                student_id = os.path.basename(root) 
                
                print(f"Processing: {student_id} -> {file}")

                image = cv2.imread(imagePath)
                if image is None:
                    continue
                    
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # ใช้ model='hog' เพื่อความเร็ว (ถ้ามี GPU แรงๆ เปลี่ยนเป็น 'cnn' ได้)
                boxes = face_recognition.face_locations(rgb, model='hog')
                encodings = face_recognition.face_encodings(rgb, boxes)

                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(student_id) # เก็บ ID เป็นชื่อ (เช่น 64080502281)

    print("Step 2: Saving data to pickle file...")
    data = {"encodings": knownEncodings, "names": knownNames}
    
    with open(encoding_file, "wb") as f:
        f.write(pickle.dumps(data))
        
    print(f"Success! Processed faces for {len(set(knownNames))} students.")

if __name__ == "__main__":
    create_encodings()