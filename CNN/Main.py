import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import time
import numpy as np
import face_recognition
import os
import psutil
import GPUtil

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  

camera = cv2.VideoCapture(0)

cameraWidth = 1280
cameraHeight = 720

if not camera.isOpened():
    print("Kamera tidak dapat diakses.")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, cameraWidth)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cameraHeight)
camera.set(cv2.CAP_PROP_FPS, 30)

known_face_encodings = []
known_face_names = []

dataset_path = "Dataset"
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(dataset_path, filename)
        face_image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(face_image)
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append("Axel")

id_tags = {}
deteksi_terakhir = {}
waktu_hilang = {}
waktu_cek_face_recognition = {}
waktu = 0  
last_update_time = time.time()
highest_confidence = 0

timeout_duration = 0
face_recognition_interval = 5

frame_count = 0
fps = 0
last_time = time.time() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    frame_tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
    frame_tensor = frame_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(frame_tensor)

    current_ids = []
    highest_confidence = 0

    for element in range(len(prediction[0]['boxes'])):
        box = prediction[0]['boxes'][element].cpu().numpy()  
        score = prediction[0]['scores'][element].item() 
        label = prediction[0]['labels'][element].item() 

        if score > 0.5:  
            x1, y1, x2, y2 = box
            if label == 1: 
                highest_confidence = max(highest_confidence, score)

                person_id = len(current_ids) + 1
                current_ids.append(person_id)

                if person_id in id_tags:
                    name = id_tags[person_id]
                else:
                    name = "Tidak dikenal"
                    current_time = time.time()

                    if person_id not in waktu_cek_face_recognition or (current_time - waktu_cek_face_recognition[person_id] > face_recognition_interval):
                        face_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                        rgb_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

                        face_encodings = face_recognition.face_encodings(rgb_face_frame)

                        if face_encodings:
                            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
                            best_match_index = np.argmin(face_distances)  

                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]
                                accuracy = (1 - face_distances[best_match_index]) * 100
                                id_tags[person_id] = name
                            else:
                                id_tags[person_id] = name

                        waktu_cek_face_recognition[person_id] = current_time

                if name == "Axel":
                    current_time = time.time()
                    waktu += current_time - last_update_time

                last_update_time = time.time()

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {person_id} | {name} | {accuracy:.2f}%",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                deteksi_terakhir[person_id] = time.time()

    current_time = time.time()
    for person_id in list(id_tags.keys()):
        if person_id not in current_ids:
            if current_time - deteksi_terakhir.get(person_id, 0) > timeout_duration:
                print(f"ID {person_id} telah hilang, tag akan dihapus.")
                del id_tags[person_id]
    
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()

    gpus = GPUtil.getGPUs()
    gpu_info = gpus[0] if gpus else None 
    if gpu_info:
        gpu_name = gpu_info.name
        gpu_load = gpu_info.load * 100  
        gpu_memory_used = gpu_info.memoryUsed 
        gpu_memory_total = gpu_info.memoryTotal  

    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - last_time

    if elapsed_time >= 1.0:  
        fps = frame_count / elapsed_time 
        frame_count = 0 
        last_time = current_time  

    cv2.putText(frame, f"GPU: {gpu_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.putText(frame, f"GPU Load: {gpu_load:.2f}%",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, f"Memory: {gpu_memory_used:.0f}/{gpu_memory_total:.0f} MB",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.putText(frame, f"Axel's Precemse: {int(waktu)} detik",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Faster R-CNN Accuration: {highest_confidence * 100:.2f}%",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"CPU Usage: {cpu_usage}%",
                (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.putText(frame, f"RAM Usage: {memory_info.percent}%",
                (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Presensi Mahasiswa", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
