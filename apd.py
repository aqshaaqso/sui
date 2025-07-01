import cv2
from ultralytics import YOLO

model = YOLO("best.pt")  

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Gagal membuka kamera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal menangkap frame")
        break

    # Deteksi objek
    results = model(frame)[0]

    # Gambar hasil deteksi
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = box.conf[0].item()
        label = model.names[cls_id]

        # Ambil koordinat bbox
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Gambar kotak dan label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Deteksi Objek", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
