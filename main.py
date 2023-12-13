from ultralytics import YOLO
from modules.boundingbox_module import gambar_boundingbox_jersey, gambar_boundingbox_bola, deteksi_player_ballpossession, gambar_segitiga_pemain, keterangan_ballpossession, hitung_total_ballpossession
from modules.jersey_module import klasifikasi_warnajersey
from modules.hardwarecheck import cudagpu

import cv2
import os
# import time

cudagpu()

# Load model
model = YOLO('weights/football-scouting-best-m-1695-aug-segonlyplayer.pt')
# results = model('sample_2.jpg', show=True, save=True) # Contoh prediksi pada single image

# Membaca file video.mp4
VIDEO_PATH = "input/2f54ed1c_0.mp4" # Source video dalam lokal
cap = cv2.VideoCapture(VIDEO_PATH) # 0 = webcam, 1 = external webcam, VIDEO_PATH = lokasi video lokal

# Membaca detail frame video
file_base = os.path.basename(VIDEO_PATH) # Get nama direktori file dan nama file
file_name = os.path.splitext(file_base) # Get nama file saja 
frame_jumlah = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Get total frame dalam file video masukan
fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS (Frame Per Second) dalam file video masukan
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # Get ukuran width dari frame video masukan
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # Get ukuran height dari frame video masukan

# Menampilkan informasi video
print("============= DETAIL FILE VIDEO =============")
print(f"Nama File Video :", file_name, 
      "\nTotal Frame :", int(frame_jumlah), 
      "\nFPS :", int(fps), 
      "\nDurasi Video (Detik) :", frame_jumlah/fps,
      "\nOriginal Ukuran Frame :", int(frame_width), int(frame_height),
      "\nModel Label/Kelas :", model.names),
print("=============================================")

# Simpan video ke ukuran 1280x720 (supaya lebih efisien & kompresi ukuran file)
NEW_FRAME_WIDTH = 1280
NEW_FRAME_HEIGHT = 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Format video
OUTPUT_PATH_FOLDER = os.path.join("output/videos\playerball", "football-scouting-best-m-1695-aug-segonlyplayer.pt") # Lokasi simpan video hasil inference
try:
  os.makedirs(OUTPUT_PATH_FOLDER)
  print("Folder %s terbuat!\n" % OUTPUT_PATH_FOLDER)
except FileExistsError:
  print("Folder %s telah tersedia\n" % OUTPUT_PATH_FOLDER)
OUTPUT_PATH_VIDEOS = OUTPUT_PATH_FOLDER+f"/{file_name[0]} output.mp4"
out = cv2.VideoWriter(OUTPUT_PATH_VIDEOS, fourcc, fps, (NEW_FRAME_WIDTH, NEW_FRAME_HEIGHT)) # Untuk simpan

#################
frame_nomor = 0

total_ballpossession = []
warnajerseyterdeteksi_temp = []

total_possession = []
warnatext_possession = []
#################

# while frame_nomor < 100:
while cap.isOpened(): # True
    # start = time.time()
    success, frame = cap.read()     # Membaca frame saat ini yang telah diekstrak
    frame_nomor += 1

    if success:
        # Jalankan inference/prediksi pada frame saat ini dan persisting tracks between frames 
        results = model(frame, imgsz=1280, conf=0.6)
        print(f"Memproses Frame Urutan ke", frame_nomor)

        # Menampilkan hasil prediksi (per bunding box) pada frame ini
        for r in results:
            boxes = r.boxes

            player_list = []
            bola_list = []
            playerwarnajersey_list = []

            bola_terdeksi = False

            jumlah_warnajerseyterdeteksi = 0

            for box in boxes:
                # print(box)
                # print(box.cls[0])
                kelas = int(box.cls[0])
                # print(kelas)
                # conf = int(box.conf[0])
                x, y, w, h = box.xywh[0] # xcenter, ycenter, width, height
                x1, y1, x2, y2 = box.xyxy[0] # x1 (xmin), y1 (ymin), x2 (xmax), y2 (ymax)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert setiap value ke int
                x_tengah, y_tengah, w, h = int(x), int(y), int(w), int(h) # convert setiap value ke int
                # print(x_tengah, y_tengah)
                if kelas == 2:
                    playerwarnajersey = klasifikasi_warnajersey(frame[y1:y2, x1:x2]) # crop pemain sesuai ymin:ymax, xmin:xmax
                    # print(playerwarnajersey)
                    if (len(warnajerseyterdeteksi_temp) < 2) and (playerwarnajersey not in warnajerseyterdeteksi_temp) and (jumlah_warnajerseyterdeteksi < 2):
                        jumlah_warnajerseyterdeteksi += 1
                        warnajerseyterdeteksi_temp.append(playerwarnajersey)
                    else:
                        pass
                    # print(warnajerseyterdeteksi_temp)

                    for i in warnajerseyterdeteksi_temp:
                        if i == playerwarnajersey:
                            bbox_frame, player_xyxywh = gambar_boundingbox_jersey(frame, kelas, i, x1, y1, x2, y2, x_tengah, y_tengah, w, h)
                            player_list.append(player_xyxywh)
                            playerwarnajersey_list.append(playerwarnajersey)
                        else:
                            pass
                    # bbox_frame, player_xyxywh, playerwarnajersey = gambar_boundingbox_jersey(frame, kelas, x1, y1, x2, y2, x_tengah, y_tengah, w, h)
                    # player_list.append(player_xyxywh)
                    # playerwarnajersey_list.append(playerwarnajersey)
                elif (kelas == 0) and (bola_terdeksi == False):
                    bbox_frame, bola_xy = gambar_boundingbox_bola(frame, kelas, x1, y1, x2, y2, x_tengah, y_tengah)
                    bola_list.append(bola_xy)
                    bola_terdeksi = True
                else:
                    bbox_frame = frame

            frame_copy = bbox_frame.copy() # duplikat frame bbox_frame yang sudah ada boundingbox player dan bola (jika terdeteksi)
            if bola_list != []: # jika list bola tidak kosong (bola terdeteksi)
                player_ballposession, playerwarnajersey_ballpossession = deteksi_player_ballpossession(player_list, bola_list, playerwarnajersey_list, 35) # 35 = jarak antar objek dalam satuan pixel
                text_player_ballposession, warnatext_player_ballposession = keterangan_ballpossession(playerwarnajersey_ballpossession) # keterangan siapa yang membawa bola sekarang (jika bola dan player terdeteksi membawa bola)
                if (playerwarnajersey_ballpossession != []):
                    total_ballpossession.append(playerwarnajersey_ballpossession) # masukkan warna ke array
                else:
                    pass
                # print(total_ballpossession)
                
                total_possession, warnatext_possession = hitung_total_ballpossession(total_ballpossession) # hitung kemunculan dari warna
                
                if player_ballposession != []:
                    bbox_frame_copy = gambar_segitiga_pemain(frame_copy, player_ballposession)
                    cv2.putText(bbox_frame_copy, f"Ball Possession :", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(bbox_frame_copy, f"{text_player_ballposession}", (285,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warnatext_player_ballposession, 2)
                    cv2.putText(bbox_frame_copy, f"Total Possession :", (50,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if total_possession != [] and warnatext_possession != []:
                        pxke = 300
                        for i, j in enumerate(total_possession):
                            cv2.putText(bbox_frame_copy, f"{j}%", (pxke,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warnatext_possession[i], 2)
                            pxke += 100
                else:
                    bbox_frame_copy = frame_copy
                    cv2.putText(bbox_frame_copy, f"Ball Possession : -", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(bbox_frame_copy, f"Total Possession :", (50,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if total_possession != [] and warnatext_possession != []:
                        pxke = 300
                        for i, j in enumerate(total_possession):
                            cv2.putText(bbox_frame_copy, f"{j}%", (pxke,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warnatext_possession[i], 2)
                            pxke += 100
            else:
                bbox_frame_copy = frame_copy
                cv2.putText(bbox_frame_copy, f"Ball Possession : -", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(bbox_frame_copy, f"Total Possession :", (50,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if total_possession != [] and warnatext_possession != []:
                    pxke = 300
                    for i, j in enumerate(total_possession):
                        cv2.putText(bbox_frame_copy, f"{j}%", (pxke,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warnatext_possession[i], 2)
                        pxke += 100

        # Resize dari 1920x1080 ke 1280x720
        resized_frame = cv2.resize(bbox_frame_copy, (NEW_FRAME_WIDTH, NEW_FRAME_HEIGHT))
        # end = time.time()
        # fps_count = 1/(end-start)
        cv2.putText(resized_frame, f"FPS : {int(fps)}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Simpan/add per frame ke format video
        out.write(resized_frame)

        # # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # Resize dari 1920x1080 ke 1280x720
        resized_frame = cv2.resize(resized_frame, (NEW_FRAME_WIDTH, NEW_FRAME_HEIGHT))

        # Tampilkan hasil di layar
        cv2.imshow("{} Tracking".format(file_name[0]), resized_frame)
        # Untuk menghentikan looping ekstraksi frame dari video dengan menekan 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Berhenti ketika sampai frame terakhir
        break

# Release the video capture object and close the display window
print("\nOutput video telah berhasil disimpan pada '{}!'".format(OUTPUT_PATH_VIDEOS))
cap.release()
out.release()
cv2.destroyAllWindows()