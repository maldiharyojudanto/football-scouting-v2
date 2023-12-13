import cv2
import numpy as np

def gambar_boundingbox_jersey(frame, kelas, warnajersey, x1, y1, x2, y2, x_tengah, y_tengah, w, h):
    if kelas == 2: # Jika kelas ini player
        bbox_frame = cv2.rectangle(frame, (x1,y1), (x2,y2), warnajersey, 2)
        # top_left_x = x1 - 35
        # top_left_y = y1 - 35
        # widthx2 = w + 35*2
        # heightx2 = h + 35*2
        # bbox_frame = cv2.rectangle(frame, (top_left_x,top_left_y), (top_left_x+widthx2,top_left_y+heightx2), warnajersey, 2)
        # bbox_frame = cv2.putText(frame, "id", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warnajersey, 2)
        # bbox_frame = cv2.circle(frame, (x_tengah, y_tengah), radius=1, color=(0, 0, 0), thickness=2)
        player_posisi = [x1, y1, x2, y2, x_tengah, y_tengah, w, h] # xmin, ymin, xmax, ymax, x_tengah, y_tengah, w, h
    else:
        bbox_frame = frame
    
    return bbox_frame, player_posisi

def gambar_boundingbox_bola(frame, kelas, x1, y1, x2, y2, x_tengah, y_tengah):
    if kelas == 0: # Jika kelas ini bola
        segitiga_bola = np.array([[x_tengah - 10 // 2, y_tengah - 20 - 10],
                [x_tengah, y_tengah - 20],
                [x_tengah + 10 // 2, y_tengah - 20 - 10]])
        
        bbox_frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
        # bbox_frame = cv2.circle(frame, (x_tengah, y_tengah), radius=1, color=(0, 0, 0), thickness=1)
        bbox_frame = cv2.drawContours(frame, [segitiga_bola], 0, (0,0,0), thickness=2)
        bbox_frame = cv2.drawContours(frame, [segitiga_bola], 0, (0,255,0), thickness=-1)
        bola_posisi = [x_tengah, y_tengah]
    else:
        bbox_frame = frame

    return bbox_frame, bola_posisi

def deteksi_player_ballpossession(player_list, bola_list, player_warnajersey_list, jarak=30):
    bola_x = bola_list[0][0]
    bola_y = bola_list[0][1]
    player_index = []
    player_warnajersey_index = []

    for i, player in enumerate(player_list):
        top_left_x = player[0] - jarak
        top_left_y = player[1] - jarak
        widthx2 = player[6] + jarak*2
        heightx2 = player[7] + jarak*2
        
        # print(top_left_x, top_left_y, widthx2, heightx2)
        # print(bola_list[0][0], bola_list[0][1])

        """
        Jika min_x < x_center_bola < max_x and min_y < y_center_bola < max_y
        min_x dari player (top left x) kurang dari titik center x bola kurang dari max_x dari player (bottom right x)
        dan
        min_y dari player (top left y) kurang dari titik center y bola kurang dari max_y dari player (bottom right y)
        """
        if top_left_x < bola_x < top_left_x+widthx2 and top_left_y < bola_y < top_left_y+heightx2:
            # print("True")
            player_index.append(player)
            player_warnajersey_index.append(player_warnajersey_list[i])
        else:
            # print("False")
            pass
    
    # print(len(player_index))
    player_akan_digambar = []
    if len(player_index) == 0:
        pass
    elif len(player_index) == 1:
        player_akan_digambar = player_index[0]
    elif len(player_index) > 1:
        player_akan_digambar = player_index[0]
    else:
        pass

    # print(player_warnajersey_index)
    ballpossession_akan_ditulis = []
    if len(player_warnajersey_index) == 0:
        pass
    elif len(player_warnajersey_index) == 1:
        ballpossession_akan_ditulis = player_warnajersey_index[0]
    elif len(player_warnajersey_index) > 1:
        ballpossession_akan_ditulis = player_warnajersey_index[0]
    else:
        pass

    # print(player_akan_digambar)
    # print(playerball_akan_digambar)
    return player_akan_digambar, ballpossession_akan_ditulis
    
def gambar_segitiga_pemain(frame, player):
    x_tengah = player[4]
    y_tengah = player[5]
    if player != []: # jika player tidak kosong
        segitiga_player = np.array([[x_tengah - 10 // 2, y_tengah - 50 - 10],
                [x_tengah, y_tengah - 50],
                [x_tengah + 10 // 2, y_tengah - 50 - 10]])
        
        # bbox_frame = cv2.circle(frame, (x_tengah, y_tengah), radius=1, color=(0, 0, 0), thickness=3)
        bbox_frame = cv2.drawContours(frame, [segitiga_player], 0, (0,0,0), thickness=2)
        bbox_frame = cv2.drawContours(frame, [segitiga_player], 0, (255,0,0), thickness=-1)
    else:
        bbox_frame = frame

    return bbox_frame

def keterangan_ballpossession(playerwarnajersey_ballpossession):
    if playerwarnajersey_ballpossession == (0, 0, 128):
        bola_dikuasai_oleh = "Sekarang bola dikuasai oleh tim biru"
        warnatext_ballpossession = (0, 0, 128)
    elif playerwarnajersey_ballpossession == (0, 192, 192):
        bola_dikuasai_oleh = "Sekarang bola dikuasai oleh tim biru muda"
        warnatext_ballpossession = (0, 192, 192)
    elif playerwarnajersey_ballpossession == (0, 128, 0):
        bola_dikuasai_oleh = "Sekarang bola dikuasai oleh tim hijau"
        warnatext_ballpossession = (0, 128, 0)
    elif playerwarnajersey_ballpossession == (0, 0, 0):
        bola_dikuasai_oleh = "Sekarang bola dikuasai oleh tim hitam"
        warnatext_ballpossession = (0, 0, 0)
    elif playerwarnajersey_ballpossession == (192, 192, 0):
        bola_dikuasai_oleh = "Sekarang bola dikuasai oleh tim kuning"
        warnatext_ballpossession = (192, 192, 0)
    elif playerwarnajersey_ballpossession == (255, 0, 0):
        bola_dikuasai_oleh = "Sekarang bola dikuasai oleh tim merah"
        warnatext_ballpossession = (255, 0, 0)
    elif playerwarnajersey_ballpossession == (255, 255, 255):
        bola_dikuasai_oleh = "Sekarang bola dikuasai oleh tim putih"
        warnatext_ballpossession = (255, 255, 255)
    else:
        bola_dikuasai_oleh = "Sekarang bola dikuasai oleh tim putih"
        warnatext_ballpossession = (255, 255, 255)

    return bola_dikuasai_oleh, warnatext_ballpossession

def hitung_total_ballpossession(total_possession):
    total_poss = []
    res = {}
    warnanya = []

    kata_sebelum = []
    jumlah_kata_sebelum = []

    counter_isi = 0

    for i in total_possession:
        res[i] = total_possession.count(i)
        if (i not in kata_sebelum) and (counter_isi < 2):
            counter_isi += 1
            kata_sebelum.append(i)
            jumlah_kata_sebelum.append(res[i])
            # print(kata_sebelum, jumlah_kata_sebelum)
            # print(sum(jumlah_kata_sebelum))
    
    for i in jumlah_kata_sebelum:
        ball_persen = round(i/sum(jumlah_kata_sebelum)*100)
        total_poss.append(ball_persen)

    for i in kata_sebelum:
        warnanya.append(i)

    return total_poss, warnanya