
import torch
import numpy as np
import cv2
from time import time
import datetime

qr=cv2.QRCodeDetector()
class UAVDetector:
    

    def __init__(self, capture_index, model_name):
        """
        hangi kamerayı kullancağımız, hangi modeli kullanacağımız ekran kartı mı yoksa işlemci mi kullanacağız
        ve bazı değişkenlere atama yapıyoruz
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        kameradan görüntü alıyoruz
        """
        qr=cv2.QRCodeDetector()
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Pytorch hub'dan Yolov5 modelini indiriyoruz
        ve bunu modüle geri döndürüyoruz 
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):

        sol_ust_kose=(104,42)
        sag_alt_kose=(312,375)
        cv2.rectangle(frame,(sol_ust_kose),(sag_alt_kose),(0,0,255),2)
        """
        kameradan aldığı görüntüyü modele sokarak ondan tahmin oranı alıyoruz 
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
        

    def class_to_label(self, x):
        """
        classlarımızı labela dönüştürüyoruz.
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        aranan objenin hangi konumlar içinde olduğunu buluyoruz.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.7:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                
                w,h = abs(x1 - x2), abs(y1 - y2)
                center = (x_shape // 2, y_shape // 2)
                
                obj_center=[int(x1 + w / 2), int(y1 + h / 2)]
                cv2.line(frame, center, obj_center, (0, 0, 0), 2)

        return frame

    def __call__(self):
        
        """
        kameramızı açarak aranan nesnenin nerede olduğunu hangi nesne olduğunu ve % kaç olasılıkla onun olduğunu yazıyoruz.
        """
        cap = self.get_video_capture()
        assert cap.isOpened()
      
        while True:
              
            ret, frame = cap.read()
            assert ret
            

        
            frame = cv2.resize(frame, (416,416))
            
            decodedText, points, _ = qr.detectAndDecode(frame)
            if points is not None:
                nrOfPoints = len(points)
                for i in range(nrOfPoints):
                    nextPointIndex = (i + 1) % nrOfPoints
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, decodedText, (140 , 50), font, .5, (0, 0, 0), 1, cv2.LINE_AA)
                    print(decodedText)

            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)

            t= datetime.datetime.now()  ###anlık saat içerisinden saat dakika ve mikrosaniye cinsinden veri yazdırma
            x= (t.strftime("%X,%f"))
            print(x)
            
            


            #print(f"her saniye frame yaz : {fps}")
            
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.putText(frame, str(x),(180,20),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(132,225,86),2)
            cv2.imshow('YOLOv5 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
      
        cap.release()
        cv2.destroyAllWindows()
        
# yeni bir obje oluşturarak çalıştırıyoruz.

detector = UAVDetector(capture_index=0, model_name='best.pt')
detector()

 