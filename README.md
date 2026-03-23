# 🤖 Yapay Zeka Destekli Fotoğraftan BMI (Vücut Kitle İndeksi) Tahmini

Bu Proje Makine öğrenmesi ders projesi için yapılmıştır, finansal amaç güdülmemektedir.
Bu proje, derin öğrenme (Deep Learning) ve Evrişimli Sinir Ağları (CNN) teknikleri kullanılarak, kişilerin yüz ve boyun/omuz bölgesi fotoğraflarından Vücut Kitle İndeksi (BMI) değerini saniyeler içinde tahmin eden bir yapay zeka uygulamasıdır. 

Uygulama, PyTorch ile eğitilmiş özel bir sinir ağı modelini (BmiNet) arka planda çalıştırır ve **Gradio** kütüphanesi ile geliştirilmiş kullanıcı dostu bir web arayüzü üzerinden hizmet verir.

## 🚀 Projenin Öne Çıkan Özellikleri

- **Özel CNN Mimarisi (BmiNet):** Yüz hatlarındaki hacimsel ve yapısal farklılıkları öğrenmek üzere 3 evrişim (convolutional) ve 3 tam bağlı (fully connected) katmandan oluşan özgün bir mimari tasarlanmıştır.
- **Data Augmentation (Veri Çeşitlendirme):** Modelin ezberlemesini (overfitting) önlemek ve farklı açılara toleransını artırmak için eğitim sırasında `RandomHorizontalFlip` ve `RandomRotation` gibi görüntü işleme teknikleri kullanılmıştır.
- **Kullanıcı Dostu Web Arayüzü:** Gradio ile tasarlanan arayüz, kullanıcının fotoğraf yüklemesine, anlık BMI değerini ve Dünya Sağlık Örgütü (WHO) kategorisini görmesine olanak tanır.
- **Sürekli Öğrenme (Flagging):** Arayüze entegre edilen "Geri Bildirim" sistemi sayesinde hatalı tahminler kategorize edilerek kaydedilir ve modelin gelecekteki eğitimleri için yeni bir veri havuzu oluşturur.

## 📊 Veri Seti ve Performans

Projenin eğitim aşamasında yaklaşık **6.000 fotoğraftan** oluşan geniş çaplı bir görsel veri seti kullanılmıştır. Tüm görseller `224x224` piksel boyutuna standardize edilmiş ve profesyonel renk normalizasyonu (RGB kanalları için) uygulanmıştır.

- **Eğitim Süresi:** 20 Epoch (NVIDIA RTX 3080 GPU ile)
- **Kayıp Fonksiyonu (Loss Function):** Mean Squared Error (MSE)
- **Final Eğitim Hatası (MSE Loss):** ~1.57
- **Doğruluk:** Stüdyo standartlarındaki fotoğraflarda gerçek değere **±0.5 ila ±1.2** puanlık hata payıyla (gerçeğe çok yakın) tahmin yapabilmektedir.

## 🧠 Model Mimarisi Detayları

Model, PyTorch altyapısı ile sıfırdan inşa edilmiştir:
1. `Conv2d(3, 32)` -> `MaxPool2d` -> `ReLU`
2. `Conv2d(32, 64)` -> `MaxPool2d` -> `ReLU`
3. `Conv2d(64, 128)` -> `MaxPool2d` -> `ReLU`
4. Flatten -> `Linear(128*28*28, 512)` -> `Linear(512, 128)` -> `Linear(128, 1)`

## ⚙️ Kurulum ve Çalıştırma (Lokal Ortam)

**ÖNEMLİ NOT:** Model dosyasının boyutu (~200 MB) GitHub limitlerini aştığı için harici bir sunucuda tutulmaktadır. 

**1. Depoyu Klonlayın:**
`bash
git clone https://github.com/KULLANICI_ADIN/ai-bmi-predictor.git
cd ai-bmi-predictor
`

**2. Gerekli Kütüphaneleri Yükleyin:**
`bash
pip install -r requirements.txt
`

**3. Yapay Zeka Modelini İndirin:**
- [BURAYA TIKLAYARAK](https://drive.google.com/file/d/1IVttpG0qap3TWODvrRbIKIv82jpEuWvl/view?usp=sharing)  `bmi_model_buyuk_veri_v1.pt` isimli model dosyasını indirin.
- İndirdiğiniz bu dosyayı, `APP_baslat.py` dosyası ile **aynı klasörün içine** atın.

**4. Uygulamayı Başlatın:**
`bash
python APP_baslat.py
`
(Kod çalıştıktan sonra tarayıcınız otomatik olarak açılacak ve Gradio arayüzü karşınıza gelecektir. Sunucu linki üzerinden projeyi başka cihazlardan da test edebilirsiniz.)

⚠️ Limitasyonlar ve Gelecek Çalışmalar (Future Work)

-Kamera Lensi ve Açı Yanılsamaları: Selfie kamerasıyla çok yakından çekilen ("balık gözü" etkisine maruz kalmış) fotoğraflarda model, yüzün ekranda kapladığı piksel yoğunluğunu "hacim artışı" olarak yorumlayıp skoru olduğundan yüksek (Örn: +4 puan) tahmin edebilmektedir.

-Arka Plan ve Işık: "In-the-wild" (doğal ortam) fotoğraflarında arka plandaki karmaşıklık ve ışık gölgeleri, çene/boyun hattının yanlış algılanmasına sebep olabilmektedir.
