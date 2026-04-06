import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

# --- 1. MODEL MİMARİSİ (BmiNet Sınıf Tanımlaması) ---
class BmiNet(nn.Module):
    def __init__(self):
        super(BmiNet, self).__init__()
        # Evrişim (Convolution) Katmanları
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Tam Bağlı (Fully Connected) Katmanlar
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1) # Regresyon çıktısı (Tek nöron)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28) # Düzleştirme (Flattening)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. ÖNCEDEN EĞİTİLMİŞ MODELİN YÜKLENMESİ ---
# Sistemde CUDA destekli bir GPU varsa onu kullan, yoksa CPU'ya geç (Dinamik Cihaz Yönetimi)
cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sistem Başlatıldı. Kullanılan Donanım: {cihaz}")

model = BmiNet().to(cihaz)

# Model ağırlıklarının (state_dict) yüklenmesi ve değerlendirme moduna alınması
model.load_state_dict(torch.load("bmi_model_buyuk_veri_v1.pt", map_location=cihaz))
model.eval() 

# Girdi görüntüleri için standart normalizasyon adımları
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. ÇIKARIM (INFERENCE) FONKSİYONU ---
def tahmin_et(foto):
    if foto is None:
        return "Lütfen bir fotoğraf yükleyin."
    
    # Görüntünün modele uygun tensör formatına dönüştürülmesi
    foto = foto.convert("RGB")
    foto_tensor = transform(foto).unsqueeze(0).to(cihaz)
    
    # Gradyan hesaplamaları kapatılarak (no_grad) bellek optimizasyonu sağlanır
    with torch.no_grad():
        tahmin = model(foto_tensor).item()
        
    # Tahmin edilen regresyon skorunun DSÖ standartlarına göre sınıflandırılması
    kategori = ""
    if tahmin < 18.5: kategori = "Zayıf"
    elif tahmin < 25.0: kategori = "Normal"
    elif tahmin < 30.0: kategori = "Fazla Kilolu"
    else: kategori = "Obezite"
        
    return f"Tahmini BMI Skoru: {tahmin:.2f}\nDSÖ Sınıflandırması: {kategori}"

# --- 4. KULLANICI ARAYÜZÜ (GRADIO) ---
arayuz = gr.Interface(
    fn=tahmin_et,
    inputs=gr.Image(type="pil", label="Kişinin Tam Boy Fotoğrafını Yükleyin"),
    outputs=gr.Textbox(label="Yapay Zeka Analiz Sonucu"),
    title="Evrişimli Sinir Ağları (CNN) Tabanlı BMI Tahmin Sistemi",
    description="Uyarı: Model 6000 adet tam boy fotoğraf ile eğitilmiştir. Doğru proporsiyon analizi için lütfen sadece yüz veya üst gövde değil, tüm vücudu gösteren fotoğraflar yükleyin."
)

if __name__ == "__main__":
    arayuz.launch()
