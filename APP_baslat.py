import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr
from PIL import Image
import os

# --- 1. CİHAZ VE MODEL MİMARİSİ ---
cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelin iskeleti (Eğitimdekiyle birebir aynı olmak zorunda)
class BmiNet(nn.Module):
    def __init__(self):
        super(BmiNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. MODELİ YÜKLEME ---
# DİKKAT: .pth dosyasının bu app.py dosyası ile aynı klasörde olması gerekir.
model_yolu = "bmi_model_buyuk_veri_v1.pt"
model = BmiNet().to(cihaz)

if os.path.exists(model_yolu):
    # map_location=cihaz diyerek, modelin doğrudan RTX 3080'e gitmesini garantiye alıyoruz
    model.load_state_dict(torch.load(model_yolu, map_location=cihaz))
    model.eval()
    print("✅ Yapay Zeka Motoru Başarıyla Uyandırıldı!")
else:
    print(f"⚠️ HATA: '{model_yolu}' bulunamadı! Lütfen model dosyasının bu kodla aynı klasörde olduğundan emin olun.")

# --- 3. GÖRÜNTÜ İŞLEME VE TAHMİN FONKSİYONU ---
donusum = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def yapay_zeka_analizi(fotograf):
    if fotograf is None:
        return "Lütfen bir fotoğraf yükleyin."
    
    img_tensor = donusum(fotograf).unsqueeze(0).to(cihaz)
    
    with torch.no_grad():
        tahmin = model(img_tensor)
        bmi_degeri = tahmin.item()
        
    # WHO Kategorisi
    if bmi_degeri < 18.5:
        kategori = "Zayıf"
    elif 18.5 <= bmi_degeri < 25.0:
        kategori = "Normal Kilolu"
    elif 25.0 <= bmi_degeri < 30.0:
        kategori = "Fazla Kilolu"
    else:
        kategori = "Obezite"
        
    rapor = f"""
### 📊 Vücut Kitle İndeksi (BMI) Analiz Raporu

**Hesaplanan BMI:** {bmi_degeri:.2f}
**Dünya Sağlık Örgütü (WHO) Kategorisi:** **{kategori}**

---
*Not: Bu tahmin, Derin Öğrenme (CNN) modeli tarafından yüz ve omuz hatlarına bakılarak yapılmıştır. Işık, kamera açısı (zoom) ve kıyafetler sonucu etkileyebilir.*
"""
    return rapor

# --- 4. ARAYÜZ TASARIMI ---
arayuz = gr.Interface(
    fn=yapay_zeka_analizi,
    inputs=gr.Image(type="pil", label="Fotoğraf Yükle"),
    outputs=gr.Markdown(label="Sonuç Paneli"),
    title="Yapay Zeka Destekli Vücut Analizi",
    description="Modeli test etmek için bir fotoğraf yükleyin. Algoritmamız saniyeler içinde tahmini BMI değerinizi hesaplayacaktır."
)
# --- 5. UYGULAMAYI BAŞLATMA ---
if __name__ == "__main__":
    arayuz.launch(share=True)