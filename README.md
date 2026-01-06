# DevAI – CIFAR-10 Ensemble Learning Projesi

Bu proje, **Derin Öğrenme** dersi kapsamında geliştirilmiştir.  
Amaç, **CIFAR-10** veri seti üzerinde farklı CNN tabanlı modelleri eğitmek ve
**Ensemble Learning** yaklaşımı ile performansı karşılaştırmaktır.

## 📌 Proje Özeti
- Veri seti: CIFAR-10
- Problem tipi: **Multi-class görüntü sınıflandırma (10 sınıf)**
- Kullanılan yöntemler:
  - Simple CNN
  - Advanced CNN
  - Residual CNN
  - CNN + LSTM (Hybrid)
  - ResNet18 (Transfer Learning)
  - SE-Attention CNN
- Ensemble yöntemi ile model çıktıları birleştirilmiştir.

## 👥 Proje Ekibi
- Abdullah
- Cuneyd
- Kasim

## 📁 Proje Klasör Yapısı
DevAI_Project/
│
├── src/
│   ├── data/          # Veri yükleme işlemleri
│   ├── models/        # CNN modelleri
│   ├── runs/          # Eğitim ve test scriptleri
│   └── utils/         # Ensemble ve görselleştirme
│
├── results/
│   └── classwise/     # Sınıf bazlı sonuçlar (CSV)
│
├── data/              # CIFAR-10 (GitHub’a eklenmedi)
├── requirements.txt
└── README.md

## ▶️ Çalıştırma
Gerekli kütüphaneleri yüklemek için:
```bash
pip install -r requirements.txt
python src/runs/main.py
📊 Sonuçlar

Model performansları classwise CSV dosyaları olarak results/classwise klasöründe bulunmaktadır.

Ensemble Learning, tekil modellere göre daha dengeli sonuçlar vermiştir.

⚠️ Not

CIFAR-10 veri seti GitHub dosya boyutu sınırı (100MB) nedeniyle repoya eklenmemiştir.
Veri seti resmi kaynaktan indirilebilir.

📚 Kullanılan Teknolojiler

Python

PyTorch / TensorFlow

NumPy

Matplotlib
