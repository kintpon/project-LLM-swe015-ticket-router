# SWE015 Proje Planı – İlerleme Kutuları
Kaynak: "SWE015 Ödevi – Proje Planı.pdf". Bir madde gerçekten çalıştığında (kontrolü gerçekten çalıştırdıktan sonra) [x] yapılır.

## P0 · Proje iskeleti
- [x] Git deposu ve klasör yapısı kuruldu
- [x] .venv oluşturuldu, paketler requirements.txt ile kuruldu
- [x] config.py, src/logutil.py ve .gitignore yazıldı
- [x] İlk çalışma günlüğü kaydı (worklog_001.md) ve PROGRESS.md açıldı

## P1 · Veri, ön işleme ve klasik model (M1)
- [x] CLINC150 indirildi, 100 beyaz eşya mesajı eklendi, bölümler data/splits/ altına kaydedildi
- [x] teams.py: her niyetin bir ekibi var
- [x] Ön işleme: regex temizleme, model kodu çıkarma, Levenshtein ile yazım düzeltme
- [x] ELIZA tarzı küçük konuşma kuralları
- [x] Ortak değerlendirme fonksiyonu (evaluate.py) ve tests/test_basic.py testleri geçiyor
- [x] M1 (tf-idf + lojistik regresyon) eğitildi ve sonuç tablosuna girdi
- [x] Veri bütünlüğü testleri (test_data.py) geçiyor: eğitim ve test verisi çakışmıyor
- [x] İlk karar kayıtları docs/decisions.md dosyasına yazıldı

## P2 · Kendi sinir ağımız (M2)
- [x] word2vec eğitildi
- [x] MLP, elle yazılmış SGD döngüsüyle eğitildi
- [x] M2 sonuç tablosuna girdi
- [x] Uçtan uca test (test_pipeline.py) geçiyor: aynı model iki kez ölçülünce tabloda tek satır kalıyor

## P3 · Transformer, distillation ve quantization (M3, M3s, M4)
- [x] M3: BERT fine-tune edildi (Colab ya da CPU)
- [x] M3s: DistilBERT, BERT'ten distillation ile eğitildi
- [x] M4: öğrenci model ONNX'e aktarıldı ve INT8 quantize edildi
- [x] M4 dosyası M3s'den küçük; ikisinin tahminleri en az %90 örtüşüyor
- [x] Üç model de sonuç tablosuna girdi

## P4 · Uygulama
- [x] Gradio arayüzü açılıyor
- [x] Dört örnek cümle de mantıklı sonuç veriyor
- [x] Uygulama quantize modeli (M4) çalıştırıyor, dört modeli yan yana gösteriyor
- [x] failures.csv ve doğruluk–gecikme grafiği üretildi
- [x] Zor girdiler çökmüyor: boş metin, yalnızca model kodu, Türkçe cümle, çok uzun metin

## P5 · Dokümanlar ve teslime hazırlık
- [x] Mimari diyagram yazıldı ve kodla karşılaştırıldı
- [x] slides_material.md ve explain.md hazır
- [x] Ağırlıklar Hugging Face'e yüklendi
- [x] README temiz ortamda denendi, python app.py açılıyor
- [x] Repo GitHub'a gönderildi

## P6 · Son denetim
- [x] Gereksinim tablosundaki her madde OK
- [x] Kod dosyaları 500, log dosyaları 300 satırı geçmiyor
- [x] Dokümanlardaki sayılar results.csv ile aynı
- [x] Kod incelemesi (docs/code_review.md) yapıldı: gizli anahtar yok, test verisi eğitime sızmıyor
- [x] Yayın kontrolü geçti: testler başarılı, repoda 5 MB üstü dosya yok, temiz klondan uygulama açılıyor
- [x] submission etiketi (git tag) atıldı

## Senin tarafında kalanlar
- [x] GitHub'da boş bir repo aç (public ya da hocayla paylaşılmış)
- [x] Hugging Face hesabı aç ve huggingface-cli login ile giriş yap
- [ ] Colab'da eğitimi çalıştır ve models/ klasörünü indir
- [x] Ağırlıkları scripts/upload_weights.py ile Hugging Face'e yükle
- [x] Repoyu GitHub'a gönder (git push)
- [ ] docs/explain.md dosyasını okuyup her bileşeni kendi cümlelerinle anlatabildiğini dene
- [ ] Sunumu docs/slides_material.md içindeki malzemeyle hazırla
- [ ] AI kullanım slaydını kendi ifadenle yaz (Claude Code'un neyi yaptığını açıkça belirt)
- [ ] Videoyu çek, YouTube'a (liste dışı) ya da Drive'a yükle
- [ ] İlk slayda ad, öğrenci numarası, senaryo, repo linki ve video linkini koy

## Teslim öncesi kontrol listesi (Cumartesi, hepsi işaretli olmalı)
- [x] Uygulama yeni girdi alıp sonuç döndürüyor (R1)
- [ ] Dört teknik alanı kullanılmış ve technique map slaydı var (R2)
- [x] Temel model, sinir ağı, Transformer ve quantize Transformer aynı test setinde ölçülmüş (R3)
- [x] Sonuç tablosunda her model için metrik, boyut ve gecikme var; donanım yazılı (R4)
- [x] Uygulama quantize Transformer'ı çalıştırıyor (R5)
- [x] Mimari diyagram kodla uyuşuyor (R6)
- [x] README'deki kurulum ve çalıştırma komutu temiz ortamda çalışıyor; repoda büyük ağırlık dosyası yok (R7)
- [ ] AI kullanım slaydı var (R8)
- [ ] Slaytlar Giriş / Geliştirme / Sonuç sırasında, 8–12 slayt, ilk slaytta iki link de var
- [ ] Sonuç bölümünde en az bir hata örneği var
- [ ] Video en fazla 2:00, sesli, canlı uygulamayı, mimariyi ve sonuçları gösteriyor
- [ ] Video ve repo linkleri gizli pencerede açılıyor
- [ ] Veri kaynakları (CLINC150) sunumda ve README'de belirtilmiş
- [ ] Teslim formu gönderildi
