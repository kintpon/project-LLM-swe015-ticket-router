# Bileşenlerin açıklaması (sunum hazırlığı)

Her bileşen için: ne yapar, neden böyle, ve hocanın sorabileceği 2 soru + kısa cevap. Sayılar için `results/results.md` tablosuna bak; burada sayı yazılmadı.

## Veri – `src/data.py`, `data/appliance.csv`
CLINC150 `plus` veri setini (150 niyet + `oos`, resmi train/val/test) Hugging Face'ten indirir ve 100 satırlık beyaz eşya verisini ekler. Bölmeler bir kez, seed 42 ile kurulur ve `data/splits/*.parquet` dosyalarına yazılır; her model yalnızca bu dosyaları okur. Test'te de geçen birkaç train cümlesi train'den çıkarıldı.
- **S: Neden CLINC150 `plus`?** C: Gerçek müşteri mesajlarına benzeyen kısa cümleler, çok sayıda niyet ve ayrı bir "kapsam dışı" (`oos`) sınıfı var; insan temsilciye yönlendirmeyi ölçmek için tam uygun.
- **S: 15 test satırıyla beyaz eşya doğruluğu güvenilir mi?** C: Hayır, bu yüzden tabloda "15 satır" diye uyarıyla verdik; asıl metrik CLINC test setindeki in-scope doğruluk ve oos recall.

## Ekipler – `src/teams.py`
Her niyeti bir ekibe eşleyen sözlük (bankacılık, seyahat, yemek, …). Beyaz eşya niyetleri Technical Service / Spare Parts / Warranty'ye, `oos` Human agent'a, sohbet niyetleri Auto reply'a gider.
- **S: Modeller ekibi mi tahmin ediyor, niyeti mi?** C: Niyeti; ekip bu tablodan bakılarak bulunur. Böylece ekip yapısı değişirse modeli yeniden eğitmek gerekmez.
- **S: Neden 10 ekip?** C: CLINC alan gruplarından çıktı; bu karar D10'da kayıtlı.

## Ön işleme – `src/preprocess.py`
Sırasıyla: model kodunu (`DW-6534`) çıkarır, küçük harfe çevirir, URL/e-postayı yer tutucuya çevirir, uzatılmış harfleri kısaltır (`soooo`→`soo`), yazım hatasını düzeltir. Düzeltme kendi yazdığımız Levenshtein mesafesiyle, sadece ürün kelimelerine doğru ve sadece eğitim sözlüğünde olmayan, en az 5 harfli kelimelerde yapılır. Aynı fonksiyon eğitimde, değerlendirmede ve uygulamada kullanılır.
- **S: "over" neden "oven" olmuyor?** C: "over" eğitim sözlüğünde olan gerçek bir kelime; sözlükteki kelimelere hiç dokunulmuyor.
- **S: Levenshtein nasıl çalışıyor?** C: Dinamik programlama tablosu: bir kelimeyi diğerine çevirmek için gereken en az ekleme/silme/değiştirme sayısı.

## Kurallar – `src/rules.py`
ELIZA tarzı düzenli ifadeler: "hello", "thanks", "bye" gibi sohbet mesajlarına hazır cevap verir. Kural mesajın tamamına uymalı, yoksa "thanks, my oven is still broken" gibi bir şikayet yutulurdu. Sadece uygulamada, modellerden önce çalışır.
- **S: Kurallar değerlendirmeye dahil mi?** C: Hayır; modelleri adil karşılaştırmak için değerlendirme sadece modelleri ölçer.
- **S: Neden sohbet için model değil kural?** C: Az ve sabit kalıplar; kural hızlı, açıklanabilir ve yanlış ekibe göndermez.

## M1 – `src/m1_tfidf.py` (alan B)
Kelime ve ikili kelime (bigram) tf-idf vektörleri + lojistik regresyon. Klasik, hızlı ve GPU gerektirmeyen bir taban çizgisi.
- **S: tf-idf ne demek?** C: Bir kelime mesajda sık, tüm veride seyrekse ağırlığı yüksek olur; ortak kelimeler önemsizleşir.
- **S: M1 neden M2'den büyük dosya?** C: Bigram sözlüğü çok büyük ve her sınıf için her özelliğe bir ağırlık tutuluyor.

## M2 – `src/m2_ffn.py` (alan A)
Train metinleri üzerinde word2vec (100 boyut) eğitilir; mesaj, kelime vektörlerinin ortalaması olur. Bu vektör kendi yazdığımız MLP'ye (100→256→sınıf, ReLU, dropout 0.2) girer. Eğitim döngüsü elle yazıldı: SGD + momentum, cross-entropy, `loss.backward()`, val doğruluğuna göre erken durdurma.
- **S: Neden en düşük doğruluk M2'de?** C: Ortalama almak kelime sırasını kaybeder ve word2vec sadece bizim küçük train metinleriyle eğitildi.
- **S: Erken durdurma nasıl?** C: Her epoch sonunda val doğruluğu ölçülür; 15 epoch boyunca iyileşmezse durur ve en iyi epoch'un ağırlıkları kullanılır.

## M3 – `src/m3_bert.py` (alan C)
`bert-base-uncased` (~110M parametre) sınıflandırma için ince ayar yapıldı: max uzunluk 32, lr 3e-5, batch 32, 3 epoch, en iyi epoch val doğruluğuna göre saklandı. En doğru ama en büyük ve en yavaş model; distilasyonda öğretmen.
- **S: Fine-tuning ne demek?** C: Önceden eğitilmiş BERT'in üstüne yeni bir sınıflandırma katmanı eklenip tüm ağ bizim verimizle birkaç epoch daha eğitiliyor.
- **S: Neden max_length 32?** C: CLINC mesajları kısa; 32 token neredeyse hepsine yetiyor ve hızı artırıyor.

## M3s – `src/m4_distill.py` (alan D, adım 1)
DistilBERT (~67M parametre) öğrenci, M3 öğretmenden öğrenir: kayıp = 0.5·CE(etiket) + 0.5·T²·KL(öğrenci/T ‖ öğretmen/T), T = 2, 5 epoch. Öğretmen çıktıları train için bir kez hesaplanıp kaydedildi.
- **S: Sıcaklık T ne işe yarıyor?** C: Olasılıkları yumuşatır; öğrenci sadece doğru sınıfı değil, öğretmenin hangi sınıfları birbirine benzer bulduğunu da öğrenir. T² ile çarpmak gradyan ölçeğini dengeler.
- **S: Distilasyon ne kazandırdı?** C: Tabloda M3 ile M3s'i karşılaştır: doğruluk neredeyse aynı, gecikme ve boyut belirgin şekilde düşük.

## M4 – `src/m4_quantize.py` (alan D, adım 2)
M3s ONNX'e aktarılır, sonra ONNX Runtime `quantize_dynamic` ile ağırlıklar INT8'e çevrilir. CPU'da sadece ONNX Runtime ile çalışır. Uygulamada yönlendirme kararını M4 verir.
- **S: Neden PyTorch quantization değil?** C: PyTorch'un `quantize_dynamic` API'si kullanımdan kaldırılıyor; ONNX Runtime ödevin teknik listesinde var ve PyTorch'suz çalışıyor.
- **S: Dinamik kuantizasyon ne demek?** C: Ağırlıklar önceden INT8'e çevrilir, aktivasyonların ölçeği her çalıştırmada anlık hesaplanır; kalibrasyon verisi gerekmez.

## Yönlendirme ve değerlendirme – `src/evaluate.py`
Tüm modeller için tek fonksiyon. En olası niyet `oos` ise ya da güven eşiğin altındaysa mesaj Human agent'a gider. Eşik her model için sadece val setinde, (in-scope doğruluk + oos recall)/2'yi en yüksek yapan değer olarak seçilir. Test'te metrikler, dosya boyutu ve 200 mesajlık CPU gecikmesi (4 thread) ölçülür.
- **S: Eşik neden test'te değil val'de seçildi?** C: Test'te seçmek sonucu test'e göre ayarlamak olurdu (sızıntı); test sadece son ölçüm için.
- **S: Human agent'a giden doğru niyet neden yanlış sayılıyor?** C: Müşteri doğru ekibe otomatik ulaşmadı; metrik "otomatik doğru yönlendirme" oranını ölçüyor.

## Uygulama – `app.py`
Gradio Blocks: metin kutusu + 4 örnek. Önce kurallar, sonra beş modelin hepsi çalışır; kararı M4 verir, tablo hepsini yan yana gösterir. Boş mesajda model çağrılmaz, 500 karakterden uzun mesaj kesilir. `models/` yoksa ağırlıklar Hugging Face'ten indirilir.
- **S: Neden beş model birden çalışıyor?** C: Karşılaştırmayı canlı göstermek için; karar yine yalnızca M4'ün.
- **S: Türkçe bir mesaj yazarsam ne olur?** C: Model güveni düşük çıkar ve mesaj Human agent'a gider; testlerde bu durum kontrol ediliyor.
