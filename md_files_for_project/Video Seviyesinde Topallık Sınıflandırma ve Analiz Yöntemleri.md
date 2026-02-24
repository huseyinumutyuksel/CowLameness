Klasör seviyesinde etiketleme (örneğin; bir klasördeki tüm videoların "Sağlıklı", diğerindekilerin "Topal" olarak etiketlenmesi), teknik literatürde **"Video Seviyesinde Sınıflandırma" (Video-Level Classification)** veya **"Zayıf Denetimli Öğrenme" (Weakly Supervised Learning)** kapsamına girer. Bu senaryoda, modelin her bir karede (frame) ne olduğunu bilmesi gerekmez; bunun yerine videonun bütününden bir çıkarım yaparak klasörün etiketini (yani ineğin genel durumunu) tahmin etmeye çalışır.  
Kaynaklara göre, bu tür veri setlerinde izlenen yaklaşımlar şunlardır:

### 1\. Uçtan Uca Uzamsal-Zamansal (Spatiotemporal) Modelleme

Bu yöntem, özellik çıkarmayla (feature extraction) uğraşmadan, videoyu bir bütün olarak modele verip doğrudan "Sağlıklı" veya "Topal" çıktısı almayı hedefler. Klasör seviyesindeki etiket, modelin eğitim hedefidir.

* **3D CNN (3 Boyutlu Konvolüsyonel Ağlar):** Geleneksel 2D CNN'ler tek bir kareyi analiz ederken, 3D CNN'ler (örneğin **C3D**, **R3D**, **R2Plus1D**) videodaki hem görsel (uzamsal) hem de hareket (zamansal) bilgisini aynı anda işler. Yakın tarihli bir çalışmada, 3D CNN modelleri, poz kestirimi gibi ara adımları atlayarak videoyu doğrudan sınıflandırmış ve **%90** doğruluk oranına ulaşmıştır 1, 2\. Ancak bazı karşılaştırmalı çalışmalarda, C3D ve R3D gibi modellerin, özellik çıkarımı yapan yöntemlere göre daha düşük performans (%74-75 civarı) gösterebildiği de not edilmiştir 3, 4\.  
* **ConvLSTM (Konvolüsyonel LSTM):** Bu model, video karelerindeki mekansal özellikleri öğrenirken aynı zamanda zaman içindeki değişimleri (yürüyüş ritmini) takip eder. Videoları "Lame" (Topal) ve "Non-Lame" (Topal Değil) klasörleri bazında ayırarak %85 civarında başarı sağlamıştır 1, 5\. Bu yöntem, çok aşamalı (multistage) boru hatlarına göre daha basit ve gerçek zamanlı uygulamaya uygundur 1\.

### 2\. Dizi Analizi ve Toplulaştırma (Sequence Modeling & Aggregation)

Bu yaklaşımda, videodaki her kare veya adım için bir veri üretilir, ancak karar videonun sonunda verilir. Klasör etiketi, bu zaman serisinin sonucunu belirler.

* **LSTM ve BiLSTM Kullanımı:** Model, videodan elde edilen adım büyüklüğü veya eklem koordinatları gibi verileri bir zaman serisi (dizi) olarak alır. Sisteme "bu video dizisi Topal sınıfına aittir" denilir. LSTM ağı, dizideki anormallikleri (örneğin adım atarkenki duraksamaları) öğrenerek videonun tamamı için bir karar üretir. Bir çalışmada **YOLOv3** ile bacak tespiti yapılıp **LSTM** ile sınıflandırma yapılarak %98.57 doğruluk elde edilmiştir 6, 7\.  
* **İstatistiksel Toplulaştırma (Aggregation):** Videodaki her kare için hesaplanan özellikler (sırt kavisi, baş yüksekliği vb.) tek bir istatistiksel değere (medyan, ortalama, maksimum veya 95\. persentil) indirgenir. Örneğin, bir videodaki tüm karelerin "ortalama sırt kavisi" hesaplanır ve bu tek değer, "Topal" klasöründeki etiketle eşleştirilerek **Random Forest** veya **SVM** gibi makine öğrenmesi algoritmalarına verilir 8-11. Bu yöntem, videodaki gürültülü (hatalı) karelerin etkisini azaltmak için etkilidir.

### 3\. Çok Modlu ve Hibrid Yaklaşımlar (Dikkat Mekanizmaları)

Klasör etiketinin genel durumu yansıttığı ancak videonun her anında topallığın belli olmadığı durumlarda, modelin "önemli anlara" odaklanması sağlanır.

* **Dikkat Mekanizmaları (Attention Mechanisms):** Model, videonun tamamına bakmak yerine, topallık belirtilerinin en belirgin olduğu (örneğin ineğin tam adım attığı veya başını indirdiği) karelere veya bölgelere daha fazla ağırlık vermeyi öğrenir. **DenseNet** mimarisine entegre edilen **CBAM (Convolutional Block Attention Module)**, modelin sağlıklı ve topal klasörleri arasındaki farkı ayırt etmek için hangi özelliklere (sırt veya ayak) odaklanması gerektiğini kendi kendine öğrenmesini sağlar 12, 13\.  
* **Çoklu Veri Füzyonu:** Bazı çalışmalarda, klasör seviyesindeki etiket (örneğin "Hasta İnek"), sadece görüntüyle değil, o ineğe ait sensör verileri (ivmeölçer) veya süt verimi verileriyle birleştirilerek modele verilir. **Random Forest** veya **Ensemble (Topluluk)** modelleri, bu farklı veri kaynaklarını birleştirerek klasör etiketini (Hasta/Sağlıklı) tahmin eder 14, 15\.

### Özet Yaklaşım Stratejisi

Eğer elinizde klasör bazlı (video seviyesinde) etiketlenmiş bir veri seti varsa, izlenen en yaygın ve başarılı yol şudur:

1. **Veri Artırma (Data Augmentation):** Klasörlerdeki video sayısını sanal olarak artırmak (döndürme, kırpma vb.) 1\.  
2. **Model Seçimi:** Özellik çıkarmayla uğraşmak istemiyorsanız **3D CNN** veya **ConvLSTM**; daha yüksek doğruluk ve açıklanabilirlik istiyorsanız özellik çıkarımı (pose estimation) sonrası **LSTM** veya **Random Forest** kullanmak 1, 2, 8\.  
3. **Doğrulama:** Modelin başarısını ölçerken, kare bazlı değil, video bazlı doğruluk (video-level accuracy) metriklerine bakmak.

