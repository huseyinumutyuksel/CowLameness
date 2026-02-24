Kaynaklara göre, topallık tespitinde "zayıf denetimli öğrenme" (genellikle video seviyesinde etiketleme veya uçtan uca öğrenme olarak adlandırılır) **yüksek derecede güvenilir (%85-%90 aralığında)** sonuçlar vermektedir. Bu yöntemler, kare kare (frame-by-frame) detaylı işaretleme gerektirmeyen, doğrudan videonun sınıflandırıldığı modelleri kapsar ve geleneksel çok aşamalı yöntemlerle rekabet edebilir düzeydedir.  
Elde edilen veriler ışığında güvenilirlik düzeyi şu şekilde özetlenebilir:  
**1\. Doğruluk Oranları ve Performans**

* **3D CNN Modelleri:** Videoların "topal" veya "sağlıklı" olarak etiketlendiği veri setlerinde, 3D Konvolüsyonel Sinir Ağları (3D CNN) kullanılarak yapılan çalışmalarda **%90** oranında video seviyesinde sınıflandırma doğruluğu elde edilmiştir. Bu modellerin kesinlik (precision) ve duyarlılık (recall) oranları da %90-92 seviyelerindedir 1-3.  
* **ConvLSTM Modelleri:** Aynı veri setlerinde Konvolüsyonel LSTM (ConvLSTM) modelleri kullanıldığında doğruluk oranı **%85** olarak ölçülmüştür. 3D CNN modelleri, ConvLSTM'e göre daha yüksek performans göstermiştir 1, 3\.  
* **Ticari Sistemler (CattleEye):** Güvenlik kamerası görüntüleri üzerinden çalışan ve video seviyesinde işlem yapan ticari sistemlerin (CattleEye), uzman veterinerlerin değerlendirmeleriyle **%81.5 ile %86.3** arasında uyum (agreement) sağladığı raporlanmıştır 4, 5\.

**2\. Geleneksel Yöntemlerle Karşılaştırma**

* Zayıf denetimli yaklaşımlar (uçtan uca öğrenme), nesne tespiti ve poz kestirimi gibi ara adımları içeren çok aşamalı (multi-stage) yöntemlerle karşılaştırıldığında, **benzer doğruluk oranlarına** ulaşabilmektedir. Örneğin, bir çalışmada çok aşamalı makine öğrenimi modeli (özellik çıkarımı \+ sınıflandırma) %82.1 doğruluk sağlarken, uçtan uca derin öğrenme modeli %81.3 doğrulukla hemen hemen aynı performansı göstermiştir 6, 7\.  
* Bu yöntemlerin en büyük avantajı, hesaplama maliyeti yüksek olan poz kestirimi (pose estimation) ve nesne tespiti adımlarını atlayarak süreci basitleştirmesi ve gerçek zamanlı çiftlik uygulamaları için uygun hale getirmesidir 1, 2\.

**3\. Güvenilirliği Etkileyen Faktörler**

* **Veri Dengesizliği:** Modellerin güvenilirliği, eğitim verisindeki "sağlıklı" ve "topal" inek videolarının dengesine bağlıdır. Dengeli bir veri seti oluşturulduğunda %90 başarıya ulaşılırken 1, dengesiz veri setlerinde hassas ayarlar (feature engineering) yapan modellerin (örneğin Random Forest) daha kararlı sonuçlar (%97 üzeri) verebildiği görülmüştür 8, 9\.  
* **Yer Gerçeği (Ground Truth) Kalitesi:** Zayıf denetimli modeller, videoya verilen tek bir etiketi öğrendiği için, bu etiketin doğruluğu kritiktir. İnsan gözlemciler arasındaki uyumun bile bazen düşük (Kappa: 0.23-0.60) olması, modelin öğrenebileceği tavan başarıyı sınırlayabilir 10, 11\.

Sonuç olarak, zayıf denetimli öğrenme yöntemleri, detaylı anatomik işaretlemeye gerek duymadan **%85-90** bandında güvenilir sonuçlar üretebilmekte ve özellikle gerçek zamanlı izleme sistemleri için maliyet-etkin bir alternatif sunmaktadır.  
