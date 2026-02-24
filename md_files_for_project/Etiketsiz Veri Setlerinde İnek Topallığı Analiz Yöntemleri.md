İnek bazlı "topal" veya "sağlıklı" etiketlemesinin (ground truth) bulunmadığı veya yetersiz olduğu veri setlerinde, araştırmacılar genellikle **denetimsiz öğrenme (unsupervised learning)** tekniklerine, **bireyselleştirilmiş anomali tespitine** ve **yapısal/fiziksel kısıtlamalara** dayalı yöntemlere başvurmuşlardır.  
Kaynaklara göre etiketsiz veri setlerinde yürütülen çalışmalar şu başlıklar altında toplanmaktadır:

### 1\. Denetimsiz Performans Metrikleri ve Yapısal Modeller

Etiketlemenin çok zahmetli olduğu video analizlerinde, modelin başarısını ölçmek için insan etiketleri yerine "önsel bilgiye" (prior knowledge) dayalı denetimsiz metrikler geliştirilmiştir.

* **Geçerli İnek Yüzdesi (Valid Cow Percentage \- VCP):** Bu yöntem, modelin tespit ettiği anahtar noktaların (eklemler, omurga vb.) uzaysal konumlarının gerçekten bir inek şekli oluşturup oluşturmadığını kontrol eder. Etiket olmasa bile, noktalar anatomik olarak mantıksız bir şekil oluşturuyorsa model başarısız kabul edilir 1, 2\.  
* **Zamansal Tutarlılık (Temporal Consistency \- TC):** İneğin yürüyüşü sırasında vücut şeklinin stabil kalması ve anahtar noktaların pürüzsüz yörüngeler izlemesi gerektiği varsayımına dayanır. Etiket kullanılmadan, sadece video kareleri arasındaki hareketin fiziksel tutarlılığı ölçülerek modelin başarısı değerlendirilir 1, 2\.

### 2\. Bireyselleştirilmiş "Sağlıklı Referans" Oluşturma (Anomali Tespiti)

Bu yaklaşımda, ineğin "sağlıklı" veya "topal" olarak etiketlenmesine ihtiyaç duyulmaz; bunun yerine her inek kendi kendisinin kontrol grubu olarak kullanılır.

* **Tarihsel Veri Analizi:** Her ineğin geçmiş verileri kullanılarak o hayvana özgü bir "sağlıklı referans" değeri hesaplanır. Örneğin, bir çalışmada ineğin geçmişteki **en düşük %5'lik sırt duruşu değerlerinin ortalaması** "sağlıklı referans" olarak kabul edilmiştir 3\.  
* **Sapma Tespiti (Deviation Detection):** Gerçek zamanlı veriler, bu kişisel referans değeriyle karşılaştırılır. Eğer güncel veri, belirlenen bir eşik değerin (örneğin referansın belirli bir katı) üzerine çıkarsa, sistem ineği otomatik olarak "anormal" (topal) olarak sınıflandırır 4, 5\. Bu yöntem, genel bir "topal" etiketi yerine, ineğin kendi normalinden sapmasını izlediği için etiketli veri setine olan bağımlılığı azaltır 6\.

### 3\. Kümeleme (Clustering) Yöntemleri

Veri setindeki inekleri önceden tanımlanmış etiketlere göre değil, verilerin kendi içindeki benzerliklerine göre gruplandıran denetimsiz makine öğrenimi algoritmaları kullanılır.

* **K-Ortalamalar (K-Means) Kümeleme:** İvmeölçer verilerinin analizinde, veriler zaman pencerelerine bölünerek özellikler çıkarılır ve K-Means algoritması ile gruplandırılır. Bu gruplar daha sonra aktivite yoğunluğuna (yüksek, orta, düşük) göre etiketlenir 7\.  
* **Davranışsal Gruplandırma:** Başka bir çalışmada, ineklerin yürüyüş verileri ve lezyon kayıtları kullanılarak kümeleme analizi yapılmış ve inekler "Cluster 1, 2, 3" şeklinde objektif gruplara ayrılmıştır. Bu kümeler daha sonra geriye dönük olarak lokomosyon skorlarıyla (LS1, LS2, LS3) karşılaştırıldığında yüksek uyum (%79-83) gösterdiği görülmüştür 8\. Bu, etiketleme yapılmadan da verinin doğal yapısının topallık seviyelerini ayırt edebildiğini göstermektedir.

### 4\. Sentetik Veri Üretimi (Data Augmentation)

Etiketli verinin az olduğu durumlarda, mevcut hareket verilerine yapay varyasyonlar eklenerek veri seti genişletilir.

* **Gürültü Ekleme:** Mevcut yürüme eğrilerine %1 ile %20 arasında rastgele gürültü eklenerek sentetik hareket örnekleri oluşturulur. Bu sentetik verilerin, orijinal ineklerin hareket özelliklerini koruyup korumadığı, yine denetimsiz bir yöntem olan **K-Means kümeleme** ve siluet indeksi hesaplamasıyla doğrulanır 9, 10\.

Özetle, etiketsiz veri setlerinde çalışmalar; **ineğin kendi geçmişini referans alma**, **verileri benzerliklerine göre kümeleme** ve **anatomik/fiziksel kurallara uygunluğu denetleyen metrikler kullanma** stratejileri üzerine kuruludur.  
