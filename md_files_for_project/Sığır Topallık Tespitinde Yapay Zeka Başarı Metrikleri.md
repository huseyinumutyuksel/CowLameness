Akademik çalışmalarda, süt sığırlarında topallık tespiti yapan yapay zeka modellerinin performansını ölçmek için hem standart sınıflandırma metrikleri hem de problemin doğasına özgü (sınıf dengesizliği ve gözlemci öznelliği gibi) gelişmiş kriterler kullanılmaktadır.  
Değerlendirmeye alınan temel başarı kriterleri şunlardır:

### 1\. Temel Sınıflandırma Metrikleri

Modellerin "Sağlıklı" ve "Topal" ayrımını ne kadar doğru yaptığını ölçmek için en yaygın kullanılan metriklerdir:

* **Doğruluk (Accuracy \- ACC):** Modelin doğru tahmin ettiği (hem sağlıklı hem topal) ineklerin toplam inek sayısına oranıdır. Birçok çalışma, ikili sınıflandırmada (sağlıklı/topal) %80 ile %99 arasında değişen doğruluk oranlarını başarı ölçütü olarak kullanmıştır 1-4.  
* **Duyarlılık (Sensitivity/Recall):** Gerçekten topal olan ineklerin ne kadarının model tarafından tespit edilebildiğini gösterir. Topallık, gözden kaçırılmaması gereken ağrılı bir durum olduğu için bu metrik, yanlış negatifleri (topal olduğu halde sağlıklı denilenleri) en aza indirmek adına kritiktir. Bazı çalışmalarda %100'e varan duyarlılık değerlerine ulaşılmıştır 5-9.  
* **Özgüllük (Specificity):** Sağlıklı ineklerin ne kadar doğru bir şekilde sağlıklı olarak sınıflandırıldığını gösterir. Yanlış alarmları (sağlıklı ineğe topal denmesi) ve gereksiz tedavi maliyetlerini önlemek için önemlidir. Yüksek özgüllük (%99 üzeri), sistemin çiftçiler tarafından kabul görmesi için bir kriter olarak önerilmektedir 5-9.  
* **Kesinlik (Precision \- PPV):** Modelin "topal" olarak işaretlediği ineklerin ne kadarının gerçekten topal olduğunu ifade eder 3, 7, 10\.  
* **F1-Skoru (F1-Score):** Kesinlik ve Duyarlılık değerlerinin harmonik ortalamasıdır. Veri setinde sağlıklı ve topal inek sayıları dengesiz olduğunda (örneğin sağlıklı inek sayısı çok daha fazlaysa) modelin gerçek başarısını göstermek için sadece "Doğruluk" yerine F1-Skoru veya Macro-F1 tercih edilmektedir 1, 7, 11, 12\.

### 2\. İleri Seviye ve Dengesiz Veri Metrikleri

Topallık verilerinin dengesiz doğası (sürüde sağlıklı ineklerin çoğunlukta olması) nedeniyle kullanılan daha sağlam metriklerdir:

* **Matthews Korelasyon Katsayısı (MCC):** Dengesiz veri setlerinde modelin başarısını ölçmek için F1 skorundan daha güvenilir kabul edilen bir metriktir. Özellikle biyokimyasal ve çok modlu (multimodal) veri setleriyle çalışan araştırmalarda, modelin rastgele tahmin yapıp yapmadığını anlamak için kullanılmıştır. \-1 ile \+1 arasında değer alır; \+1 mükemmel tahmini gösterir 13-15.  
* **ROC Eğrisi Altındaki Alan (AUC \- Area Under Curve):** Modelin farklı eşik değerlerinde sınıflandırma yapabilme yeteneğini ölçer. AUC değeri 1'e yaklaştıkça modelin ayırt etme gücünün arttığı kabul edilir 13, 16-18.

### 3\. Derecelendirme ve Hata Payı Metrikleri

Topallığın sadece "var/yok" şeklinde değil, 1-5 veya 1-7 arası skorlarla derecelendirildiği çalışmalarda kullanılan kriterlerdir:

* **Esnek Doğruluk (Relaxed Accuracy):** Topallık skorlamasında uzman veterinerler arasında bile tam uyuşma (örneğin herkesin 3 vermesi) zordur. Bu nedenle, modelin tahmininin uzman skorundan ±1 puan sapmasını (örneğin uzman 3 dediğinde modelin 2 veya 4 demesi) "doğru" kabul eden bu metrik geliştirilmiştir. Bu, modelin saha performansını daha gerçekçi yansıtır 10, 19-22.  
* **Ortalama Karesel Hata (MSE \- Mean Squared Error):** Modelin tahmin ettiği skor ile gerçek skor arasındaki farkın karesini alarak ölçer. Skorlama (regresyon) modellerinde hata payını minimize etmek için kullanılır. Örneğin, bir çalışmada makine öğrenimi modelleri 1.631 MSE değeri ile en düşük hata oranına ulaşmıştır 10, 23\.

### 4\. Güvenilirlik ve Uyum Metrikleri

Modelin insan gözlemcilerle veya "altın standart" ile ne kadar uyumlu olduğunu ölçen istatistiksel testlerdir:

* **Kappa Katsayısı (Cohen's Kappa):** Model ile uzman veterinerlerin kararları arasındaki uyumun, şans faktöründen arındırılmış halidir. Değer 0.60 ve üzeri olduğunda "önemli düzeyde uyum" kabul edilir. Ancak çalışmalarda uzmanlar arasındaki uyumun bile bazen 0.23-0.60 aralığında kalabildiği görülmüştür 10, 24-27.  
* **Gwet’s AC1:** Kappa katsayısına alternatif olarak, özellikle dengesiz veri setlerinde (topallık prevalansının düşük olduğu durumlar) değerlendiriciler arası uyumu ölçmek için kullanılmıştır 26, 28\.

### 5\. Bilgisayarlı Görüye Özgü Metrikler

Görüntü işleme modellerinin (Pose Estimation) başarısını ölçmek için kullanılan teknik kriterlerdir:

* **PCK (Percentage of Correct Keypoints):** Modelin ineğin vücudundaki anahtar noktaları (eklem yerleri, toynaklar vb.) ne kadar doğru tespit ettiğini ölçer. Örneğin, bir çalışmada PCK@0.05 değeri %100 olarak raporlanmıştır, bu da modelin belirlenen eşik değer içinde noktaları hatasız bulduğunu gösterir 29, 30\.  
* **mAP (mean Average Precision):** Nesne tespiti (örneğin ineğin videoda bulunması) modellerinin genel hassasiyetini ölçmek için kullanılır 31, 32\.

