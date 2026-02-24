Etiketlenmemiş veri setlerinde ve aynı karede birden fazla ineğin bulunduğu karmaşık senaryolarda, klasik denetimli öğrenme (supervised learning) yöntemleri yetersiz kalır. Bu tür bir durumda, **"İnek Yapısal Modeli" (Cow Structural Model)** tabanlı nesne ayrıştırması ve **"Sentetik Veri Üretimi"** ile desteklenen bir **denetimsiz/yarı denetimsiz (unsupervised/semi-supervised)** öğrenme hattı kurmanız gerekir.  
Kaynaklara dayanarak, bu zorlu senaryo için uygulayabileceğiniz çalışmanın teknik detayları adım adım aşağıdadır:

### 1\. Adım: Çoklu İnek Ayrıştırması ve Yapısal Modelleme (Multi-Cow Separation)

Veri setinizde etiket olmadığı için, sistemin öncelikle görüntüdeki piksellerin hangi ineğe ait olduğunu "kendi kendine" öğrenmesi veya önceden eğitilmiş genel modelleri kullanması gerekir.

1. **Yöntem:** **Uzamsal Kümeleme (Spatial Clustering) ve CNN Füzyonu**Liu ve arkadaşlarının geliştirdiği yöntem, etiketlenmemiş çoklu inek videoları için ideal bir referanstır. Bu yöntemde, tek bir ineği tespit etmek yerine, görüntüdeki tüm potansiyel "vücut parçaları" (eklemler, baş, sırt) önce bağımsız olarak bulunur, sonra bu parçalar matematiksel kısıtlamalarla kümelenerek bireysel inekler oluşturulur 1, 2\.  
2. **Uygulama Detayı:**  
3. **Çift CNN Mimarisi:** İki ayrı Konvolüsyonel Sinir Ağı (CNN) kullanın. Birincisi (CNN1) renkli (RGB) görüntüyü işleyerek ineğin görsel özelliklerini (renk, doku) öğrenir. İkincisi (CNN2), ardışık kareler arasındaki farkı (frame difference) alarak "hareket eden" bölgeleri, yani yürüyen inekleri tespit eder. Bu, sabit duran çitler veya arka plan gürültüsünü eler 3, 4\.  
4. **Güven Haritaları (Confidence Maps):** Her iki ağ, görüntü üzerinde belirli vücut parçalarının (örneğin sol ön ayak, boyun) bulunma olasılığını gösteren "güven haritaları" üretir 3\.  
5. **Uzamsal Kümeleme (Post-Processing):** Burası çoklu inek probleminin çözüldüğü yerdir. Algoritma, tespit edilen tüm vücut parçalarını alır ve bunları **Mean-Shift Clustering** (Ortalama Kaydırma Kümeleme) algoritması ile gruplar. Bir ineğin anatomik yapısı (başın boyuna, boynun sırta olan mesafesi) matematiksel bir kısıtlama (constraint) olarak tanımlanır. Eğer bir baş ve bir ayak, anatomik olarak imkansız bir mesafedeyse, algoritma bunların farklı ineklere ait olduğunu anlar ve onları ayrı nesneler (İnek A, İnek B) olarak etiketler 5, 6\.

### 2\. Adım: Etiketsiz Ortamda Segmentasyon (Zero-Shot Segmentation)

Elinizde ineklerin nerede olduğuna dair "bounding box" (sınırlayıcı kutu) etiketleri yoksa, önceden eğitilmiş devasa modellerden (Foundation Models) yararlanmalısınız.

* **Teknoloji:** **Segment Anything Model (SAM)**Kang ve arkadaşlarının kullandığı gibi, **SAM** modeli, herhangi bir ek eğitim gerektirmeden (zero-shot), görüntüdeki nesneleri segmentlere ayırabilir.  
* **Uygulama:**SAM modeline videonuzu verirsiniz. Model, görüntüdeki "inek benzeri" formları otomatik olarak maskeler ve arka plandan ayırır. Bu işlem, etiketleme yapmadan ineğin siluetini çıkarmanızı sağlar. Elde edilen bu siluet (maske), daha sonra topallık analizi (sırt kavisi ölçümü) için kullanılır 7, 8\.

### 3\. Adım: Veri Artırma ve Sentetik Veri Üretimi (Data Augmentation)

Etiketli veriniz (yani hangi ineğin topal olduğuna dair bilgi) olmadığı veya çok az olduğu için, elinizdeki sınırlı hareket verisini çoğaltarak modeli eğitmeniz gerekir.

1. **Yöntem:** **Sentetik Kinematik Veri Üretimi**Karoui ve arkadaşlarının önerdiği yöntem, sınırlı veri setlerinde başarıyı %90'ın üzerine çıkarmaktadır.  
2. **Uygulama:**  
3. Elde ettiğiniz az sayıdaki hareket eğrisine (bir ineğin adım atarken ekleminin çizdiği grafik) rastgele gürültü (random noise) eklersiniz.  
4. Bu gürültü oranları %1, %2, %5, %10 gibi kademeli artırılır.  
5. Ardından bu veriler bir **Medyan Filtresi (Median Filter)** ile pürüzsüzleştirilir.  
6. Böylece, aslında elinizde olmayan "farklı topallık varyasyonlarını" yapay olarak üretmiş olursunuz. Bu sentetik verilerin gerçekten bir ineğin hareketini temsil edip etmediğini doğrulamak için **K-Means Kümeleme** kullanırsınız. Eğer sentetik veri, orijinal veri kümesine yakın bir kümede yer alıyorsa, eğitim verisi olarak kabul edilir 9-11.

### 4\. Adım: Denetimsiz Başarı Ölçümü (Unsupervised Evaluation)

Elinizde "doğru cevap anahtarı" (ground truth) olmadığı için modelin başarısını klasik doğruluk (accuracy) oranıyla ölçemezsiniz. Bunun yerine Liu ve arkadaşları tarafından önerilen **Denetimsiz Metrikleri** kullanmalısınız 12, 13:

1. **Geçerli İnek Yüzdesi (Valid Cow Percentage \- VCP):** Modelin tespit ettiği anahtar noktaların (baş, sırt, ayaklar) oluşturduğu şekil, anatomik olarak bir ineğe benziyor mu? Örneğin, baş kuyruktan daha aşağıda veya bacaklar sırttan yukarıda olamaz. Model bu kurallara uyan bir iskelet çıkarıyorsa "başarılı" sayılır.  
2. **Zamansal Tutarlılık (Temporal Consistency \- TC):** Videoda bir karede tespit edilen inek, bir sonraki karede aniden kayboluyor veya şekli bozuluyor mu? İneğin hareket yörüngesinin pürüzsüzlüğü (smoothness) ölçülür. Düşük TC değeri (daha az titreme), modelin kararlı çalıştığını gösterir 14\.

### 5\. Adım: Anomali Tespiti ile Sınıflandırma

Modeliniz inekleri ayırıp hareketlerini takip ettikten sonra, hangisinin topal olduğuna karar vermek için "Anomali Tespiti" yaklaşımını kullanırsınız.

* **Mantık:** Sürüdeki ineklerin büyük çoğunluğu sağlıklıdır. Bu nedenle, sürü ortalamasından sapan (outlier) hareketler "topallık" olarak işaretlenir.  
* **Uygulama:** Çıkarılan özellikler (adım hızı, sırt kavisi, baş sallama) **K-Means** veya **Isolation Forest** gibi denetimsiz algoritmalarla kümelenir. Büyük küme "Sağlıklı", ondan uzak kalan küçük kümeler "Topal" (Anormal) olarak sınıflandırılır.

**Özet İş Akışı:**

1. **Giriş:** Çoklu inek içeren ham video.  
2. **Segmentasyon:** SAM veya YOLOv5 ile inekleri bul (Etiketsiz).  
3. **İzleme:** Mean-Shift Clustering ile parçaları birleştirip "İnek A", "İnek B"yi oluştur.  
4. **Özellik Çıkarımı:** Sırt kavisi ve adım mesafesini hesapla.  
5. **Veri Zenginleştirme:** Sentetik veri ile hareket varyasyonlarını artır.  
6. **Karar:** K-Means ile sürü ortalamasından sapanları "Topal" olarak işaretle.  
7. **Kontrol:** VCP ve TC metrikleri ile modelin mantıklı çalışıp çalışmadığını denetle.

