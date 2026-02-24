Elinizdeki videolarda birden fazla ineğin bulunması ve görüntülerin karmaşık olması (sağlıklı/topal karışık), standart bir analizden ziyade çok aşamalı ve gürültüye dayanıklı bir sistem kurmanızı gerektirir. Yeni eklenen kaynaklar ışığında, çoklu inek takibi ve detaylı yürüyüş analizi için önerilen güncel proje akışı ve teknolojiler şöyledir:

### 1\. Adım: Çoklu Nesne Tespiti ve Takibi (Multi-Object Detection & Tracking)

Videoda aynı anda birden fazla inek bulunduğu için, sistemi her bir ineği ayrı bir birey olarak tanıması ve video boyunca (diğer ineklerle çakışsa bile) kimliğini kaybetmeden takip etmesi üzerine kurmalısınız.

* **Teknoloji:** **Mask R-CNN** veya **YOLO (v5/v8)** ile birlikte **SORT (Simple Online and Realtime Tracking)** algoritması.  
* **Nasıl Kullanılır:**  
* Öncelikle **YOLOv5** veya **Mask R-CNN** kullanarak her karedeki inekleri tespit edip etrafına sınırlayıcı kutular (bounding box) çizersiniz 1-4.  
* Ardından, **SORT** algoritmasını entegre ederek, bir karede "İnek A" olarak etiketlenen hayvanın, sonraki karelerde de aynı kimlikle takip edilmesini sağlarsınız. Bu, ineklerin birbirinin önüne geçtiği (oklüzyon) durumlarda verinin karışmasını engeller 4, 5\.  
* Alternatif olarak, eğer kamera açınız uygunsa (tepeden bakış), ineklerin birbirini kapatmasını önlemek için **RGB-D (Derinlik)** kameralarından alınan verilerle çalışmak, kalabalık ahır ortamlarında oklüzyonu minimize eder 6, 7\. Ancak elinizdeki mevcut videolar standart (RGB) ise YOLO+SORT kombinasyonu en iyi seçenektir.

### 2\. Adım: Poz Kestirimi ve Anahtar Nokta Çıkarımı (Pose Estimation)

İneği takip ederken, topallığı teşhis edecek anatomik noktaları (eklemler, sırt hattı, baş) belirlemeniz gerekir.

* **Teknoloji:** **MMPose**, **DeepLabCut** veya **T-LEAP**.  
* **Nasıl Kullanılır:**  
* Tespit edilen her ineğin kutusu (bounding box) içinde **MMPose** veya **DeepLabCut** çalıştırarak 22 adede kadar anahtar nokta (toynaklar, dizler, kalça, omuz, boyun, baş, sırt omurları vb.) çıkarırsınız 2, 3, 8\.  
* Eğer videolarda ineklerin bacakları çitler veya diğer inekler tarafından sıkça kapanıyorsa, **T-LEAP** modelini tercih etmelisiniz. Bu model, zamansal bilgiyi (önceki kareleri) kullanarak görünmeyen (kapanan) uzuvların konumunu tahmin etmede daha başarılıdır 6, 9\.

### 3\. Adım: Biyomekanik Özellik Mühendisliği (Feature Engineering)

Ham koordinat verilerini, veteriner hekimlerin teşhis kriterlerine (Sprecher skoru gibi) uygun sayısal değerlere dönüştürmelisiniz. Literatür, bu aşamada "Temel Algoritmalar" (Base Algorithms) oluşturulmasını önerir.

1. **Hesaplanacak Kritik Özellikler:**  
2. **Sırt Kavisi (Spine Curvature):** İneğin boyun ve sırt omurları arasındaki açıyı veya mesafeyi hesaplayarak, sırtın ne kadar kamburlaştığını (kavis oranı) ölçmelisiniz. Topal ineklerde bu kavis belirgindir 2, 10, 11\.  
3. **Baş Pozisyonu (Head Bobbing):** Yürüyüş sırasında başın dikey salınımını analiz etmelisiniz. Topal inekler, ağrılı ayağa basarken başlarını belirgin şekilde aşağı indirip kaldırırlar 11, 12\.  
4. **Adım Mesafesi ve Takibi (Tracking Distance/Step Overlap):** Arka ayağın, ön ayağın bastığı yere ne kadar yakın bastığını ölçmelisiniz. Sağlıklı ineklerde arka ayak ön ayağın izine basar; topallarda bu mesafe (tracking distance) bozulur 13-15.  
5. **Hız ve Ritim:** Adım süreleri (stance/swing duration) ve yürüme hızı da ayırt edici özelliklerdir 15\.

### 4\. Adım: Sınıflandırma (Classification)

Çıkarılan özellikleri kullanarak ineğin topallık durumuna karar veren yapay zeka modelini eğitmelisiniz.

* **Teknoloji Seçenekleri:**  
* **Seçenek A (Makine Öğrenimi \- ML):** Eğer elinizdeki veri seti çok büyük değilse, **CatBoost**, **Random Forest** veya **XGBoost** algoritmalarını kullanmalısınız. Özellikle çoklu inek takibi yapılan çalışmalarda, çıkarılan özelliklerin (baş, boyun, sırt açıları) **CatBoost** ile sınıflandırılması %94 gibi yüksek doğruluk oranları vermiştir 4, 16\. Ayrıca Random Forest, özelliklerin önem derecesini belirlemede etkilidir 17\.  
* **Seçenek B (Derin Öğrenme \- DL):** Veri setiniz genişse ve özellik çıkarmayla uğraşmadan "uçtan uca" (end-to-end) bir çözüm istiyorsanız, **3D CNN** veya **ConvLSTM** kullanabilirsiniz. Bu modeller, videoyu doğrudan girdi olarak alıp zamansal ve mekansal özellikleri kendi öğrenerek %90 civarında doğrulukla "Topal/Sağlam" kararı verebilir 18, 19\.  
* **Seçenek C (Hibrid \- Dikkat Mekanizmaları):** **DenseNet121** modeline **CBAM (Convolutional Block Attention Module)** ekleyerek, ağın hafif topallıkta bacak hareketlerine, şiddetli topallıkta ise sırt kavislenmesine "dikkat etmesini" (attention) sağlayabilirsiniz. Bu yöntem, sınıflandırma başarısını %99'lara kadar çıkarabilmektedir 20\.

### 5\. Adım: Doğrulama ve Performans Ölçümü

Modelinizin başarısını ölçerken, veterinerler arasındaki görüş ayrılıklarını da hesaba katan metrikler kullanmalısınız.

* **Yöntem:**  
* **İkili Sınıflandırma (Binary):** İnekleri sadece "Sağlıklı (Skor 1-3)" ve "Topal (Skor 4-5)" olarak ayırmak, saha kullanımı için genellikle yeterli ve daha yüksek doğruluklu (%82-%87) bir yaklaşımdır 2, 17, 21\.  
* **Esnek Doğruluk (Relaxed Accuracy):** 1-5 arası skorlama yapacaksanız, modelin tahmininin uzman skorundan ±1 puan sapmasını kabul eden "Esnek Doğruluk" metriğini kullanın. Bu, modelin saha performansını daha gerçekçi yansıtır 21, 22\.

**Özet Proje Akışı:**Video \-\> YOLOv5 \+ SORT (Çoklu İnek Takibi) \-\> MMPose/DeepLabCut (Anahtar Nokta) \-\> Matematiksel Analiz (Sırt Kavisi, Baş Sallama, Adım Mesafesi) \-\> CatBoost veya Random Forest (Sınıflandırma) \-\> Sonuç: Sağlıklı/Topal  
