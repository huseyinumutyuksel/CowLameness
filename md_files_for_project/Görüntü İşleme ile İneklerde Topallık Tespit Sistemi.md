Elinizdeki videoların birden fazla ineği içermesi ve hem sağlıklı hem de topal hayvanların karışık bulunması nedeniyle, geliştireceğiniz proje çok aşamalı bir "bilgisayarlı görü" (computer vision) boru hattı (pipeline) gerektirmektedir. Kaynaklara dayanarak, bu projeyi baştan sona şu teknolojik sıralama ile geliştirebilirsiniz:

### 1\. Adım: Nesne Tespiti ve Takibi (Object Detection & Tracking)

Videolardaki çoklu inek karmaşasını çözmek ve her bir ineği bireysel olarak analiz edebilmek için öncelikle hayvanları tespit edip takip etmeniz gerekir.

* **Teknoloji:** **YOLO (You Only Look Once)** mimarisi, özellikle **YOLOv5** veya **YOLOv8**, ineklerin video karelerinde tespiti (bounding box oluşturma) için en uygun teknolojidir 1-3. Alternatif olarak **Mask R-CNN** de ineklerin tespiti ve bölütlenmesi (segmentation) için kullanılmaktadır 4, 5\.  
* **Uygulama Şekli:** Videoyu karelere (frame) böldükten sonra, her bir karedeki inekleri tespit etmek için YOLOv5 modelini kullanırsınız. Her ineği video boyunca ayırt edebilmek ve diğer ineklerden veya engellerden kaynaklanan geçici kapanmaları (occlusion) yönetebilmek için bir takip algoritması (örneğin **SORT** veya IoU tabanlı takip) entegre etmelisiniz 4-6. Bu aşama, videodaki kalabalığı filtreleyerek analizi "tekil inek geçişlerine" indirgemenizi sağlar 2\.

### 2\. Adım: Poz Kestirimi ve Anahtar Nokta Çıkarımı (Pose Estimation)

İneği tespit ettikten sonra, topallığı belirleyen anatomik noktaların (ayaklar, sırt, baş) koordinatlarını çıkarmanız gerekir.

* **Teknoloji:** **MMPose**, **DeepLabCut (DLC)** veya **T-LEAP** kütüphaneleri.  
* **MMPose:** **HRNet** ve **DarkPose** omurgalarını kullanarak inek üzerindeki 22 anahtar noktayı (göz, boyun, omurga noktaları, tırnaklar, eklemler vb.) yüksek doğrulukla tespit etmek için kullanılır 7-9.  
* **DeepLabCut:** İneklerin yürüyüşü sırasındaki nirengi noktalarını (örneğin baş, sırt ve toynaklar) izlemek için eğitilebilir ve hareket analizinde etkilidir 10, 11\.  
* **T-LEAP:** Videolardaki zamansal bilgiyi kullanarak, çit gibi engellerin arkasında kalan ineklerin duruşunu tahmin etmede (occlusion-robust) başarılıdır 12\.  
* **Uygulama Şekli:** Seçtiğiniz modeli (örneğin MMPose), tespit edilen ineklerin sınırlayıcı kutuları (bounding box) içinde çalıştırarak her kare için (x, y) koordinatlarını üretirsiniz. Bu noktalar; toynakların yere basışını, sırtın kavisini ve başın konumunu temsil eder 7, 13\.

### 3\. Adım: Biyomekanik Özellik Çıkarımı (Feature Extraction)

Ham koordinat verilerini, veteriner hekimlerin topallık teşhisinde kullandığı anlamlı verilere dönüştürmeniz gerekir.

1. **Teknoloji:** Matematiksel ve geometrik algoritmalar ("Temel Algoritmalar").  
2. **Uygulama Şekli:** Aşağıdaki üç ana özelliği hesaplayan algoritmalar yazmalısınız:  
3. **Sırt Kavisi (Spine Curvature):** İneğin boyun ve sırt omurları arasındaki açıyı veya mesafeyi hesaplayarak sırtın ne kadar kamburlaştığını ölçmelisiniz. Sağlıklı ineklerde sırt düzdür, topallık arttıkça kavis artar 14, 15\.  
4. **Baş Pozisyonu (Head Bobbing):** İneğin yürürken başını ne kadar aşağı yukarı salladığını analiz etmelisiniz. Şiddetli topallıkta, hasta ayağa basıldığında baş belirgin şekilde aşağı iner 16, 17\.  
5. **Adım Mesafesi ve Takibi (Tracking Distance/Step Overlap):** Arka ayağın, ön ayağın bastığı yere ne kadar yakın bastığını ölçmelisiniz. Sağlıklı ineklerde arka ayak, ön ayağın izine basar; topal ineklerde bu mesafe açılır 18, 19\.  
6. *Alternatif İleri Yöntem:* **Cow Lameness Feature Maps (CLFM)** tekniğini kullanarak, toynak yörüngelerini ve sırt konturunu tek bir özellik haritasında birleştirip görüntü olarak işleyebilirsiniz 20, 21\.

### 4\. Adım: Sınıflandırma ve Karar Verme (Classification)

Çıkarılan özellikleri kullanarak ineğin "Sağlıklı", "Hafif Topal" veya "Şiddetli Topal" olduğuna karar veren yapıyı kurmalısınız.

* **Teknoloji:**  
* **Makine Öğrenimi (ML):** **Destek Vektör Makineleri (SVM)**, **Rastgele Orman (Random Forest)** veya **XGBoost**. Bu algoritmalar, çıkarılan sayısal özellikleri (sırt açısı, adım mesafesi vb.) sınıflandırmada yüksek başarı (özellikle SVM ve Random Forest) göstermektedir 22-24.  
* **Derin Öğrenme (DL):** **LSTM (Uzun Kısa Süreli Bellek)** veya **DenseNet**. Eğer veriyi zaman serisi olarak işleyecekseniz (hareketin zaman içindeki değişimi), LSTM veya BiLSTM ağlarını kullanabilirsiniz 25, 26\. Eğer CLFM gibi özellik haritaları (görüntü) oluşturduysanız, **DenseNet** ve **Dikkat Mekanizmaları (Attention Modules)** kullanarak özelliklerin önem derecesine göre sınıflandırma yapabilirsiniz 27, 28\.  
* **Uygulama Şekli:**  
* *Senaryo A (Basit ve Etkili):* Özellikleri bir vektörde toplayıp **XGBoost** veya **SVM** ile sınıflandırma yaparak ikili (Sağlıklı/Topal) veya çoklu (1-5 arası skor) sonuç üretebilirsiniz 22, 29\.  
* *Senaryo B (Gelişmiş):* **DenseNet121** modeline **CBAM (Convolutional Block Attention Module)** entegre ederek, modelin hafif topallıkta ayak hareketlerine, şiddetli topallıkta ise sırt kavislenmesine daha fazla odaklanmasını sağlayabilirsiniz 28\.

### 5\. Adım: Doğrulama ve Raporlama

Sonuçlarınızı veteriner hekimlerin kullandığı skorlama sistemleriyle (örneğin 1-5 arası Sprecher skalası veya Zinpro) eşleştirerek sunmalısınız 30-32. Modelinizin başarısını ölçerken, "Tam Doğruluk" (Exact Match) yerine, 1 puanlık sapmaları kabul eden **"Esnek Doğruluk" (Relaxed Accuracy)** metriğini kullanmanız önerilir, çünkü uzmanlar arasında bile skorlama tutarlılığı %100 değildir 33-35.  
**Özet Akış Şeması:**Video Girişi \-\> YOLOv5 (İnek Tespiti & Takibi) \-\> MMPose/DeepLabCut (Anahtar Nokta Çıkarımı) \-\> Geometrik Analiz (Sırt, Baş, Ayak Verileri) \-\> SVM/XGBoost veya DenseNet (Sınıflandırma) \-\> Topallık Skoru (1-5)  
