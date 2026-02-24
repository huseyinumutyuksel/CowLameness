Süt sığırlarında topallık tespiti için kullanılan yapay zeka yöntemleri, genellikle **bilgisayarlı görü (computer vision)** ve **giyilebilir sensör verilerinin analizi** olmak üzere iki ana kategoride toplanmaktadır. Bu sistemler, ineklerin yürüyüş ve duruş özelliklerini analiz ederek hastalıkları erken evrede teşhis etmeyi amaçlamaktadır.  
Kullanılan ana yapay zeka yöntemleri ve teknikleri şunlardır:

### 1\. Derin Öğrenme (Deep Learning) Tabanlı Görüntü İşleme

Bu yöntemler, video veya görüntülerden hayvanın tespiti, anahtar noktaların (eklemler, omurga vb.) belirlenmesi ve hareket analizi için kullanılır:

* **Evrişimli Sinir Ağları (CNN):** Görüntü verilerini işlemek için en yaygın kullanılan mimaridir. **LeNet**, **ResNet** (ResNet-50, ResNet-101), **MobileNetV2** ve **EfficientNetV2** gibi farklı CNN mimarileri, ineklerin özelliklerini çıkarmak ve sınıflandırmak için kullanılmaktadır 1-3. Ayrıca, 3 boyutlu verileri (video kareleri arasındaki zamansal ilişkiyi) işleyebilen **3D CNN** modelleri de doğrudan video sınıflandırması için kullanılmaktadır 4, 5\.  
* **Nesne Tespiti Modelleri:** İneklerin video karelerinde konumunu belirlemek ve onları arka plandan ayırmak için **YOLO (You Only Look Once)** ailesi (özellikle **YOLOv3**, **YOLOv4**, **YOLOv5**) sıklıkla tercih edilmektedir 6-11. Ayrıca **Mask R-CNN**, ineklerin tespiti ve duruş kestirimi (pose estimation) için kullanılan bir diğer güçlü algoritmadır 12-14.  
* **Poz Kestirimi ve Anahtar Nokta Takibi:** İneklerin omurga eğriliği, baş pozisyonu ve ayak basış noktaları gibi anatomik özelliklerini izlemek için **DeepLabCut**, **MMPose** ve **T-LEAP** gibi araçlar ve modeller kullanılmaktadır 3, 7, 15-19. Bu modeller, ineklerin eklem noktalarını işaretleyerek yürüyüş analizine olanak tanır.  
* **Segmentasyon Modelleri:** İneklerin vücut hatlarını ve uzuvlarını arka plandan net bir şekilde ayırmak için **Segment Anything Model (SAM)** gibi gelişmiş segmentasyon algoritmalarından yararlanılmaktadır 20, 21\.

### 2\. Zamansal ve Sıralı Veri Analizi Modelleri

İneklerin yürüyüşü zaman içinde değişen bir süreç olduğu için, bu dinamik verileri işlemek adına tekrarlayan sinir ağları kullanılır:

* **LSTM (Uzun Kısa Süreli Bellek):** Video kareleri veya sensörlerden gelen zaman serisi verilerindeki kronolojik desenleri öğrenmek için kullanılır. Özellikle **YOLOv3** veya **CNN** ile birlikte hibrid yapılar (örneğin **ConvLSTM**) oluşturularak, ineklerin adım atma özellikleri ve hareket dizileri analiz edilir 5, 6, 22-24.  
* **BiLSTM (Çift Yönlü LSTM):** Verileri hem ileri hem de geri yönde işleyerek zaman serisi sınıflandırmasında daha yüksek performans sağlayabilen bir yapıdır 24, 25\.  
* **InceptionTime:** Özellikle giyilebilir sensörlerden (ivmeölçer) gelen zaman serisi verilerini sınıflandırmak için kullanılan, çok ölçekli özellik çıkarımı yapabilen gelişmiş bir derin öğrenme modelidir. Bu model, **YOLOConv1D** gibi modüllerle hafifletilerek uç cihazlarda (edge devices) çalışabilecek hale getirilebilir 26, 27\.

### 3\. Makine Öğrenimi (Machine Learning) Algoritmaları

Görüntü işleme veya sensörlerden elde edilen sayısal verilerin (sırt kavisi açısı, adım uzunluğu, baş sallama genliği vb.) sınıflandırılmasında kullanılan klasik yöntemlerdir:

* **Destek Vektör Makineleri (SVM):** Topallık tespitinde inekleri "sağlıklı" veya "topal" olarak sınıflandırmak için yaygın olarak kullanılan ve yüksek doğruluk (%95 üzeri) sağlayabilen bir algoritmadır 28-31.  
* **Karar Ağaçları ve Rastgele Orman (Random Forest):** Birden fazla değişkenin (yürüyüş hızı, adım mesafesi vb.) analiz edilerek karar verilmesinde etkilidir. Rastgele Orman algoritmaları, özellikle sensör ve hava durumu verilerini birleştirerek topallık olaylarını tahmin etmede kullanılır 32, 33\.  
* **Diğer Algoritmalar:** **K-En Yakın Komşu (KNN)**, **AdaBoost**, **XGBoost** ve **CatBoost** gibi algoritmalar, çıkarılan özelliklerin sınıflandırılmasında ve özellik öneminin belirlenmesinde kullanılmaktadır 6, 13, 34, 35\.

### 4\. Gelişmiş ve Hibrid Yaklaşımlar

* **Dikkat Mekanizmaları (Attention Mechanisms):** **DenseNet** gibi ağlara entegre edilen **Konvolüsyonel Blok Dikkat Modülleri (CBAM)**, modelin yürüyüş veya sırt bölgesi gibi topallıkla daha ilişkili özelliklere odaklanmasını sağlayarak tespit doğruluğunu artırır 36, 37\.  
* **Veri Artırma (Data Augmentation):** Yeterli veri bulunmayan durumlarda, mevcut hareket verilerine sentetik varyasyonlar eklenerek veya sensör verileri simüle edilerek yapay zeka modellerinin genelleme yeteneği artırılmaktadır 1, 38, 39\.  
* **Uzman Sistemler:** Veteriner hekimler tarafından belirlenen kuralları (örneğin sırt kavisi eşik değerleri) makine öğrenimi çıktılarıyla birleştiren kural tabanlı sistemlerdir 40-42.

Bu teknolojiler, **CattleEye** gibi ticari sistemlerde güvenlik kameraları üzerinden otonom izleme yapmak veya **giyilebilir ivmeölçerler** aracılığıyla hayvanın aktivitesini takip etmek amacıyla sahada uygulanmaktadır 43, 44\.  
