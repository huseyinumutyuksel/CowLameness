proje_analiz_raporu_v1.md içerisindeki sorulara yanıt veriyorum.

1. Veteriner tarafından binary olarak klasör seviyesinde etiketlenen verileri kullanarak bir model eğitilecek. Elimizde ordinal veri elde etme durumu yok. Bu durumda binary veri üzerinden model eğitilecek.
2. Pose verisi kullanılacak. DLC ile elde edilen veri kullanılacak.
3. VideoMAE partial fine-tuning kullanılarak model eğitilecek.
4. Temporal Transformer kullanılacak MIL Attention kullanılmayacak.
5. versiyonlamaya 32 olarak devam edilecek ancak yapı sıfırdan oluşturulacak.
6. MMPose şimdilik kullanılmayacak. İleride belki kullanılabilecek. İleride dahil edilmesi halinde kod önceden buna uygun şekilde kodlanmış olmalı ki otomatik olarak seçim yapabilsin output arasında.
7. Hedeflenen Q1 dergiye Akademik makaledir. Bu kapsamda modelin doğruluk oranı %80 üzerinde olmalıdır. Ayrıca Q1 dergide sorulacak tüm sorulara yanıt olabilecek şekilde gerekli sonuçlar görsel ve tablolar halinde sunulmalıdır. Sunulamayan, makalede atlanan, anlatılmayan bir başarı anlamsızdır. 
