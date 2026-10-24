Kredi Kartı Dolandırıcılığı Tespiti (Credit Card Fraud Detection)

Bu proje, son derece dengesiz finansal veri setleri üzerinde makine öğrenmesi modelleri kullanarak şüpheli işlemleri otomatik olarak tespit etmek ve banka kayıplarını minimize etmek amacıyla geliştirilmiştir.

--------------------------------
1. İş Problemi (Business Problem)

Günümüz dijital bankacılığında kredi kartı sahtekarlığı (fraud) finans kurumları için büyük bir maliyet kalemidir. 

Temel Amaç: Şüpheli işlemleri anında tespit edip engellemek; ancak bunu yaparken gerçek müşterilerin kartlarını yanlışlıkla bloke ederek müşteri memnuniyetsizliği yaratmamaktır.
Maliyet Dengesi: Yanlış alarmlar ile kaçırılan dolandırıcılık vakaları arasında optimum bir denge kurmak gerekir.

-----------------------------------------------------
2. Verinin Zorluğu: Dengesiz Veri (The Challenge of Imbalanced Data)

Veri setindeki sınıflar arasında ciddi bir dengesizlik bulunmaktadır:
Normal İşlem (0): 284,315
Dolandırıcılık İşlemi (1): 492 (%0.17)

Kritik Eşik Vurgusu: Eğer model hiçbir şey öğrenmeyip gelen her işleme ezbere "Normal" derse, zaten %99.8 oranında Accuracy (Doğruluk) bilecektir. Ancak bu durum bankayı büyük bir zarara sokar. Bu yüzden dengesiz veri setlerinde Accuracy yanıltıcı bir metriktir.

----------------------------------------------------

 3. Metodoloji ve Stratejiler

Modelin ezberleme (`overfitting`) yapmasını önlemek ve veriyi adil bir şekilde işlemek için şu adımlar izlenmiştir:

 Eğitim / Test Ayrımı ve Veri Sızıntısı (Data Leakage) Önleme
 Modelin daha önce hiç görmediği yeni işlemlerde başarılı olabilmesi için veri seti %80 Eğitim (Train) ve %20 Test olarak ayrılmıştır.
Veri Sızıntısını Önlemek İçin: Ölçeklendirici (`Scaler`) sadece eğitim seti üzerinde `fit` (öğrenme) edilmiş, test setine ise yalnızca `transform` (uygulama) yapılmıştır. Tıpkı bir öğrenciye sınavdan önce soruların ipucunu vermemek gibi, test verisi model için tamamen sürpriz bırakılmıştır.

 Özellik Ölçeklendirme (Feature Scaling)
Kredi kartı özelliklerinin matematiksel boyutları birbirinden çok farklıdır (örneğin işlem tutarı `Amount` kolonu 0 ile 25,000$ arasında değişirken, şifrelenmiş değişkenler farklı aralıktadır). 
Eğer veriler aynı matematiksel ölçeğe çekilmeseydi, model sadece sayısal olarak büyük olduğu için `Amount` değişkenine gereğinden fazla önem verirdi. `StandardScaler` kullanılarak tüm verilerin ortalaması 0, standart sapması 1 olacak şekilde eşit ağırlığa getirilmesi sağlanmıştır.

-----------------------------------------

4. Metriklerin İş Mantığı & Değerlendirme

Finansal modellemede hata türlerinin maliyeti birbirinden çok farklıdır:

FN (False Negative - Kaçırılan Dolandırıcılık): Modelin dolandırıcılık işlemine "Normal" demesidir. Tüm para dolandırıcının eline geçer. Bankaya maliyeti en yüksek olan ve minimuma indirmek istediğimiz en tehlikeli hata türüdür.
FP (False Positive - Yanlış Alarm): Modelin normal bir işleme "Dolandırıcılık" diyerek müşterinin kartını bloke etmesidir. Müşteri memnuniyetsizliği yaratır, operasyonel yük oluşturur ancak doğrudan anlık para kaybına yol açmaz.

Temel Performans Metrikleri:
1. Recall (Duyarlılık):Gerçek dolandırıcılıkların yüzde kaçını yakalayabildiğimizi gösterir. Veri dengesizliği olan finansal senaryolarda odaklanılması gereken en kritik metriktir.
2. Precision (Hassasiyet): Modelin "Dolandırıcılık" dediklerinin yüzde kaçı gerçekten suçludur? Yanlış alarmları en aza indirmeyi amaçlar.
3. F1-Score: Precision ve Recall metriklerinin dengeli birleşimini sunar.
4. Accuracy (Doğruluk): Genel başarıyı gösterir ancak dengesiz verilerde yanıltıcıdır.
5. Macro / Weighted Avg: Sınıf dağılımlarının ortalamaya etkilerini dengeler veya ağırlıklandırır.


-------------Detaylı Rapor Karşılaştırması--------
Lojistik Regresyon Sonuçları:

Confusion Matrix: [[56851, 13], [36, 62]] 
98 gerçek dolandırıcılığın 62 tanesini yakalarken, 36 tanesini kaçırmıştır (FN = 36).
13 masum müşterinin kartına yanlış alarm vermiştir (FP = 13). 
Fraud Sınıfı Skorları: Precision: 0.83, Recall: 0.63, F1-Score: 0.72 (Macro Avg: 0.86).

Karar Ağacı (Decision Tree) Sonuçları:
Confusion Matrix: [[56855, 9], [22, 76]]
98 gerçek dolandırıcılığın 76 tanesini başarıyla yakalamış, kaçırılan dolandırıcılık sayısını 22'ye düşürmüştür (FN = 22).
Yanlış alarm sayısı 9'a gerilemiştir (FP = 9).
Fraud Sınıfı Skorları: Precision: 0.89, Recall: 0.78, F1-Score: 0.83 (Macro Avg: 0.92).

---------------------------------------
5. Banka Yönetimi Stratejik Karar Kriteri

Precision Odaklı Tercih: Yönetim "Masum müşterimizin kartı gereksiz yere bloke olmasın, çağrı merkezine şikayet yağmasın" derse, Precision skoru yüksek model seçilir. (Varsın 1-2 dolandırıcı kaçsın ama dürüst müşteriler mağdur olmasın mantığı).
Recall Odaklı Tercih: Yönetim "Müşteri rahatsız olsa da olur, yeter ki hiçbir dolandırıcıyı gözden kaçırmayalım, banka milyonlarca lira zarar etmesin" derse, Recall skoru yüksek model seçilir. (Gerekirse masumların kartı güvenlik için geçici teyit alırken hırsızlar kaçamasın mantığı).

------------------------------------------
 6. Sonuç ve İş Etkisi (Results & Business Impact)

 Geliştirilen modellerin (Lojistik Regresyon ve Karar Ağacı (Decision Tree)) karşılaştırmalı analizleri sonucunda, özellikle Recall skoru optimize edilen modellerin dolandırıcılık tespitindeki başarısı kanıtlanmıştır.
Kapanış:""Elde edilen bu sonuçlar doğrultusunda, Karar Ağacı (Decision Tree) modeli sayesinde manuel inceleme süreçleri otomatikleştirilebilir, bankanın sahtekarlık kaynaklı finansal kayıpları minimize edilirken müşteri memnuniyetsizliği de en düşük seviyede tutulabilir."

-------------------------------------------------
 7. Web Arayüzü ve Canlı Test (Streamlit App)

Geliştirilen modelin canlı ortamda test edilebilmesi ve kullanıcı dostu bir simülasyon sunulması amacıyla Streamlit ile interaktif bir web arayüzü (`app.py`) tasarlanmıştır.
Amaçlar
* Eğitilen makine öğrenmesi modelinin pratik bir arayüz üzerinden test edilebilmesini sağlamak.
* Gerçek dünya senaryolarına benzer bir akış oluşturarak modelin anlık karar verme yeteneğini gözlemlemek.
* Manuel inceleme süreçlerinin yapay zeka ile nasıl otomatikleştirilebileceğini prototiplemek.

-- Çalışma Mantığı ve Girdiler--
İşlem Detayları: Kullanıcı tarafından girilen işlem tutarı (`Amount`), tarih, saat ve ülke bilgileri alınır.
Özellik Dönüşümü: Kullanıcıdan alınan saat bilgisi modelin kabul ettiği saniye cinsine (`Time`) çevrilir. Kaggle veri setindeki şifrelenmiş PCA bileşenleri (`V1-V28`) arka planda varsayılan nötr değer olan `0.0` ile beslenir.
Ölçeklendirme ve Tahmin: Veriler kaydedilen `Scaler` ile dönüştürülerek eğitilen model üzerinden risk analizi gerçekleştirilir.

--Çıktılar ve Geri Bildirim--
Modelin yaptığı değerlendirme sonucunda arayüz üzerinden anlık olarak şu çıktılar üretilir:
Yüksek Risk (`Prediction = 1`): İşlem şüpheli bulunarak `BLOKE EDILDI` şeklinde görsel uyarı verilir.
Güvenli İşlem (`Prediction = 0`): İşlem güvenli bulunarak `İŞLEM ONAYLANDI` şeklinde onay mesajı gösterilir.

--------------------------------
-------Veri Seti Bilgisi------ 
Kaynak: Kaggle (Machine Learning Group - ULB) 
Lisans: Open Data Commons Open Database License (ODbL) / Akademik Kullanıma Açık
Boyut: 284.807 örnek, 31 öznitelik
Öznitelikler: PCA Dönüşümü Yapılmış Değişkenler (V1-V28), İşlem Tutarı (Amount), Zaman (Time), Dolandırıcılık Durumu (Class - Hedef)
Kullanılan Teknolojiler: Python 3.x, Pandas & NumPy (Veri Ön İşleme),
Scikit-Learn (Logistic Regression, Decision Tree, StandardScaler),
Streamlit (Web Arayüzü ve Simülasyon) 








