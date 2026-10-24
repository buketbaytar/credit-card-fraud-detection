import pandas as pd

# Veriyi yükleme (creditcard.csv dosyasının Python dosyası ile aynı dizinde olduğundan emin ol)
df = pd.read_csv('creditcard.csv')

# X ve y ayrımı
# X: Tüm satırlar (:), son sütun hariç tüm sütunlar (:-1)
X = df.iloc[:, :-1]

# y: Tüm satırlar (:), sadece son sütun (-1)
y = df.iloc[:, -1]

# Veri dengesizliğini kontrol etme
sinif_dagilimi = y.value_counts()
print("Sınıf Dağılımı:\n", sinif_dagilimi)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Veriyi %80 Eğitim ve %20 Test olacak şekilde bölüyoruz.
# random_state=42 parametresi, kodu her çalıştırdığımızda aynı satırların teste ayrılmasını sağlar (sonuçlarımız sabit kalsın diye).
# stratify=y parametresi çok önemli! 492 adet olan dolandırıcılık işlemini train ve test setlerine eşit oranda dağıtır.
#stratify y orantılı bölmek için uygunulandı. 492 adedi %80 e %20 böldü 
#Eğitim Seti (%80)	227.452 (284.315'in %80'i)	394 (492'nin %80'i)
#Test Seti (%20)	56.863 (284.315'in %20'si)	98 (492'nin %20'si)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Veri Bölme İşlemi ---")
print("Eğitim seti satır sayısı:", X_train.shape[0])
print("Test seti satır sayısı:", X_test.shape[0])

# 2. Veri Ölçeklendirme (StandardScaler)
# Verilerimizin birimlerini standardize ederek modelin daha sağlıklı öğrenmesini sağlıyoruz.
#Sütunun ortalamasını 0'a, standart sapmasını 1'e çeker). yani 10 tlnin de güvenlik skorunun da aynı önemde incelemesi sağlıyor 
# Sonuç olarak tüm sütunlardaki değerler genellikle -3 ile +3 arasında bir sayılara dönüşür. 
scaler = StandardScaler()

# Scaler'ı sadece eğitim verisi üzerinde 'fit' ediyoruz (öğreniyor) ve uyguluyoruz (transform).
#fit, o sütunun ortalamasını ve standart sapmasını hesaplar. fit sadece ve sadece eğitim verisinde yapılır.
#Modelin henüz görmediği test verilerinden kopya çekmesini (veri sızıntısını) engellemek 
#ve ölçüm referansını sadece geçmiş tecrübelere göre sabitleyerek 
#gerçek dünya koşullarını taklit etmek için fit (öğrenme) işlemini yalnızca eğitim setinde uygularız. X_train_scaled = scaler.fit_transform(X_train)
#  transform ise bu hesaplanan ortalamayı kullanarak sayıları -3 ile +3 arasına çeker. 
#Eğitim seti (Train) "geçmişi", test seti (Test) ise "geleceği" (yani henüz yaşanmamış, yeni gelecek kredi kartı işlemlerini) temsil eder.

X_train_scaled = scaler.fit_transform(X_train)


#Eğitim setinden öğrendiğimiz sabit ölçüm kurallarını (ortalamayı) hiç değiştirmeden 
#doğrudan test verisine uygulayarak, modelin test verisinden kopya çekmesini engelliyor 
#ve tüm verileri aynı standart ölçekte buluşturuyoruz. X_test_scaled = scaler.transform(X_test)
# Test verisini ise SADECE dönüştürüyoruz. Test verisinden hiçbir şey 'öğrenmemeli' (veri sızıntısını önlemek için).
X_test_scaled = scaler.transform(X_test)

print("Veriler başarıyla ölçeklendirildi (Standardized).")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

print("\n--- Adım 3: Modellerin Kurulması ve Eğitilmesi ---")

# 1. Lojistik Regresyon Modeli
print("Lojistik Regresyon modeli eğitiliyor...")
# random_state=42 ile sonuçların her çalıştırdığımızda aynı çıkmasını sağlıyoruz.
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)

# 2. Karar Ağacı (Decision Tree) Modeli
print("Karar Ağacı modeli eğitiliyor...")
# Aşırı öğrenmeyi (overfitting - veriyi ezberlemesini) engellemek için ağacın derinliğini (max_depth) 5 ile sınırlıyoruz.
tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
tree_model.fit(X_train_scaled, y_train)

print("Her iki modelin de eğitimi başarıyla tamamlandı!")


from sklearn.metrics import confusion_matrix, classification_report

print("\n--- Adım 4: Lojistik Regresyon Sonuçları ---")


# Modeldan test verisi (daha önce görmediği 56 bin satır) üzerinden tahminleri alıyoruz
#Tahmin Yapma (predict) y_pred_log = log_model.predict(X_test_scaled) komutuyla model, 
#sadece özellikleri içeren ama cevap anahtarı olmayan 56 bin satırlık test verisine bakar. 
#Her bir satır için "0 (Normal)" veya "1 (Dolandırıcı)" tahmini üretir ve bunları kaydeder.

y_pred_log = log_model.predict(X_test_scaled)



# Karmaşıklık Matrisi (Confusion Matrix) hesaplama
#Karmaşıklık Matrisi (confusion_matrix)
#Modelin tahminleri ile gerçek cevap anahtarının (y_test) karşılaştırıldığı 2x2'lik bir ölçüm tablosudur. 
#Ekranda şuna benzer bir sonuç basar:[[56850    13] [   40    58]]

print("Karmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred_log))


# Recall, Precision ve F1-Score detaylarını görme
#Sınıflandırma Raporu (classification_report)
#Modelin başarısını yüzdelik skorlarla özetler. 
#Ekranda Precision (Hassasiyet) ve Recall (Duyarlılık) değerlerini görürsünüz.
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_log))

print("\n--- Adım 4: Karar Ağacı Sonuçları ---")
# Karar Ağacı modelinin tahminleri
y_pred_tree = tree_model.predict(X_test_scaled)

print("Karmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred_tree))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_tree))


import joblib

# Eğittiğiniz lojistik regresyon veya karar ağacı modelini ve scaler'ı kaydediyoruz
joblib.dump(log_model, 'kredi_karti_modeli.pkl')
joblib.dump(scaler, 'kredi_karti_scaler.pkl')

print("Model ve Scaler başarıyla kaydedildi!")











