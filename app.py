import streamlit as st
import joblib
import numpy as np
import datetime

# Sayfa Yapılandırması
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")

# Kaydettiğimiz modeli ve scaler'ı içeri aktarıyoruz
model = joblib.load('kredi_karti_modeli.pkl')
scaler = joblib.load('kredi_karti_scaler.pkl')

st.title("💳 Finansal İşlem Güvenliği")
st.write("Lütfen risk analizi yapılacak işlemin detaylarını girin.")
st.markdown("---")

# Ekranı iki sütuna bölüyoruz
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🌍 İşlem Lokasyonu ve Para Birimi**")
    ulke = st.selectbox("İşlem Yapılan Ülke", ["Türkiye", "Amerika Birleşik Devletleri", "Almanya", "İngiltere", "Fransa", "İtalya", "Diğer"])
    para_birimi = st.selectbox("Para Birimi", ["TRY (₺)", "USD ($)", "EUR (€)", "GBP (£)"])
    amount = st.number_input("İşlem Tutarı", min_value=0.0, value=1500.0, step=50.0)

with col2:
    st.markdown("**📅 İşlem Zamanı**")
    
    # Yıl, Ay ve Gün seçimini anlık çalışan açılır menülere bölüyoruz
    col_yil, col_ay, col_gun = st.columns(3)
    
    with col_yil:
        # 2030'dan 2010'a kadar yıllar listesi (2019 dahil)
        yil = st.selectbox("Yıl", list(range(2030, 2009, -1)), index=5) 
        
    with col_ay:
        aylar = {
            "Ocak": 1, "Şubat": 2, "Mart": 3, "Nisan": 4, "Mayıs": 5, "Haziran": 6,
            "Temmuz": 7, "Ağustos": 8, "Eylül": 9, "Ekim": 10, "Kasım": 11, "Aralık": 12
        }
        ay_adi = st.selectbox("Ay", list(aylar.keys()), index=9) # Varsayılan Ekim
        ay = aylar[ay_adi]
        
    with col_gun:
        gun = st.number_input("Gün", min_value=1, max_value=31, value=2, step=1)

    # Seçilen parçaları birleştirip tarihi oluşturuyoruz
    try:
        islem_tarihi = datetime.date(yil, ay, gun)
    except ValueError:
        st.error("⚠️ Geçersiz tarih! (Örneğin Şubat 30 çekmez). Lütfen günü kontrol edin.")
        islem_tarihi = datetime.date.today()

    islem_saati = st.time_input(
        "İşlem Saati", 
        value=datetime.datetime.strptime("14:30", "%H:%M").time(),
        help="Saati listeden seçmek yerine klavyenizi kullanarak doğrudan yazabilirsiniz."
    )
    st.caption("💡 İpucu: Yıl veya ay menüsünden herhangi bir yıla tıkladığınızda sayfa anında güncellenir.")

st.markdown("---")

if st.button("Güvenlik Analizini Başlat", use_container_width=True):
    
    toplam_saniye = islem_saati.hour * 3600 + islem_saati.minute * 60 + islem_saati.second
    
    # V1-V28 için ortalama/etkisiz değer olan 0.0'ı arka planda otomatik ekliyoruz.
    v_features = [0.0] * 28 
    
    # Verileri Kaggle veri setindeki sıraya göre birleştiriyoruz
    input_data = [toplam_saniye] + v_features + [amount]
    input_array = np.array([input_data])
    
    # Scaler işlemi
    scaled_data = scaler.transform(input_array)
    
    # Tahmin
    prediction = model.predict(scaled_data)
    
    # Sonucu Ekrana Yazdırma
    if prediction[0] == 1:
        st.error(f"🚨 YÜKSEK RİSKLİ İŞLEM: {ulke} konumundan yapılmak istenen {amount} {para_birimi} tutarındaki işlem şüpheli bulunarak BLOKE EDİLMİŞTİR.")
    else:
        st.success(f"✅ İŞLEM ONAYLANDI: {ulke} konumundan yapılan {amount} {para_birimi} tutarındaki işlem güvenli bulunmuştur.")