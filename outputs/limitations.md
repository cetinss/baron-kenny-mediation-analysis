# LEMURS Dış Doğrulama — Sınırlılıklar

Bu doküman, `lemurs_validation_results.md`'deki dış doğrulama analizinin tüm proxy/varsayım noktalarını ve LEMURS'un veri statüsünü açık biçimde özetler.

---

## 1. LEMURS'un veri statüsü — "sentetik ama belgelenmiş"

LEMURS (`survey_syn_5.csv`) **teknik olarak sentetiktir**, ancak orijinal analizde kullanılan Global Coffee Health veri setinden kategorik olarak farklıdır:

| | Global Coffee Health (orijinal analiz) | LEMURS (dış doğrulama) |
|---|---|---|
| Kaynak | Kaggle, "illustrative/synthetic" olarak etiketli | Gerçek bir çalışmadan (LEMURS, UVM, ~600 öğrenci, Oura Ring + haftalık anket) |
| Üretim yöntemi | Belgelenmemiş | Differential Privacy, AIM algoritması, ε=5 (formal matematiksel gizlilik garantisi) |
| Gerçek veriyle ilişki | Belirtilmemiş/bilinmiyor | Marjinal dağılımlar ve Spearman korelasyon yapısı gerçek veriyle karşılaştırılarak doğrulanmış |
| Doğrulama | Yok | Hakemli yayın: Ghasemizade ve ark. (2025), *JAMIA Open* |

Bu nedenle LEMURS, "gerçek veri" olarak sunulmamalıdır — ama "hiçbir dayanağı olmayan sentetik veri" de değildir. Doğru çerçeveleme: **gerçek insan verisinden formal gizlilik garantisiyle türetilmiş, üretim yöntemi ve istatistiksel sadakati bağımsız olarak hakem denetiminden geçmiş sentetik referans veri seti.**

---

## 2. Ölçüm proxy'leri (asıl değişkenlerle birebir değil)

### 2.1 Kafein (X)
LEMURS'ta mg/gün dozu **hiç yok**. `caffeine_exposure_score` (0–6), sadece kahve/çay/soda/enerji içeceği/antrenman içeceği/diğer kategorilerinden kaçının tüketildiğinin sayısıdır — **doz değil, çeşitlilik/sıklık proxy'sidir**. Bir kişi günde 3 fincan kahve içse de 1 fincan içse de bu skor aynıdır (=1). Bu, X'in ölçüm hassasiyetini ciddi şekilde azaltır ve path a/c'nin anlamlılık gücünü düşürür (bkz. Bölüm 4).

### 2.2 Stres (M)
Orijinal analizde stres, validasyonu belgelenmemiş, keyfi bir 3-kategori kodlamadır (Low=2/Medium=5/High=8). LEMURS'ta ise gerçek, validasyonu yapılmış PSS-10 (Cohen, Kamarck & Mermelstein, 1983) kullanılmıştır — bu yönde LEMURS orijinal veriden **daha güçlü bir ölçüm** sağlıyor, dolayısıyla bu karşılaştırma stres ölçümü açısından orijinal analizin lehine değil, LEMURS'un lehine bir asimetri içeriyor.

### 2.3 Fiziksel aktivite (kovaryat)
`activity_days` (haftalık aktif gün sayısı, 0–21) kullanıldı — orijinaldeki `Physical_Activity_Hours` (haftalık toplam saat) ile birim olarak örtüşmüyor. Süre-bazlı bir proxy de (`activity_hours_sensitivity`) hesaplanmıştır, ancak bu proxy'nin dayandığı iki IPAQ süre kolonunun (`hoursminutes1/4`) şiddetli mi orta şiddetli aktiviteye mi ait olduğu codebook'tan **kesin olarak belirlenemedi** (bkz. `feature_matching_report.md`, Kritik Bulgu B). Duyarlılık analizi sonuçları ana analizle pratik olarak aynı olduğu için bu belirsizliğin sonuçları etkilemediği gösterildi, ama yapısal belirsizlik kaynak veride kalıcı olarak mevcuttur.

### 2.4 Kovaryatların eksikliği
`Age`, `Gender`, `BMI`, `Heart_Rate` LEMURS survey verisinde **hiç bulunmuyor** — modelden tamamen çıkarılmıştır (icat edilmemiş, simüle edilmemiştir). Bu, LEMURS modelindeki confounding kontrolünü orijinal modelden daha zayıf hale getiriyor.

---

## 3. Yapısal/metodolojik kısıtlamalar

### 3.1 `record_id` eksikliği
LEMURS README'si bir `record_id` (katılımcı kimliği) kolonu belgeliyor, ancak indirilen `survey_syn_5.csv` dosyasında bu kolon **fiilen yok** (doğrulandı: ham CSV başlığı + codebook, 90 kolon). Bu nedenle:
- Kişi-bazlı ortalamaya indirgeme yapılamadı.
- Cluster-robust standart hatalı duyarlılık analizi yapılamadı.
- Analiz, 3,852 person-week satırını **bağımsız kesitsel gözlemler** olarak ele almak zorunda kaldı; bu sınırlılığın sayısal etkisi `lemurs_clustering_sensitivity.md` dosyasında ayrıca değerlendirilmiştir.

Bu, aynı kişiye ait birden fazla haftalık gözlemin birbiriyle korelasyonlu olabileceği (within-person autocorrelation) ihtimalini modele yansıtamadığımız anlamına gelir — standart hatalar hafifçe olduğundan düşük tahmin edilmiş olabilir (anti-konservatif). Ancak path a ve b'nin p değerleri (<.001) bu tahmin hatasını tolere edecek kadar güçlü; asıl marjinal olan toplam etki (c, p=.089) bu belirsizlikten etkilenmiş olabilir.

### 3.2 Örneklem/popülasyon farkı
LEMURS örneklemi tek bir üniversitenin (UVM) dar yaş aralığındaki (~18–19 yaş, birinci sınıf öğrenciler) popülasyonundan geliyor. Global Coffee Health veri seti ise çok-ülkeli, 18–65 yaş aralığında genel yetişkin popülasyonunu temsil ediyor (iddiasıyla). Bu, dış geçerliliği "genel yetişkin popülasyonuna" değil, "üniversite öğrencisi alt-popülasyonuna" sınırlıyor.

### 3.3 Eksik veri ve aykırı değer temizliği
Ham 3,852 satırdan 899'u (%23.3) çıkarıldı: 637'si eksik veri, 262'si IQR aykırı değer filtresi nedeniyle. PSS-10 skoru, 10 maddenin **hepsi** dolu olmadıkça hesaplanmadı (prorating yapılmadı) — bu, PSS madde-bazında %10–11 eksik veri oranı nedeniyle örneklem kaybına katkıda bulundu.

### 3.4 DP gürültüsü
ε=5 düzeyinde eklenen kalibre gürültü, ham marjinal ve ortak dağılımları hafifçe bozabilir (LEMURS yayınında belgelenen bir etki). Bu, gözlenen zayıf etki büyüklüklerinin bir kısmını açıklayabilir, ancak yayında bu ε düzeyinde Spearman korelasyon yapısının korunduğu gösterildiği için, temel yön/ilişki örüntüsünün DP gürültüsünden kaynaklı bir artefakt olması düşük ihtimaldir.

---

## 4. Etki büyüklüğü karşılaştırılabilirliği

R² (0.629 vs 0.007) ve oran-mediated (%60.6 vs %23.3) değerleri **doğrudan karşılaştırılmamalıdır** — bu bir "doğrulama başarısızlığı" değil, ölçüm hassasiyetindeki temel farkın matematiksel bir sonucudur: X sürekli mg-dozundan 0–6 kaba sayaca, M'nin ölçüm gücü orijinalde zayıf iken LEMURS'ta güçlüye dönüştüğünde, iki modelin açıkladığı varyans oranlarının örtüşmesi zaten beklenemez. Dış doğrulamada aranan kanıt standardı yön ve anlamlılık tutarlılığıdır (bkz. `comparison_summary.md`), büyüklük tutarlılığı değil.

---

## 5. Dış doğrulamanın kanıt gücü ve kalan zayıf noktalar (özet)

Bu bölüm, yukarıdaki ayrıntıların dosya-içi hızlı referansıdır:

- **Güçlü yön:** Bağımsız, farklı üretim yöntemiyle (DP + hakemli doğrulama) elde edilmiş bir veri setinde, tüm path'lerde işaret tutarlılığı ve 3/5 path'te anlamlılık tutarlılığı gösterilmiştir.
- **Zayıf yön:** Toplam etki (c) ve doğrudan etki (c′) LEMURS'ta anlamlı değil; kafein proxy'sinin kabalığı ve record_id eksikliği bu zayıflığın en olası kaynaklarıdır ve bunlar giderilemeyen, kaynak-veri-kaynaklı sınırlılıklardır (icat edilerek düzeltilmemiştir).
