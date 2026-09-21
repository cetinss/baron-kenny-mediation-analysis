# Dağılımsal Gerçekçilik Kontrolü: Global Coffee Health Veri Seti

**Amaç:** LEMURS doğrulaması (bkz. `lemurs_validation_results.md`) sadece **ilişkilerin yönünü** test etti. Bu doküman farklı bir soruyu yanıtlıyor: kendi veri setimizdeki **değerlerin/dağılımların büyüklüğü**, gerçek bir insan popülasyonuyla ne kadar uyumlu? Üç kanıt hattı toplandı: (1) veri seti sahibiyle doğrudan yazışma, (2) dağınık literatür kıyaslaması (zayıf kanıt — bkz. Bölüm 2), (3) **NHANES (CDC, gerçek ABD ulusal sağlık anketi) ile doğrudan, aynı-örneklem satır-bazlı karşılaştırma** (güçlü kanıt — bkz. Bölüm 3, raporun asıl bulgusu).

---

## 1. Üretici ile Doğrudan Yazışma (Birincil Kaynak — Kişisel İletişim)

Kullanıcı, Kaggle'daki "Global Coffee Health Dataset"in sahibi **Laksika**'ya (uom190346a) doğrudan ulaşmış ve şu bilgileri almış (Mart 2026, WhatsApp/Telegram yazışması):

> *"The GlobalCoffeeHealth dataset contains 10,000 synthetic records reflecting real-world patterns of coffee consumption, sleep behavior, and health outcomes across 20 countries."*

Takip sorusuna ("So you get the original part from somewhere/recorded and then augmented it right?") verilen yanıt:

> **"No, It is purely synthetic."**

**Epistemik durum:** Bu, Kaggle'ın kendi "About this dataset" açıklamasıyla tutarlı ama **doğrulanabilir bir metodoloji değil** — hakemli değil, algoritma/kaynak belirtilmemiş, gayri-resmi bir yazışma yanıtı (kişisel iletişim). LEMURS'un DP/AIM sürecinin hakemli yayında belgelenmesiyle (Ghasemizade ve ark., 2025) **aynı epistemik ağırlıkta değildir**. Bunu "kişisel iletişim" (personal communication) kaynağı olarak kullanmak akademik olarak meşrudur, ama "bağımsız doğrulanmış üretim metodolojisi" olarak sunulmamalıdır.

---

## 2. Dağılımsal Kıyas — Bizim Temizlenmiş Örneklem (n=9,795) vs. Dağınık Literatür (Zayıf Kanıt)

**Not:** Bu bölümdeki kıyas değerleri, birbirinden bağımsız, farklı popülasyon/yıl/yönteme sahip ayrı çalışmalardan alınmış tek-sayı istatistikleridir (tutarlı, tek bir kaynak değil). Bölüm 3'teki NHANES kıyası bunun yerine geçen, çok daha güçlü bir kanıttır — bu bölüm sadece ilk yönelim/bağlam için tutuluyor.

Kaynak: `src/main.py`'nin `load_and_clean_data()` fonksiyonuyla üretilen aynı analitik örneklem (yaş 18–65 filtresi + IQR aykırı değer temizliği sonrası).

| Değişken | Bizim Örneklem | Literatür Kıyas Değeri | Değerlendirme |
|---|---|---|---|
| **Caffeine_mg** | Ort. 236.5, SS 135.1, aralık 0–620.6 | ABD genel yetişkin ort. **135 mg/gün**; sadece kafein tüketenler arası ort. **210 mg/gün**; ülkeye göre 110–260 mg/gün (Japonya ~260); yaygın "güvenli orta düzey" tavan **400 mg/gün** | **Kısmen gerçekçi.** Ortalamamız genel popülasyon ortalamasının üstünde ama "tüketen alt grup" ve yüksek-tüketim ülke ortalamalarına yakın. Maksimum (620 mg) yaygın güvenli sınırın belirgin üzerinde bir kuyruk oluşturuyor. |
| **Sleep_Hours** | Ort. 6.65 saat, SS 1.21, aralık 3.3–10.0 | ABD genel ort. **6.8 saat/gece**; yetişkinlerin %30–35'i önerilen 7–9 saati karşılamıyor (CDC) | **Gerçekçi — çok yakın eşleşme.** |
| **BMI** | Ort. 23.93, SS 3.84, aralık 15–34.5 | Küresel ort. BMI ~**25** (WHO, 2016); yetişkinlerin %43'ü fazla kilolu (BMI≥25) | **Makul aralıkta.** Ortalamamız hafif düşük ama örneklemdeki Asya ağırlıklı ülke karışımıyla (Çin, Japonya, G. Kore, Hindistan — düşük-BMI bölgeleri) tutarlı olabilir. |
| **Heart_Rate** (dinlenik) | Ort. 70.5, SS 9.64, aralık 50–96 | Klinik normal aralık **60–100 bpm**; büyük kohort (n=92,457) ort. **65 bpm** | **Gerçekçi.** Normal klinik aralıkta, büyük-kohort ortalamasından biraz yüksek ama 1 SS içinde. |
| **Physical_Activity_Hours** | Ort. **7.49 saat/hafta** (≈449 dk/hafta), SS 4.32 | WHO önerisi: **150–300 dk/hafta** orta-şiddetli aktivite; fayda 300 dk sonrası düzleşiyor | **Gerçekçi değil / iyimser.** Ortalamamız WHO'nun önerilen aralığının üst sınırının bile ~1.5 katı, alt sınırının ~3 katı. Sentetik popülasyon, gerçek dünya ortalamasına göre belirgin şekilde daha "aktif" görünüyor. |
| **Stress_Level** | Low %70.1 / Medium %20.6 / High %9.3 | Gallup (2021/2024, 122–144 ülke): yetişkinlerin **%37–41'i** "dün çok stres hissettim" diyor; APA (ABD): **%75–80'i** son bir ayda en az orta düzey stres bildiriyor | **Kıyaslanamaz / metodoloji belirsiz.** Ölçek, zaman çerçevesi ve soru formatı tamamen farklı (kategorik keyfi 3-etiket vs. gerçek anket maddeleri). Kaba olarak Medium+High=%29.9, her iki literatür rakamının da belirgin altında — ama bu bir "yanlış" olduğu anlamına gelmiyor, sadece **karşılaştırılabilir bir ölçüm olmadığı** anlamına geliyor. |
| **Country dağılımı** | 20 ülke, kayıt sayısı 437–528 arası (ort. 489.75) — neredeyse tam dengeli | Gerçek küresel örneklemede ülke payları nüfus/erişime göre büyük farklılık gösterir (Çin ~1.4 milyar vs. İsviçre ~9 milyon) | **Gerçekçi değil — açık sentetik-üretim imzası.** 20 ülkenin bu kadar yakın sayıda (%±6 içinde) temsil edilmesi, gerçek bir örnekleme sürecinde neredeyse imkansızdır; bu, ülke etiketlerinin **eşit olasılıkla rastgele atandığının** doğrudan istatistiksel kanıtıdır. |

---

## 3. NHANES ile Doğrudan Karşılaştırma (Güçlü Kanıt — Aynı Yöntem, Gerçek Veri)

**Kaynak:** NHANES 2017–2018 (CDC/NCHS, ABD ulusal temsili sağlık anketi), `src/nhanes_realism_check.py`. Kaggle'daki hazır "Sleep Health/Lifestyle" veri setleri **kontrol edildi ve elenidi** — ikisi de sentetikti (biri Global Coffee Health ile aynı yaratıcıya ait), dairesel kanıt olurdu. Bunun yerine CDC'nin resmi sitesinden **sadece ihtiyaç duyulan 6 dosya** doğrudan indirildi (demografi, diyet-kafein, vücut ölçümleri, nabız, fiziksel aktivite anketi, uyku anketi) — 5,078 değişkenli devasa birleştirilmiş dosya kullanılmadı.

**Değişken eşlemesi:** `DR1TCAFF`→Caffeine_mg (diyet hatırlama, mg — doğrudan ölçülen), `BMXBMI`→BMI (muayenede ölçülen), `BPXPLS`→Heart_Rate (60-sn nabız), `SLD012`→Sleep_Hours (hafta içi alışılmış uyku saati — Coffee Health'in "haftalık ortalama" tanımından farklı, bu bir kapsam notu), `PAQ*`→Physical_Activity_Hours (GPAQ, 5 kategori × gün × dakika/60 toplamı, iş+ulaşım+rekreasyon dahil — LEMURS'taki IPAQ mantığıyla aynı yöntemle inşa edildi).

**Örnekleme:** 18–65 yaş filtresi + 7 değişkende eksiksiz satır → n=3,666 (tam havuz). **"İlk 300 kişi" YERİNE** sabit seed (42) ile basit rastgele örneklem çekildi (n=300) — çünkü NHANES `SEQN` sırası toplama sırasına göre kümelenmiştir, ilk-N seçimi sistematik yanlılık riski taşır.

### Tablo: Coffee Health vs. NHANES (Tam Havuz, n=3,666) vs. NHANES Rastgele Alt-örneklem (n=300)

| Değişken | Coffee Health (n=9,795) | NHANES n=300 (rastgele) | NHANES Tam (n=3,666) |
|---|---|---|---|
| Caffeine (mg/gün) | Ort. 236.5, SS 135.1, **medyan 234.8** | Ort. 166.6, SS 337.2 | Ort. 140.0, SS 220.9, **medyan 89.0** |
| Sleep (saat) | Ort. 6.65, SS 1.21 | Ort. 7.55, SS 1.52 | Ort. 7.51, SS 1.63 |
| BMI | Ort. 23.93, SS 3.84 | Ort. 28.83, SS 7.10 | Ort. 29.83, SS 7.62 |
| Heart Rate (bpm) | Ort. 70.48, SS 9.64 | Ort. 71.91, SS 11.61 | Ort. 72.52, SS 11.70 |
| Physical Activity (h/hafta) | Ort. 7.49, SS 4.32, **medyan 7.5** | Ort. 14.41, SS 21.15 | Ort. 15.24, SS 21.50, **medyan 6.0** |
| Age (yıl) | Ort. 34.77, SS 10.92 | Ort. 42.65, SS 14.74 | Ort. 42.58, SS 14.60 |

*(Tam tablo ve n=300 örneklem CSV olarak: `outputs/nhanes_comparison_table.csv`, `outputs/nhanes_random_sample_n300.csv`. Görsel: `outputs/figures/NHANES_Comparison_Distributions.png`.)*

### Bulgular — Önceki (Bölüm 2) Değerlendirmeleri Güncelliyor/Düzeltiyor

**BMI — önceden "makul" denmişti, gerçek veriyle KESİN UYUMSUZ çıktı:**
NHANES'te BMI≥25 (fazla kilolu+) oranı **%72.2**, Coffee Health'te **%40.3**. Gerçek popülasyonun neredeyse dörtte üçü fazla kiloluyken, sentetik veri setinde bu oran dörtte biri civarında. Dağılım grafiğinde de açıkça görülüyor: NHANES sağa doğru belirgin şekilde kaymış ve daha uzun bir kuyruğa sahip (70'lere kadar), Coffee Health ise dar ve "normal kilo" merkezli. **Bu, Bölüm 2'deki zayıf literatür kıyasının verdiği "makul" izlenimini düzeltiyor — gerçek eşleşen veriyle bakıldığında belirgin bir uyumsuzluk var.**

**Caffeine — ortalama benzer görünüyordu, ama DAĞILIM ŞEKLİ tamamen farklı:**
Medyan karşılaştırması ortalamadan çok daha çarpıcı: Coffee Health medyan **234.8 mg**, NHANES medyan sadece **89.0 mg** (yaklaşık 2.6 kat fark). Çarpıklık (skewness) katsayısı: Coffee Health **0.18** (neredeyse simetrik, çan eğrisi), NHANES **7.80** (aşırı sağa çarpık). Kafein tüketmeyen/az tüketen (<20mg) oranı: Coffee Health'te %6.7, NHANES'te **%28.4** — gerçek popülasyonda dört kat daha fazla düşük/sıfır tüketici var. Gerçek dünyada kafein tüketimi "az sayıda yüksek tüketici + çok sayıda düşük/sıfır tüketici" şeklinde aşırı çarpık dağılırken, Coffee Health'in ürettiği dağılım yapay bir şekilde simetrik/dengeli.

**Physical Activity — yön ters çıktı, ama şekil çok daha önemli bir sorun:**
Bölüm 2'de "WHO önerisinin üzerinde, iyimser" denmişti (sadece ortalama karşılaştırılarak). Gerçek NHANES verisiyle ortalama karşılaştırıldığında durum tam tersi: NHANES ortalaması (15.24 saat/hafta) Coffee Health'ten (7.49) bile yüksek — çünkü GPAQ iş-kaynaklı yoğun aktiviteyi de sayıyor ve bazı meslek gruplarında bu çok yüksek değerler üretiyor (SS=21.5, aşırı çarpık, çarpıklık=2.14). Ama medyanlar daha yakın (6.0 vs 7.5). **Asıl sorun ortalama değil şekil:** NHANES'te %24.2'si haftada 1 saatten az "kayıtlı" aktiviteye sahipken (büyük ölçüde hareketsiz), Coffee Health'te bu oran sadece %6.2 — ve Coffee Health'in dağılımı 15 saatte keskin bir şekilde kesiliyor (yapay bir tavan), gerçek veride böyle bir üst sınır yok (bazı değerler 100+ saate çıkıyor). Sentetik veri, gerçek dünyadaki hem düşük-aktivite hem yüksek-aktivite uç gruplarını temsil etmiyor; her ikisi de "makul bir orta bandın" içine sıkıştırılmış.

**Sleep ve Heart Rate — en tutarlı ikili, doğrulandı:**
Bu ikisi hem Bölüm 2'nin literatür kıyasında hem de burada NHANES ile doğrudan kıyasta en yakın eşleşmeyi veriyor (ortalamalar birbirine ~1 SS içinde, dağılım şekilleri histogram üzerinde büyük ölçüde örtüşüyor). Coffee Health'in en gerçekçi iki değişkeni bunlar.

---

## 4. Somut Bulgu: Caffeine_mg, Bağımsız Ölçülmüş Değil — Coffee_Intake'in Deterministik Bir Fonksiyonu

Bu, raporun en güçlü ve en kesin kanıtı:

```
corr(Coffee_Intake, Caffeine_mg) = 0.9998
Caffeine_mg / Coffee_Intake oranı: ortalama = 95.02 mg/fincan, SS = 3.20 (CV = %3.4)
```

`Caffeine_mg` değişkeni, `Coffee_Intake` (fincan/gün) değişkeninden **neredeyse sabit bir katsayıyla (~95 mg/fincan, ihmal edilebilir gürültüyle) türetilmiş** — yani bağımsız olarak modellenmiş/ölçülmüş bir değişken değil, basit bir formülün çıktısı. Gerçek dünyada mg/fincan oranı; kahve türüne (espresso ~63 mg/shot, filtre kahve ~95–200 mg/fincan, hazır kahve ~30–90 mg/fincan), demleme yöntemine, fincan boyutuna ve ülkeye göre **çok büyük varyasyon** gösterir — bizim verimizdeki %3.4'lük varyasyon katsayısı bunun çok altında.

**Bu, üreticinin "purely synthetic" ifadesini somut, ölçülebilir bir kanıtla doğruluyor**: en azından bu değişken çifti, gerçek bir kohorttan toplanmış bağımsız ölçümler değil, basit bir formülle (fincan × sabit mg-katsayısı + küçük gürültü) üretilmiş.

---

## 5. Genel Değerlendirme (NHANES Kanıtıyla Güncellenmiş)

| Kanıt türü | Sonuç |
|---|---|
| Üreticiyle doğrudan iletişim | "Purely synthetic", gerçek veriden augment edilmemiş — ama hakemsiz, doğrulanamaz beyan |
| Sleep, Heart Rate | NHANES ile **gerçekçi** düzeyde örtüşüyor (ortalama + dağılım şekli) |
| BMI | **Gerçekçi değil** — NHANES'te %72.2 fazla kilolu, Coffee Health'te %40.3; dağılım belirgin şekilde sola kaymış |
| Caffeine_mg | **Dağılım şekli gerçekçi değil** — medyan 2.6× yüksek (235 vs 89 mg), çarpıklık yapay şekilde düşük (0.18 vs 7.80); ortalama kabaca aynı büyüklük mertebesinde ama bu yanıltıcı |
| Physical Activity | **Dağılım şekli gerçekçi değil** — gerçek veri hem daha çok hareketsiz kişi (%24 vs %6) hem de üst sınırsız uç değerler içeriyor; Coffee Health 15h/hafta'da yapay şekilde kesiliyor |
| Stress_Level | **Kıyaslanamaz** — ölçüm yöntemi/skalası belgelenmemiş, NHANES'te doğrudan karşılığı yok |
| Country dağılımı | **Açıkça yapay** — istatistiksel olarak neredeyse imkansız derecede dengeli |
| Caffeine_mg ↔ Coffee_Intake ilişkisi | **Kesin kanıt**: deterministik formülle üretilmiş, bağımsız ölçüm değil |

**Sonuç (güncellenmiş):** NHANES ile yapılan doğrudan, aynı-yöntemli karşılaştırma, Bölüm 2'deki gevşek literatür kıyasının verdiği "kısmen gerçekçi" izlenimini büyük ölçüde düzeltiyor. Sadece **iki değişken (uyku süresi, dinlenik kalp atış hızı)** gerçek popülasyon verisiyle hem ortalama hem dağılım şekli bazında tutarlı. Geri kalan tüm değişkenlerde (BMI, kafein, fiziksel aktivite, ülke dağılımı) ya ortalama düzeyinde ya da — daha sık ve daha önemlisi — **dağılım şekli düzeyinde** (çarpıklık, uç değerler, hareketsiz/düşük-tüketici alt grupların eksikliği) belirgin sapmalar var. Bu, sentetik üretim sürecinin muhtemelen değişkenleri **birbirinden bağımsız, dar ve simetrik aralıklarda** örneklediğini, gerçek popülasyonların tipik özelliği olan uzun kuyruklu/çarpık dağılımları ve alt-grup heterojenliğini yakalayamadığını gösteriyor.

**Bulgunun özeti:** "Veri setinin marjinal dağılımları, ABD'yi temsil eden gerçek bir ulusal sağlık anketiyle (NHANES 2017–2018, n=3.666, aynı yaş aralığı ve değişken tanımlarıyla) doğrudan karşılaştırılmıştır. Uyku süresi ve dinlenik kalp atış hızı gerçek veriyle tutarlı bulunmuştur; ancak BMI, kafein alımı ve fiziksel aktivite değişkenlerinde hem ortalama hem de dağılım şekli (çarpıklık, uç değer varlığı) düzeyinde belirgin sapmalar tespit edilmiştir. Üretici ile doğrudan iletişimde veri setinin 'purely synthetic' olduğu doğrulanmıştır (kişisel iletişim, Mart 2026), ve bu NHANES kıyası bulgularıyla tutarlıdır." Bu çerçeveleme, "veri setimiz gerçekçidir" gibi savunulamaz bir iddiadan tamamen kaçınırken, LEMURS'taki yön-tutarlılığı bulgusuyla (bkz. `comparison_summary.md`) birlikte okunduğunda şunu gösterir: **nedensel ilişkinin yönü bağımsız gerçek veride de tekrarlanıyor (LEMURS), ancak Coffee Health'in mutlak değerleri/dağılımları gerçek bir popülasyonu temsil etmiyor (NHANES) — bu iki bulgu birbiriyle çelişmez, farklı sorulara cevap verir ve makalenin sınırlılıklar bölümünde bu ayrım net şekilde yapılmalıdır.**

**İlgili dosyalar:** `feature_matching_report.md`, `lemurs_validation_results.md`, `comparison_summary.md`, `limitations.md`, `src/nhanes_realism_check.py`, `outputs/nhanes_comparison_table.csv`, `outputs/nhanes_random_sample_n300.csv`, `outputs/figures/NHANES_Comparison_Distributions.png`
