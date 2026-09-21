# Özellik Eşleşme Raporu: Global Coffee Health (mevcut analiz) vs LEMURS (DP sentetik referans)

**Tarih:** 2026-08-14
**Amaç:** `src/main.py`'deki Baron-Kenny mediation analizinde kullanılan değişkenlerin, LEMURS DP-sentetik referans veri setindeki (`data/lemurs/survey_syn_5.csv`, ε=5) hangi kolonlarla örtüştüğünü sistematik olarak tespit etmek.

**Veri seti statüsü hatırlatması:** LEMURS de teknik olarak sentetiktir, ancak farklı bir kategoride — gerçek ~600 üniversite öğrencisinden (Oura Ring + haftalık anket) toplanan orijinal veriden, Differential Privacy (AIM algoritması, ε=5) ile türetilmiştir. Orijinal gerçek veri yeniden-teşhis riski nedeniyle paylaşılamıyor, ancak üretim yöntemi ve gerçek veriyle istatistiksel örtüşmesi Ghasemizade ve ark. (2025, JAMIA Open) tarafından hakemli bir yayında doğrulanmıştır. Bu nedenle LEMURS burada "gerçek insan verisinden formal gizlilik garantisiyle türetilmiş, bağımsız validasyonu yapılmış sentetik referans veri seti" olarak ele alınmaktadır — "gerçek veri" değil.

---

## 1. Kaynaklar

| Kaynak | Dosya |
|---|---|
| Benim analiz kodum | `src/main.py` |
| Benim veri setim | `data/synthetic_coffee_health_10000.csv` (n=10,000, 16 kolon) |
| LEMURS veri | `data/lemurs/survey_syn_5.csv` (3,852 satır × 90 kolon) |
| LEMURS codebook | `data/lemurs/survey_codebook.xlsx` (117 satır, F1_* soru metinleri + kodlamalar) |
| LEMURS README | `data/lemurs/README.md` |

---

## 2. Eşleşme Tablosu

| Benim değişkenim | Ölçek/birim | LEMURS'ta en yakın kolon(lar) | Ölçek/birim | Eşleşme derecesi | Not |
|---|---|---|---|---|---|
| `Sleep_Hours` (Y) | saat/gece, sürekli | `F1_Sleep_sleephours` | "Haftalık ortalama gece uyku saati", saat, sürekli | **TAM** | Soru metni birebir örtüşüyor. %10.3 eksik veri (395/3852). |
| `Stress_Level`/`Stress_Score` (M) | 3 kategori ordinal, keyfi kodlama (Low=2, Medium=5, High=8) | `F1_PSS_stressupset` … `F1_PSS_stressdifficulties` (10 madde) | Her madde 0–4 Likert ("Never"–"Very Often"), toplam PSS-10 skoru 0–40 | **KISMİ (LEMURS DAHA GÜÇLÜ)** | Benim tarafımda skala validasyonu belgelenmemiş, keyfi 3 kategorili bir kodlama. LEMURS tarafında ise gerçek, validasyonu yapılmış PSS-10 (Cohen ve ark. 1983) var. Codebook'ta 4 pozitif ifadeli madde teyit edildi: `stressconfident`, `stressthings`, `stressirritations`, `stressontop` — bunlar ters kodlanmalı (4−x). Ortalama %10.6 eksik veri/madde. |
| `Caffeine_mg` (X) | mg/gün, sürekli | `F1_Caffeine_coffee`, `_tea`, `_soda`, `_energydrink`, `_workoutdrink`, `_otherdrink` (6 binary bayrak) + `F1_Caffeine_caffeinethisweek` (gate sorusu) | Her biri 1=Evet/0=Hayır | **KISMİ** | mg değeri hiç yok — sadece hangi içecek türlerinin tüketildiği. `caffeinethisweek` gate sorusunda %10.2 eksik, 6 tür-bayrağında %0–0.5 eksik (tutarsızlık — muhtemelen DP gürültüsünün koşullu mantığı bozması). |
| `Age` | yıl, sürekli | **YOK** | — | **YOK** | CSV'de (90 kolon) ve codebook'ta (117 satır) demografik/yaş alanı bulunmuyor — doğrulandı. |
| `Gender` | kategorik | **YOK** | — | **YOK** | Aynı şekilde yok. |
| `BMI` | kg/m², sürekli | **YOK** | — | **YOK** | Aynı şekilde yok. |
| `Heart_Rate` (dinlenik) | bpm, sürekli | **YOK** (`survey_syn_5.csv`'de) | — | **YOK** | Oura Ring fizyolojik verisi muhtemelen ayrı bir "Ring" veri dosyasında tutuluyor (repo README'sinde ima ediliyor) ama bu görev kapsamında sadece Survey dosyası indirildi; survey formunda kalp atış hızı sorusu yok. |
| `Physical_Activity_Hours` | saat/hafta, tek sürekli değer | `F1_Activity_numvigorous`, `_nummoderate`, `_numwalking` (gün/hafta, 0–7) + `_hoursminutes1..4` (süre bileşenleri, saat+dakika) | IPAQ tabanlı, gün + süre bileşenleri | **KISMİ** | Birim dönüşümü gerekiyor (gün×süre → haftalık saat). Aşağıda **Kritik Bulgu B**'de açıklanan bir kategori belirsizliği var. |

---

## 3. Kritik Bulgular (görevde öngörülenin ötesinde)

### A. `record_id` kolonu fiilen YOK — kişi-bazlı analiz planı doğrudan uygulanamaz

LEMURS README'si şunu söylüyor: *"Participant identifier: `record_id` — a pseudo-anonymized numerical ID. Each participant may contribute multiple rows (one per study week)."*

Ancak indirilen `survey_syn_5.csv` dosyasının gerçek kolon listesi (90 kolon, doğrulandı — hem ham CSV başlığı hem codebook üzerinden) **`record_id` içermiyor**. Tek kimlik/gruplama bilgisi `week` (1–8) kolonu, ve bu tek başına kişileri birbirinden ayırmaya yetmiyor.

**Sonuç:** Adım 3'te istenen "her `record_id` için haftalık gözlemleri kişi-bazlı ortalamaya indirme" ve "ham hafta-satırlarıyla cluster-robust SE'li duyarlılık analizi" **olduğu gibi uygulanamaz** — hangi 3,852 satırın hangi ~600 kişiye ait olduğunu bilemiyoruz.

**Seçenekler (onayınızı bekliyorum):**
1. **Kesitsel havuzlama** — her satırı (person-week) bağımsız bir gözlem olarak ele al, n≈3,212 (tüm anahtar değişkenlerde eksiksiz satırlar). Cluster-robust SE analizi atlanır, limitations'da net şekilde belirtilir. *(Önerim budur — en az varsayım gerektiren, dürüst yol.)*
2. GitHub reposunda `record_id` içeren farklı bir dosya/sürüm olup olmadığını araştır (ör. repoda başka bir CSV varyantı).
3. Cluster-robust SE'yi tamamen atla, sadece tek bir ana analiz (kesitsel) raporla.

### B. Physical Activity süre kolonlarının hangi kategoriye ait olduğu codebook'tan kesin çıkarılamıyor

Codebook'taki sütun sırası: `numvigorous`(gün) → `nummoderate`(gün) → `hoursminutes1`("Hours:") → `hoursminutes4`("Minutes:") → `numwalking`(gün) → `hoursminutes2`("Hours:") → `hoursminutes3`("Minutes:") → `hoursminutes5`("Minutes:", oturma) → `hoursminutes6`("Hours:", oturma).

Standart IPAQ formunda 3 aktivite türünün (şiddetli/orta/yürüme) her biri için ayrı gün+süre çifti beklenir, yani 3 gün-sorusuna karşılık 3 süre-çifti gerekir. Burada ise **3 gün-sorusuna karşılık sadece 2 süre-çifti** var — yani şiddetli (vigorous) veya orta (moderate) şiddetli aktivitelerden birinin süresi bu ankette hiç toplanmamış, ama kolon adları ("hoursminutes1", "hoursminutes4" gibi) veya soru metinleri ("Hours:"/"Minutes:") bunun hangisi olduğunu belirtmiyor. Kolon sırası, ilk süre-çiftinin `nummoderate`'in hemen ardından geldiğini gösteriyor ki bu da orta şiddetli aktiviteye ait olabileceğine işaret ediyor, ama bu kesin değil — icat etmek istemiyorum.

**Önerim:** Bu belirsizliği icat etmeden iki paralel yol izlemek:
- **Ana proxy (belirsizlik içermez):** Haftalık toplam aktif gün sayısı = `numvigorous + nummoderate + numwalking` (0–21 gün/hafta). Kaba ama güvenilir.
- **Duyarlılık analizi (belirsizlik notuyla):** `numwalking × (hoursminutes2 + hoursminutes3/60)` (yürüme saatleri, net) + belirsiz kategori süresi (`hoursminutes1 + hoursminutes4/60`) × ilgili gün sayısı, "kategorisi belirsiz aktivite süresi" notuyla raporlanır.

### C. Demografik/fizyolojik kovaryatlar ve popülasyon farkı

`Age`, `Gender`, `BMI`, `Heart_Rate` LEMURS survey verisinde yok — modelden çıkarılacak (görevde zaten öngörülmüştü). Ayrıca LEMURS örneklemi tek bir üniversitenin (~600 birinci sınıf öğrencisi, UVM) dar yaş aralığındaki (~18-19) popülasyonundan geliyor; benim veri setim ise çok-ülkeli, 18–65 yaş aralığında genel yetişkin popülasyonu. Bu, dış geçerlilik karşılaştırmasının kapsamını sınırlıyor — limitations.md'de vurgulanacak.

---

## 4. Özet — Hangi değişkenler nasıl kullanılacak

| Kategori | Değişkenler |
|---|---|
| **Doğrudan kullanılabilir** | Sleep Duration (Y) → `F1_Sleep_sleephours` |
| **Proxy/dönüşüm gerektiren** | Stress (M) → PSS-10 toplam skoru (ters kodlamalı); Caffeine (X) → 6 içecek-türü bayrağının toplamı (0–6, doz değil çeşitlilik/sıklık proxy'si); Physical Activity (kovaryat) → haftalık aktif gün sayısı (ana) + saat bazlı duyarlılık analizi |
| **Modelden tamamen çıkarılacak** | Age, Gender, BMI, Heart_Rate — LEMURS'ta hiç yok |
| **Metodolojik kısıtlama** | Kişi-bazlı (record_id) gruplama ve cluster-robust SE — LEMURS dosyasında katılımcı kimliği yok, kesitsel havuzlama ile ikame edilmeli |

---

## Onay Gerekiyor

Adım 3'e (değişken inşası) ve mediation analizine geçmeden önce, yukarıdaki **Kritik Bulgu A** (record_id yok → kesitsel havuzlama öneriliyor) ve **Kritik Bulgu B**'deki (activity süre kategorisi belirsiz → gün-sayısı proxy'si öneriliyor) önerilen çözümleri onaylar mısınız, yoksa farklı bir yaklaşım mı istersiniz?
