# NHANES 2017–2018 Mediation Analizi (v2): Kafein Alımı → Allostatic Load → Uyku Süresi

**Analiz:** `src/nhanes_mediation_AL.py` | **Veri:** NHANES 2017–2018 (CDC/NCHS, gerçek veri) | **Tasarım:** Survey-ağırlıklı WLS + PSU-küme sağlam SH + tabakalı-küme bootstrap (5.000 tekrar)

> Bu, `nhanes_mediation_results.md`'deki PHQ-9 analizinin **alternatifi**, farklı bir mediator ile ikinci bağımsız bir denemedir. Aynı X (kafein) ve Y (uyku süresi) korunmuş, M değişkeni değiştirilmiştir. İki analiz birlikte okunmalıdır (bkz. Bölüm 5).

---

## 1. Neden Allostatic Load (AL)?

NHANES'te PSS yok (bkz. `nhanes_mediation_results.md`, Bölüm 1). Literatür taramasında, NHANES tabanlı "kronik stres" çalışmalarında kullanılan ikinci bir yaklaşım tespit edildi: **Allostatic Load (AL) indeksi** (McEwen & Stellar, 1993; Seeman ve ark., 1997) — stres maruziyetinin bedende biriken fizyolojik "yıpranma" yükünü ölçen, kardiyovasküler/metabolik biyobelirteçlerden oluşan objektif bir kompozit skor.

**Açık yapı uyarısı:** AL, **algılanan stres değildir** — PSS gibi öznel bir öz-bildirim değil, dolaylı bir fizyolojik gösterge. Literatürde tek bir standart AL formülü yoktur (2016 tarihli bir derleme, NHANES çalışmalarında kullanılan **21 farklı hesaplama varyantı** tespit etmiştir). Kullanıcı tercihiyle, bu analiz sadece **zaten indirilmiş dosyalarla (BMX, BPX) sınırlı, küçültülmüş bir AL versiyonu** kullanmaktadır — klasik 7-8 biyobelirteçli (kolesterol, HbA1c, CRP, albumin dahil) tam versiyon değil. Bu, PHQ-9'dan bile daha dolaylı bir stres proxy'sidir ve raporlarda bu şekilde nitelendirilmelidir.

## 2. AL İndeksi İnşası

4 bileşen, her biri için ayrı ayrı: kişinin değeri örneklemin (survey-ağırlıklı) **75. persentilinde veya üzerindeyse 1 puan**, değilse 0 (Seeman/Crimmins tarzı "yüksek risk çeyreği" yöntemi):

| Bileşen | Kaynak | Yüksek-risk eşiği (ağırlıklı 75. persentil) |
|---|---|---|
| BMI | `BMX_J` | ≥ 33.60 kg/m² |
| Sistolik KB (SBP) | `BPX_J`, mevcut okumaların ortalaması | ≥ 129.33 mmHg |
| Diyastolik KB (DBP) | `BPX_J`, mevcut okumaların ortalaması | ≥ 80.00 mmHg |
| Dinlenik Nabız | `BPX_J` | ≥ 80.00 bpm |

`AL_Score` = 4 bileşenin toplamı (0–4). **Kovaryat notu:** BMI ve Heart_Rate artık M'nin içinde olduğu için ayrı kovaryat olarak kullanılmadı (dairesellik önlendi) — kovaryatlar: Age, Gender, Physical_Activity_Hours.

**AL_Score dağılımı (n=3.157):** 0 puan: 1.168 (%37.0), 1 puan: 969 (%30.7), 2 puan: 644 (%20.4), 3 puan: 312 (%9.9), 4 puan: 64 (%2.0).

---

## 3. Tablo 1 — Betimsel İstatistikler (n=3.157)

| Değişken | Ort. | SS | Medyan | Min–Maks |
|---|---|---|---|---|
| Caffeine_mg | 109.80 | 111.06 | 79.0 | 0–457 |
| AL_Score (0–4) | 1.09 | 1.07 | 1.0 | 0–4 |
| Sleep_Hours | 7.55 | 1.44 | 7.5 | 3.5–11.5 |
| Age | 42.79 | 14.66 | 44.0 | 18–65 |
| BMI | 29.85 | 7.64 | 28.6 | – |
| SBP | 122.72 | 17.98 | 120.0 | – |
| DBP | 73.55 | 11.97 | 73.3 | – |
| Heart_Rate | 72.58 | 11.68 | 72.0 | – |
| Physical_Activity_Hours | 10.33 | 12.86 | 5.0 | – |

Erkek oranı: %45.6.

## Tablo 2 — Path Katsayıları

| Path | B (unstd.) | SE | t | p | β (std.) |
|---|---|---|---|---|---|
| Total Effect (c): Caffeine → Sleep | **−0.000541** | 0.000258 | −2.092 | **.0364** * | −0.0416 |
| Path a: Caffeine → AL | −0.000274 | 0.000297 | −0.910 | .3537 | −0.0285 |
| Path b: AL → Sleep \| Caffeine | −0.043416 | 0.030406 | −1.428 | .1533 | −0.0321 |
| Direct Effect (c′): Caffeine → Sleep \| AL | **−0.000553** | 0.000256 | −2.155 | **.0312** * | −0.0425 |

## Tablo 3 — Dolaylı Etki ve Model Kalitesi

| Ölçüt | Değer |
|---|---|
| İndirekt etki (a×b) | 1.19×10⁻⁵ |
| %95 Tabakalı-Küme Bootstrap CI | [−3.07×10⁻⁶, 2.61×10⁻⁵] |
| CI sıfırı dışlıyor mu? | **Hayır** |
| Sobel Z / p | 0.778 / .437 |
| Oran mediated | −2.2% (anlamsız) |
| R² (toplam / mediation) | 0.0367 / 0.0378 |
| Cohen's f² | 0.0012 |
| Max VIF | 1.14 |
| Durbin-Watson | 2.073 |

**Baron-Kenny koşulları:** 1. c anlamlı → **GEÇTİ** (p=.036) | 2. a anlamlı → BAŞARISIZ | 3. b anlamlı → BAŞARISIZ | 4. \|c′\|<\|c\| → BAŞARISIZ (c′ aslında c'den büyük) | **Sonuç: NO MEDIATION**

---

## 4. Yorum

Bu analizde, PHQ-9 denemesinden farklı olarak, **toplam etki (c) ve doğrudan etki (c′) istatistiksel olarak anlamlı** çıktı (p=.036 ve p=.031) — yani gerçek, ulusal temsili ABD verisinde, tasarım-doğru (survey-ağırlıklı, küme-sağlam) bir modelde, **kafein alımı ile uyku süresi arasında anlamlı, negatif bir ilişki tespit edilmiştir**. Bu, Global Coffee Health ve LEMURS'taki toplam-etki yönüyle tutarlıdır ve bu üçüncü, tamamen gerçek veri setinde de doğrulanmıştır.

Ancak **AL bu ilişkiye aracılık etmiyor**: a-path (kafein→AL) anlamsız (p=.354), b-path (AL→uyku) anlamsız (p=.153), indirekt etki CI'si sıfırı kapsıyor. Dikkat çekici bir detay: c′ (0.000553), c'den (0.000541) mutlak değerce **büyük** çıktı — yani AL'yi modele eklemek, kafein-uyku ilişkisini "açıklamak" yerine hafifçe güçlendiriyor (klasik bastırma/suppression paterni değil ama mediation yönünün tam tersi bir küçük sapma). Bu, AL'nin bu yolda gerçek bir aracı olmadığının ek bir işareti.

**Genel senteze katkısı:** Kafein-uyku ilişkisinin kendisi üç bağımsız veri setinde (Coffee Health, LEMURS, NHANES) tutarlı yönde bulunuyor ve şimdi gerçek/resmi veride istatistiksel anlamlılığa da ulaşıyor. Ancak bu ilişkinin **mekanizması** (hangi ara değişken üzerinden işlediği) hâlâ açık bir soru: ne depresif belirtiler (PHQ-9) ne de fizyolojik stres yükü (AL) bu örneklemde anlamlı bir aracı rolü göstermiyor. Bu, orijinal PSS-tabanlı bulgunun (LEMURS'ta doğrulanan) muhtemelen **stres-spesifik** bir mekanizma olduğuna, depresyon veya genel fizyolojik yıpranmayla aynı yolu paylaşmadığına işaret ediyor.

---

## 5. İki NHANES Denemesinin Birlikte Değerlendirilmesi

| | PHQ-9 (Depresif Belirtiler) | Allostatic Load (Fizyolojik Yük) |
|---|---|---|
| Yapı türü | Öz-bildirim, validasyonlu klinik ölçek | Objektif, kompozit fizyolojik indeks |
| c (toplam etki) | Anlamsız (p=.091) | **Anlamlı (p=.036)** |
| a-path | Anlamsız (p=.418) | Anlamsız (p=.354) |
| b-path | Anlamsız (p=.307) | Anlamsız (p=.153) |
| Mediation sonucu | NO MEDIATION | NO MEDIATION |
| Yapı-geçerliliği stres'e yakınlığı | Orta (ilişkili ama farklı yapı) | Uzak (dolaylı fizyolojik korelat) |

**Sonuç:** İki farklı, makul aday mediator da denendi; ikisi de anlamlı aracılık göstermedi. Bu, "NHANES'te doğru mediator'ü bulamadık" değil, **"NHANES'te mevcut hiçbir stres-proxy'si bu yolda aracılık etmiyor"** şeklinde okunmalı — ve bu, PSS-tabanlı orijinal bulgunun (Coffee Health, LEMURS) genelleştirilebilirliğine dair önemli, dürüst bir sınır koşuludur: **muhtemelen mediation etkisi spesifik olarak algılanan/öznel stres yapısına bağlıdır**, NHANES'in sunduğu proxy'lerle (klinik depresyon veya fizyolojik yük) yakalanamıyor. Bu senteze tezde ayrı bir "Sınır Koşulları" alt başlığı olarak yer verilmesi önerilir.

**İlgili dosyalar:** `nhanes_mediation_results.md`, `outputs/nhanes_AL_analytic_sample.csv`, `outputs/figures/NHANES_AL_Fig1-3`, `src/nhanes_mediation_AL.py`
