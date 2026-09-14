# NHANES 2017–2018 Mediation Analizi: Kafein Alımı → Depresif Belirtiler (PHQ-9) → Uyku Süresi

**Analiz:** `src/nhanes_mediation_analysis.py` | **Veri:** NHANES 2017–2018 (CDC/NCHS, ABD ulusal temsili sağlık anketi, gerçek veri) | **Tasarım:** Survey-ağırlıklı WLS + PSU-küme sağlam SH + tabakalı-küme bootstrap (5.000 tekrar)

> **Bu, Global Coffee Health veya LEMURS analizlerinin bir doğrulaması DEĞİLDİR.** Bağımsız, kendi başına akademik bir mediation çalışmasıdır. Baron-Kenny çerçevesi ve bootstrap yaklaşımı önceki analizlerle tutarlı tutulmuştur, ama **mediator farklıdır** (aşağıya bakınız) ve **NHANES'in karmaşık örnekleme tasarımı** dikkate alınmıştır.

---

## 1. Yapı (Construct) Notu — Neden "Stres" Değil "Depresif Belirtiler"

NHANES'te PSS (Perceived Stress Scale) gibi bir stres ölçeği **yoktur**, kortizol biyobelirteci bu döngüde ölçülmemiştir. Mevcut en yakın validasyonlu psikolojik ölçüm **PHQ-9 depresyon tarama ölçeğidir** (`DPQ010`–`DPQ090`, 0–3 Likert × 9 madde, toplam 0–27). Kullanıcıyla birlikte alınan karara göre (bkz. konuşma geçmişi), mediator burada açıkça **"depresif belirtiler"** olarak adlandırılmıştır — "stres" kelimesi kullanılmamıştır. Bu, LEMURS'taki PSS-10 tabanlı stres ölçümünden **farklı bir psikolojik yapıdır**; sonuçlar stres-mediation bulgularıyla doğrudan kıyaslanmamalıdır.

---

## 2. Veri ve Yöntem

**Kaynak dosyalar (CDC, doğrudan indirildi, 5.078-değişkenli birleşik mega-dosya kullanılmadı):** `DEMO_J` (demografi + tasarım değişkenleri), `DR1TOT_J` (diyet-kafein + `WTDRD1` ağırlığı), `BMX_J` (BMI), `BPX_J` (nabız), `PAQ_J` (GPAQ fiziksel aktivite), `SLQ_J` (uyku), `DPQ_J` (PHQ-9).

**Değişkenler:**
- X = `DR1TCAFF` (kafein, mg — tek günlük 24 saatlik diyet hatırlama)
- M = PHQ-9 toplam skoru (9 madde toplamı, 0–27; 7=Refused/9=Don't know → eksik değer, prorasyon yok)
- Y = `SLD012` (hafta içi alışılmış uyku saati)
- Kovaryatlar: Age, Gender, BMI, Physical_Activity_Hours (GPAQ, 5 kategori × gün × dakika/60), Heart_Rate

**Örnekleme tasarımı:** NHANES çok-aşamalı, tabakalı, küme örneklemi kullanır ve eşit olmayan seçilme olasılıklarına sahiptir. Basit OLS ile "sanki basit rastgele örneklemmiş" gibi analiz etmek, popülasyonu yanlış temsil eder — bu, hakemlerin sık karşılaştığı bir metodolojik itiraz noktasıdır. Bu nedenle:
- Tüm regresyonlar **WLS** ile, diyet-günü örnekleme ağırlığı `WTDRD1` (ortalama 1'e ölçeklendirilmiş) kullanılarak kuruldu — kafein diyet-temelli bir değişken olduğu için NHANES analitik kılavuzlarına göre doğru ağırlık budur.
- Standart hatalar **`SDMVSTRA` × `SDMVPSU`** kümelerine göre sağlamlaştırıldı (cluster-robust).
- **Sınırlılık:** Bu, tam Taylor-serisi lineerizasyon varyans tahmininin (R'nin `survey` paketi gibi) basitleştirilmiş bir yaklaşımıdır; NHANES'e özgü replicate-weight yöntemleri kullanılmamıştır — bu açıkça bir sınırlılık olarak belirtilmelidir.
- Bootstrap, satırları değil **PSU'ları tabaka içinde yeniden örnekleyen tasarım-farkında (stratified cluster) bootstrap** olarak uygulandı (5.000 tekrar) — iid satır varsayımı yapılmadı.

**Temizleme:** Yaş 18–65 filtresi + 7 anahtar değişkende eksiksiz satır → n=3.556. Sürekli değişkenlerde (Caffeine_mg, Sleep_Hours, BMI, Heart_Rate, Physical_Activity_Hours) 1.5×IQR aykırı değer temizliği uygulandı. **PHQ-9 bilinçli olarak bu temizlemeden hariç tutuldu** — ilk denemede IQR uygulanınca >10 puan alan (orta-şiddetli depresif belirti) 299 kişi "aykırı değer" olarak silinmiş olduğu fark edildi; bu, çalışmanın asıl ilgilendiği klinik olarak anlamlı grubu veriden çıkarmak anlamına geldiği için düzeltildi. Final örneklem: **n = 2.927** (15 tabaka, 30 küme).

---

## 3. Tablo 1 — Betimsel İstatistikler (n=2.927, ağırlıksız)

| Değişken | Ort. | SS | Medyan | Çarpıklık | Min–Maks |
|---|---|---|---|---|---|
| Caffeine_mg | 112.10 | 112.44 | 84.0 | 0.98 | 0–468 |
| PHQ9_Score | 3.12 | 4.12 | 2.0 | 1.98 | 0–24 |
| Sleep_Hours | 7.55 | 1.43 | 7.5 | −0.04 | 3.5–11.5 |
| Age | 42.93 | 14.68 | 44.0 | −0.15 | 18–65 |
| BMI | 29.18 | 6.46 | 28.3 | 0.51 | 14.8–47.6 |
| Physical_Activity_Hours | 10.50 | 13.10 | 5.0 | 1.45 | 0–51 |
| Heart_Rate | 71.81 | 10.54 | 72.0 | 0.24 | 44–98 |

Erkek oranı: %46.9.

## Tablo 2 — Path Katsayıları (Survey-Ağırlıklı WLS, Küme-Sağlam SH)

| Path | B (unstd.) | SE | t | p | β (std.) |
|---|---|---|---|---|---|
| Total Effect (c): Caffeine → Sleep | −0.000499 | 0.000295 | −1.691 | .0907 | −0.0393 |
| Path a: Caffeine → PHQ-9 | −0.000665 | 0.000811 | −0.809 | .4183 | −0.0179 |
| Path b: PHQ-9 → Sleep \| Caffeine | −0.009376 | 0.009179 | −1.021 | .3070 | −0.0271 |
| Direct Effect (c′): Caffeine → Sleep \| PHQ-9 | −0.000505 | 0.000292 | −1.728 | .0839 | −0.0398 |

## Tablo 3 — Dolaylı Etki

| Ölçüt | Değer |
|---|---|
| İndirekt etki (a×b) | 6.15×10⁻⁶ |
| %95 Tabakalı-Küme Bootstrap CI | [−1.03×10⁻⁵, 2.10×10⁻⁵] |
| CI sıfırı dışlıyor mu? | **Hayır** |
| Sobel Z / p | 0.634 / .526 |
| Oran mediated | −1.2% (anlamsız, yorumlanamaz) |

## Tablo 4 — Baron-Kenny Koşulları ve Model Kalitesi

| Koşul | Sonuç |
|---|---|
| 1. c anlamlı | BAŞARISIZ (p=.091) |
| 2. a anlamlı | BAŞARISIZ (p=.418) |
| 3. b anlamlı | BAŞARISIZ (p=.307) |
| 4. \|c′\|<\|c\| | BAŞARISIZ (0.000505 > 0.000499) |
| **Sonuç** | **NO MEDIATION** |

| Model kalitesi | Değer |
|---|---|
| R² (toplam-etki modeli) | 0.0372 |
| R² (mediation modeli) | 0.0379 |
| Cohen's f² | 0.0007 |
| Max VIF | 1.12 (çoklu doğrusal bağlantı yok) |
| Durbin-Watson | 2.053 (otokorelasyon yok) |

---

## 4. Yorum

**Bu bir "hata" veya başarısız pipeline değil — gerçek, dürüst bir null bulgu.** Ham (ağırlıksız) ikili korelasyonlar bile aynı tabloyu doğruluyor: Caffeine–PHQ9 r=−0.0085 (p=.64), PHQ9–Sleep r=−0.029 (p=.12). Bu örneklemde kafein alımı ile PHQ-9 depresif belirti skoru arasında pratik olarak **hiçbir ilişki yok** (a-path baştan zayıf) — dolayısıyla üzerinden bir dolaylı etki geçebilecek bir "a köprüsü" mevcut değil. Toplam etki (c) sınırda (p=.091) ve yönü negatif (daha fazla kafein → biraz daha az uyku) — bu, Global Coffee Health ve LEMURS'taki toplam-etki yönüyle tutarlı, ama burada istatistiksel olarak anlamlılığa ulaşmıyor.

**Bunun Coffee Health/LEMURS bulgularını geçersiz kılmadığını vurgulamak önemli:** M değişkeni kökten farklı bir yapı (klinik depresyon belirtisi vs. algılanan stres). Kafeinin algılanan stresle ilişkili olup depresif belirtilerle ilişkili olmaması, literatürde de bilinen bir ayrımdır — stres ve depresyon örtüşen ama farklı yapılardır, farklı nöro-davranışsal mekanizmalarla ilişkilenebilirler. Bu null bulgu, tezin ana bulgusuna bir tehdit değil, **ayrı ve dürüst bir katkıdır**: "kafein–uyku ilişkisi depresif belirtiler üzerinden değil, muhtemelen başka bir mekanizmayla (örn. algılanan stres, doğrudan farmakolojik uyarılma) işliyor olabilir" şeklinde bir sınır koşulu tartışmasına zemin sağlar.

---

## 5. Sınırlılıklar

1. **Yapı ikamesi:** PHQ-9 ≠ PSS. Sonuçlar "stres" hakkında değil, "depresif belirtiler" hakkındadır.
2. **Kesitsel tasarım:** X, M, Y aynı ziyarette/aynı hafta ölçülüyor — zamansal öncelik kurulamaz, nedensellik iddia edilemez.
3. **Tek-günlük diyet hatırlama:** `DR1TCAFF` tek bir 24 saatlik hatırlamaya dayanıyor; NHANES'in kendi metodolojik notlarına göre bu, "alışılmış alım" için gürültülü bir ölçümdür (gün-içi/günler-arası varyasyon karıştırılmış olabilir). İki-günlük ortalama (`DR1TCAFF`+`DR2TCAFF`) veya NCI usual-intake yöntemi daha isabetli olurdu ama bu analizin kapsamı dışında tutuldu.
4. **Yaklaşık varyans tahmini:** PSU-küme sağlam SH, tam Taylor-serisi lineerizasyonun (veya replicate-weight yönteminin) basitleştirilmiş bir yaklaşımıdır.
5. **Tek döngü, tek ülke:** Sadece 2017–2018, sadece ABD — LEMURS'un tek-üniversite örneklemine benzer şekilde, küresel genellenebilirlik yoktur.
6. **IQR temizliği:** PHQ-9 hariç tutuldu (yukarıda açıklandı), ama diğer değişkenlerdeki (özellikle Physical_Activity_Hours) IQR temizliği gerçek ağır-kuyruklu dağılımı bir miktar kırpıyor olabilir (bkz. `distributional_realism_check.md`, Bölüm 3).

---

## 6. "Major Revision" Sorusuna Dürüst Değerlendirme

Bu analiz, gerçek/resmi bir veri setiyle kurulmuş, tasarım-farkında (survey-weighted, cluster-robust, stratified bootstrap) bir mediation çalışmasıdır — metodolojik olarak "sentetik veri" eleştirisinin bir kademe ötesine geçer. Ama bir null bulgu ile geldi. Bu **iyi ya da kötü değil, dürüst** — ve tezde şu şekillerde kullanılabilir:
- **Ek bir bölüm/duyarlılık analizi olarak:** "Farklı bir gerçek veri setinde, farklı bir psikolojik yapıyla (depresif belirtiler), mediation bulunamamıştır — bu, orijinal bulgunun stres-spesifik olduğunu ve genellenebilirliğinin dikkatle sınırlandırılması gerektiğini gösterir." Bu çerçeve, hakemlere sizin sınırlılıkları içtenlikle araştırdığınızı gösterir — genelde olumlu karşılanır.
- **Riskler:** Bir hakem "neden PSS yerine PHQ-9 kullandınız" diye sorabilir (yapı ikamesini net açıklamak şart, bu raporda yapıldı); bir başkası kesitsel-veri + mediation nedensellik sorununu gündeme getirebilir (her üç analizde de ortak bir sınırlılık, ayrı bir savunma gerektirmez).

**Sonuç:** Bu ek analiz, "major revision alır mısınız" sorusuna kesin bir "hayır" garantisi vermez (hiçbir analiz veremez), ama elinizdeki kanıt tabanını nitel olarak güçlendirir: artık üç farklı veri kalitesi seviyesinde (sentetik → DP-sentetik-ama-doğrulanmış → tam gerçek/resmi) tutarlı bir metodolojik disiplin ve şeffaflıkla çalışılmış bir dosyanız var.

**İlgili dosyalar:** `outputs/nhanes_mediation_analytic_sample.csv`, `outputs/figures/NHANES_Med_Fig1-4`, `src/nhanes_mediation_analysis.py`

**Devamı:** Bu null bulgudan sonra, farklı bir mediator adayıyla (Allostatic Load, fizyolojik stres-yükü proxy'si) ikinci bir bağımsız deneme yapılmıştır — bkz. `nhanes_mediation_AL_results.md`. İki denemenin birleşik değerlendirmesi o dosyanın Bölüm 5'indedir.
