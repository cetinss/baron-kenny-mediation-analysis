# LEMURS Dış Doğrulama Sonuçları

**Analiz:** `src/lemurs_validation.py` | **Veri:** `data/lemurs/survey_syn_5.csv` (DP-sentetik referans, ε=5, n=3,852 person-week) | **Bootstrap:** 5,000 resample, %95 percentile CI

**Yöntem notu:** `record_id` LEMURS dosyasında bulunmadığı için (bkz. `feature_matching_report.md`, Kritik Bulgu A), her satır (person-week) bağımsız bir kesitsel gözlem olarak ele alınmıştır — kişi-bazlı ortalama veya cluster-robust SE uygulanmamıştır.

---

## Örneklem

| Adım | n | Not |
|---|---|---|
| Ham veri | 3,852 | person-week satırı |
| Eksik veri temizliği sonrası | 3,215 | sleep_hours, pss_score, caffeine_exposure_score, activity_days'te eksiksiz |
| IQR (×1.5) aykırı değer temizliği sonrası | **2,953** | sleep_hours: 251, pss_score: 5, activity_days: 6 aykırı değer çıkarıldı |
| Toplam çıkarılan | 899 (%23.3) | |

## Tablo 1 — Betimsel İstatistikler (n = 2,953)

| Değişken | Ort. | SS | Medyan | Min | Maks |
|---|---|---|---|---|---|
| Caffeine Exposure Score (0–6, proxy) | 1.020 | 1.080 | 1.0 | 0 | 5 |
| PSS-10 Toplam Skoru (0–40) | 15.824 | 7.014 | 16.0 | 0 | 36 |
| Sleep Duration (saat/gece) | 7.195 | 0.973 | 7.0 | 5.8 | 9.4 |
| Active Days/Week (0–21, kovaryat) | 10.482 | 3.594 | 10.0 | 1 | 20 |

## Tablo 2 — Path Katsayıları (Ana Analiz, kovaryat = activity_days)

| Path | B (unstd.) | SE | t | p | β (std.) |
|---|---|---|---|---|---|
| Total Effect (c): Caffeine → Sleep | −0.028192 | 0.016580 | −1.700 | .089 | −0.0313 |
| Path a: Caffeine → PSS-10 | **+0.628975** | 0.118980 | 5.286 | **1.34e-07** *** | +0.0969 |
| Path b: PSS-10 → Sleep \| Caffeine | **−0.010452** | 0.002559 | −4.085 | **4.53e-05** *** | −0.0753 |
| Direct Effect (c′): Caffeine → Sleep \| PSS-10 | −0.021618 | 0.016614 | −1.301 | .193 | −0.0240 |

## Tablo 3 — Dolaylı Etki (İndirekt Etki)

| Ölçüt | Değer |
|---|---|
| İndirekt etki (a × b) | **−0.006574** |
| %95 Percentile Bootstrap CI | **[−0.011124, −0.002972]** |
| CI sıfırı dışlıyor mu? | **Evet** |
| Sobel Z | −3.232 |
| Sobel p | .00123 |
| Oran mediated (indirekt/c) | 23.3% |

## Tablo 4 — Baron-Kenny Koşulları

| Koşul | Sonuç |
|---|---|
| 1. c anlamlı (X→Y toplam etki) | **BAŞARISIZ** (p=.089) |
| 2. a anlamlı (X→M) | GEÇTİ (p<.001) |
| 3. b anlamlı (M→Y\|X) | GEÇTİ (p<.001) |
| 4. \|c′\| < \|c\| | GEÇTİ (0.0216 < 0.0282) |
| **Klasik Baron-Kenny sonucu** | **"NO MEDIATION"** (koşul 1 nedeniyle) |

## Tablo 5 — Model Kalitesi

| Ölçüt | Toplam-Etki Modeli | Mediation Modeli |
|---|---|---|
| R² | 0.0013 | 0.0070 |
| Adj. R² | 0.0007 | 0.0060 |
| F | 1.984 (p=.138) | 6.891 (p<.001) |
| Cohen's f² (mediatör eklenmesi) | — | 0.0057 |
| Max VIF | — | 1.01 (çoklu doğrusal bağlantı yok) |
| Durbin-Watson | — | 1.992 (otokorelasyon yok) |

---

## Yorum

Klasik Baron-Kenny adım-adım anlamlılık testine göre sonuç **"NO MEDIATION"** olarak sınıflanıyor, çünkü Adım 1'deki toplam etki (c) istatistiksel olarak anlamlı değil (p=.089, sınırda). Ancak bu, modern mediation literatüründe (Hayes, Preacher & Hayes) bilinen bir sınırlamadır: Baron-Kenny'nin toplam-etki-önce-anlamlı-olmalı koşulu günümüzde terk edilmiş bir kriterdir; asıl kanıt standardı **bootstrap ile indirekt etkinin (a×b) güven aralığının sıfırı dışlayıp dışlamadığıdır**. Bu kritere göre LEMURS'ta indirekt etki **anlamlıdır** (%95 CI [−0.0111, −0.0030], sıfırı dışlıyor; Sobel p=.001).

Daha önemlisi, **yön (işaret) tutarlılığı tam**:
- a-path pozitif (daha fazla kafein türü tüketimi → daha yüksek PSS-10 stres skoru) — orijinal analizle aynı yön.
- b-path negatif (daha yüksek stres → daha kısa uyku) — orijinal analizle aynı yön.
- İndirekt etki (a×b) negatif — orijinal analizle aynı yön.
- Toplam etki (c) ve doğrudan etki (c′) de negatif — orijinal analizle aynı yön, ama burada istatistiksel olarak anlamlı değil.

Etki büyüklükleri (R²≈0.001–0.007, Cohen's f²≈0.006) orijinal analizdeki (R²=0.629) ile kıyaslanamayacak kadar küçük. Bu beklenen bir durumdur çünkü: (1) kafein değişkeni burada mg/gün değil 0–6 aralığında kaba bir "çeşitlilik" proxy'sidir, (2) LEMURS örneklemi dar bir popülasyondan (tek üniversite, ~18-19 yaş) gelir, (3) DP gürültüsü (ε=5) marjinal ilişkileri bir miktar zayıflatabilir. Büyüklük karşılaştırması bu nedenle anlamlı değildir — ayrıntılı karşılaştırma için bkz. `comparison_summary.md`.

## Duyarlılık Analizi — Süre-Bazlı Aktivite Kovaryatı (Kritik Bulgu B)

Ana analizde kovaryat olarak belirsizlik içermeyen "haftalık aktif gün sayısı" kullanılmıştır. Kategori belirsizliği taşıyan süre-bazlı proxy (`activity_hours_sensitivity`, n=2,923) ile tekrarlanan analiz pratik olarak aynı sonucu veriyor:

| Ölçüt | Ana Analiz (gün-sayısı) | Duyarlılık (süre-bazlı, belirsiz kategori) |
|---|---|---|
| a | +0.6290 (p<.001) | +0.6261 (p<.001) |
| b | −0.01045 (p<.001) | −0.01065 (p<.001) |
| c | −0.0282 (p=.089) | −0.0297 (p=.074) |
| c′ | −0.0216 (p=.193) | −0.0231 (p=.166) |
| İndirekt (a×b) | −0.00657, CI [−0.0111, −0.0030] | −0.00667, CI [−0.0114, −0.0030] |
| % mediated | 23.3% | 22.4% |

Sonuçların kovaryat seçimine duyarlı olmadığı görülüyor — bu, Kritik Bulgu B'deki belirsizliğin ana sonuçları tehdit etmediğini gösteriyor.

**Çıktı dosyaları:** `outputs/figures/LEMURS_Fig1_Distributions.png`, `LEMURS_Fig2_Path_Diagram.png`, `LEMURS_Fig3_Bootstrap.png`, `LEMURS_Fig4_Diagnostics.png`, `outputs/lemurs_descriptive_statistics.csv`
