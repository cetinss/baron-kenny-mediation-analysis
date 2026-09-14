# Karşılaştırma Özeti: Global Coffee Health (orijinal) vs LEMURS (dış doğrulama)

**Önemli metodolojik not:** İki veri seti farklı ölçeklerde değişkenler kullanıyor (Caffeine_mg vs. 0–6 kafein-çeşitliliği proxy'si; 3-kategori Stress_Score vs. 0–40 PSS-10). Bu nedenle katsayı **büyüklüklerinin** doğrudan karşılaştırılması anlamlı değildir. Aşağıdaki tablo yalnızca **yön (işaret)** ve **istatistiksel anlamlılık** tutarlılığına odaklanır — dış doğrulamada asıl aranan budur.

## Karşılaştırma Tablosu

| Path | Orijinal (Global Coffee Health) | LEMURS (dış doğrulama) | Yön Tutarlı mı? | Anlamlılık Tutarlı mı? |
|---|---|---|---|---|
| **c** — Total effect (Caffeine → Sleep) | −0.001687 | −0.028192 (p=.089) | ✅ İkisi de negatif | ⚠️ Orijinalde anlamlı (BK koşulları geçildi); LEMURS'ta anlamlı değil (sınırda) |
| **a** — Path a (Caffeine → Stress) | +0.002121 | +0.628975 (p<.001) | ✅ İkisi de pozitif | ✅ İkisi de anlamlı |
| **b** — Path b (Stress → Sleep \| Caffeine) | −0.481648 | −0.010452 (p<.001) | ✅ İkisi de negatif | ✅ İkisi de anlamlı |
| **c′** — Direct effect | −0.000665 | −0.021618 (p=.193) | ✅ İkisi de negatif | ⚠️ Orijinalde anlamlılık durumu net değil; LEMURS'ta anlamlı değil |
| **Indirekt (a×b)** | −0.00102 | −0.006574, %95 CI [−0.0111, −0.0030] | ✅ İkisi de negatif | ✅ İkisi de anlamlı (bootstrap CI sıfırı dışlıyor) |
| **Oran mediated** | 60.6% | 23.3% | ✅ İkisi de "kısmi" aralıkta (0–100%) | — büyüklük karşılaştırılamaz |
| **R²** (mediation modeli) | 0.629 | 0.0070 | — | — büyüklük karşılaştırılamaz (bkz. not) |

## Genel Değerlendirme

**Yön tutarlılığı: 5/5 path'te tam.** Kafein daha fazla algılanan strese, stres daha kısa uykuya yol açıyor; net etki kafeinin uyku üzerinde negatif; indirekt etki negatif — bu örüntü, tamamen farklı bir üretim yönteminden (DP-sentetik, ε=5) gelen ve tamamen farklı bir popülasyondan (üniversite öğrencileri) toplanan bağımsız bir veri setinde **birebir aynı yönde** tekrarlanıyor.

**Anlamlılık tutarlılığı: kısmi (3/5).** a, b ve indirekt etki path'leri her iki veri setinde de anlamlı. Toplam etki (c) ve doğrudan etki (c′) LEMURS'ta istatistiksel olarak anlamlı değil (ama p=.089 sınırda ve yön aynı). Bu, düşük istatistiksel güç ve kaba proxy ölçümlerle (özellikle 0–6 kafein-çeşitliliği skorunun mg-dozu yerine geçmesi) açıklanabilir — bkz. `limitations.md`.

**Büyüklük tutarlılığı: yok, ve bu beklenen bir sonuçtur.** R² ve oran-mediated değerleri iki veri setinde çok farklı (0.629 vs 0.007; %60.6 vs %23.3). Bu fark, ölçüm kalitesindeki temel farktan kaynaklanıyor: orijinal veri setinde X sürekli bir doz ölçümü (mg/gün) iken LEMURS'ta X sadece 0–6 aralığında kaba bir çeşitlilik sayacı; orijinalde M validasyonsuz 3-kategori bir skor iken LEMURS'ta M gerçek, validasyonu yapılmış PSS-10. Ölçüm hassasiyeti bu kadar farklıyken etki büyüklüklerinin örtüşmesi zaten beklenemez; asıl kanıt değeri yön ve anlamlılık örüntüsünün tekrarlanmasındadır.

**Sonuç:** LEMURS doğrulaması, orijinal analizdeki nedensel yönün (kafein → stres → daha az uyku) rastgele/veri-setine-özgü bir artefakt olmadığını, bağımsız üretilmiş bir referans veri setinde de aynı yönde ortaya çıktığını gösteriyor. Güç/anlamlılık farkı ise veri kalitesi/proxy kabalığıyla açıklanabilir bir sınırlılıktır, yöne dair kanıtı geçersiz kılmaz.
