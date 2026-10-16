# Project #108b — Measurement-Function Invariance in LLM Agents

## 1. Proje Özeti

Bu proje, LLM ve tool-using agent'ların **gerçek dünyadaki değişim** ile yalnızca **ölçüm / gözlem / raporlama fonksiyonundaki değişimi** ayırt edip edemediğini test eder.

Ana ayrım:

> **Sensitivity to true world changes vs. invariance to instrument-induced observational changes**

Temel nedensel yapı:

- Latent state sabit olabilir: `x1 = x2`
- Ölçüm fonksiyonu değişebilir: `f1 != f2`
- Gözlenen değer değişebilir: `y1 != y2`

Ancak:

`Δy != 0` **gerçek dünyada** `Δx != 0` anlamına gelmez.

Araştırmanın merkezindeki soru:

> **LLM agent, gözlemdeki değişimi gerçek dünyadaki değişim sanıyor mu?**

---

## 2. Ana Novelty Pozisyonu

Ana claim:

> **We distinguish sensitivity to true world changes from invariance to instrument-induced observational changes.**

Daha temkinli novelty claim:

> **To our knowledge, this is the first controlled evaluation of whether LLM agents remain invariant to changes in the measurement function when the underlying latent state is held fixed by construction.**

Kesinlikle yapılmaması gereken claim'ler:

- "First to study invariance in LLMs."
- "First counterfactual evaluation of clinical agents."
- "We show LLMs struggle with reference ranges."
- "First to identify measurement-vs-world confusion."
- "Novel benchmark for spurious trend detection."

Novelty, **invariance kavramının kendisinde değil**; latent state'in deneysel olarak sabit tutulduğu measurement-function intervention'dadır.

---

## 3. Formal Problem

Gerçek latent durum:

`x_t`

Ölçüm sistemi:

`y_t = f_t(x_t)`

Ana hata senaryosu:

`x1 = x2`

ama:

`f1 != f2`

dolayısıyla:

`y1 != y2`

Modelin yapmaması gereken çıkarım:

`y1 != y2 -> x1 != x2`

Doğru davranış:

- measurement provenance'ı kontrol etmek,
- gerekiyorsa normalize / invert etmek,
- karşılaştırma mümkün değilse bunu belirtmek,
- gerçek trend yoksa sahte trend üretmemek.

---

## 4. 2×2 Faktöriyel Tasarım

| Latent state | Measurement function | Doğru davranış |
|---|---|---|
| Aynı | Aynı | Değişim yok |
| **Aynı** | **Değişti** | **Değişim yok — ana deney** |
| Değişti | Aynı | Gerçek değişimi tespit et |
| Değişti | Değişti | Measurement function'ı hesaba katarak latent değişimi çıkar |

Ana novelty özellikle **Cell #2** üzerindedir.

Cell #3 ve Cell #4'ün sensitivity tarafı, mevcut counterfactual evaluation çalışmalarına daha yakındır.

---

## 5. Domainler

### 5.1 Klinik

Örnek:

Eski assay:

- Result = 80
- Reference range = 0–100

Yeni assay:

- Result = 120
- Reference range = 0–150

Normalize edildiğinde:

`80 / 100 = 120 / 150 = 0.8`

Underlying biological state değişmemiştir.

Yanlış LLM yorumu:

> "Değer %50 arttı."

Doğru yorum:

> Measurement / assay değişmiş olabilir; ham değerler doğrudan karşılaştırılamaz.

### 5.2 Endüstriyel Sensör

Gerçek sıcaklık:

`x = 50`

Sensor A:

`y = x`

Sensor B:

`y = 1.2x + 5`

Yeni sensör 65 gösterir ama fiziksel sıcaklık hâlâ 50'dir.

Test:

> Agent bunu process anomaly / gerçek sıcaklık artışı sanıyor mu?

### 5.3 Finans

Underlying ekonomik durum sabit tutulur, fakat:

- accounting standard,
- reporting policy,
- restatement convention,
- segment definition,
- index methodology

değiştirilir.

Test:

> Agent ham YoY değişimini ekonomik değişim sanıyor mu, yoksa reporting-function değişimini hesaba katıyor mu?

---

## 6. Provenance Salience Seviyeleri

### L0 — Explicit

Açıkça:

> "Measurement method changed on 2025-03-01."

Bu yalnızca ceiling / sanity check olarak kullanılabilir.

### L1 — Implicit Structured Signal

Sadece şunlardan biri değişir:

- reference range
- calibration coefficients
- unit scale
- assay metadata

**Asıl deney seviyesi budur.**

### L2 — Embedded Provenance

Bilgi gömülü bir alandadır:

- method_id
- assay_code
- sensor_version
- calibration_version

Modelin metadata'yı aktif biçimde kullanması gerekir.

### L3 — Provenance Missing

Measurement değişiminin açıklaması verilmez.

Doğru davranış:

- belirsizlik belirtmek,
- karşılaştırmanın güvenilir olmadığını söylemek,
- blanket trend iddiasından kaçınmak.

---

## 7. Zorunlu Competence Pre-Test

L1 başarısızlığı doğrudan measurement-invariance failure olarak yorumlanamaz.

Önce modelin şu temel yetenekleri gösterdiği doğrulanmalıdır.

### Reference Range Competence

Örnek:

> Reference range = 0–150. Result = 120. Bu sonuç range içinde mi?

### Affine Inversion Competence

Örnek:

`y = 1.2x + 5`

`y = 65`

ise:

`x = 50`

Model bunlarda başarısızsa ana deneydeki hata deployment failure değil, temel capability failure olabilir.

**Pilot competence barajı: ≥ %90.**

---

## 8. Direct vs Code-Agent Dissociation

Projenin en önemli olası bulgularından biri:

Model prose reasoning sırasında provenance'ı doğru kullanabilir fakat coding agent olarak aynı bilgiyi kaybedebilir.

Örnek doğru prose:

> "Assay değiştiği için ham değerler doğrudan karşılaştırılmamalı."

Ama aynı model şu kodu yazabilir:

```python
df["pct_change"] = df["value"].pct_change()
```

ve:

- `method_id`
- `calibration_version`
- `reference_range`

sütunlarını sessizce yok sayabilir.

Bu nedenle direct prompting tek başına yeterli değildir.

Ölçülecek ana kod davranışları:

- provenance sütunları korunuyor mu?
- assay / sensor version'a göre split veya group ediliyor mu?
- calibration tersine çevriliyor mu?
- normalize ediliyor mu?
- ham `diff()` / `pct_change()` ile sahte trend yaratılıyor mu?

---

## 9. Pilot Tasarımı

Pilotun amacı paper sonucu üretmek değil, #108b'nin deneysel olarak gerçekten ilginç olup olmadığını falsifiye etmektir.

### Pilot Kapsamı

- 4 model
- fictional / synthetic domain
- 2×2 latent × measurement design
- L1 ve L2 ağırlıklı
- minimal code-agent kolu
- hard time-series varyantı
- competence pre-test

### L0

Pilotun ana bütçesinden çıkarılır.

Sadece çok küçük bir sanity check gerekirse eklenebilir.

### Minimal Code-Agent Kolu

Özellikle:

- Cell #2
- Cell #3
- L1
- 20 örnek / hücre / model

### Hard Varyant

- 30 zaman noktası
- yaklaşık %5 measurement noise
- iki analit / iki sensör
- measurement switch serinin ortasında
- L1 provenance

Amaç clean 6-point toy-series ceiling effect'ini önlemektir.

### Competence Pre-Test

- 4 model
- 30 soru / model
- toplam 120 çağrı

---

## 10. Pilot Kill / Continue Kriteri

Bu kriter **pilot başlamadan önce sabitlenmeli ve sonuç görüldükten sonra değiştirilmemelidir.**

> **Pilot devam kriteri:** competence pre-test ≥%90 geçen modellerde, L1 Cell #2 Spurious Trend Rate en az bir frontier modelde ≥%25, **veya** code-agent kolunda Provenance Retention Rate ≤%60. İkisi de sağlanmazsa #108b bırakılır.

Bu pre-registration mantığında kullanılmalıdır.

---

## 11. Ana Metrikler

### 11.1 Spurious Trend Rate

Cell #2'de:

Underlying state değişmediği halde modelin gerçek trend iddia etme oranı.

**Düşük olmalı.**

### 11.2 Miss Rate

Cell #3'te:

Gerçek latent değişim olduğu halde modelin bunu kaçırma oranı.

Bu metrik blanket skepticism'i cezalandırır.

**Düşük olmalı.**

### 11.3 Latent Recovery Accuracy

Cell #4'te:

Measurement function değişmiş olsa bile modelin latent state değişimini doğru çıkarması.

### 11.4 Provenance Retention Rate

Özellikle code-agent kolunda:

Modelin analiz boyunca:

- method_id
- calibration_version
- reference_range
- sensor / assay provenance

bilgisini koruma ve kullanma oranı.

---

## 12. Blanket Skepticism Problemi

Cell #2 tek başına ölçülürse model her soruda:

> "Karşılaştırılamaz."

diyerek yüksek skor alabilir.

Bu nedenle Cell #1 ve Cell #3 zorunlu kontrol hücreleridir.

Ana sonuç tek skor olmamalıdır.

Önerilen birlikte raporlama:

- Spurious Trend Rate
- Miss Rate

Model hem sahte trend üretmemeli hem gerçek trendi kaçırmamalıdır.

---

## 13. Ek Kontroller

### Direction Control

Measurement change:

- upward
- downward

olarak randomize edilir.

Amaç modelin yalnızca artışlara alarm üretip üretmediğini görmek.

### Magnitude Sweep

Örnek dönüşüm büyüklükleri:

- 1.05×
- 1.2×
- 2×
- 10×

Measurement-function farkının hangi büyüklükte model tarafından fark edildiği ölçülür.

### Position / Order Randomization

Switch zamanı ve seçenek sıraları randomize edilir.

### Memorization Control

İki kol:

1. Kurgusal analit / sensör + random measurement function
2. Gerçek analit / gerçek sistem örnekleri

Bu sayede frontier modellerin ezberlenmiş klinik veya finansal dönüşüm bilgileri ayrı incelenebilir.

---

## 14. Mitigation

Paper yalnızca failure bulmakla kalmamalı.

Önerilen intervention:

### Provenance-Typed Tool Output

```json
{
  "value": 120,
  "unit": "AU",
  "method_id": "assay_B",
  "calibration_version": "v2",
  "reference_range": [0, 150]
}
```

Plain numeric output ile provenance-typed output karşılaştırılır.

Sorular:

- Spurious Trend Rate düşüyor mu?
- Miss Rate artıyor mu?
- Provenance Retention Rate yükseliyor mu?
- Code-agent daha doğru analysis code üretiyor mu?

---

## 15. Yakın Prior Art ve Konumlandırma

### Counterfactual Invariance

Veitch et al. (2021):

- counterfactual invariance kavramsal zemin
- bizim novelty'miz invariance kavramının kendisi değildir.

### Clinical Counterfactual Sensitivity

Turk (2026), CSS:

- gerçek klinik sinyali değiştirir
- modelin doğru yönde güncellenmesini ölçer
- sensitivity testidir.

Biz:

- latent state'i sabit tutarız
- yalnız measurement function değişir
- modelin **güncellenmemesi** gerekir
- invariance testidir.

### RADAR

Representation / artifact robustness'a yakındır.

Ancak RADAR'da sorun çoğunlukla input normalization / artifact handling'dir.

Bizde observation-generating process değişir.

### Hager et al.

Reference range kullanımı konusunda temel capability confound yaratır.

Bu nedenle competence pre-test zorunludur.

### Sensor Drift / Calibration Literature

Sensor drift vs process drift klasik mühendislik problemidir.

Novelty:

> Bu ayrımı LLM agent'ların yapıp yapamadığının controlled causal evaluation'ı.

### Financial Agents

Restatement / accounting-policy değişimleri mevcut sistemlerde ele alınmaktadır.

Ancak şu ana kadar bulunan prior art'ta:

`latent fixed + reporting function intervention`

şeklinde controlled invariance design bulunmamıştır.

---

## 16. Full Benchmark Hedefi

Pilot geçerse önerilen minimum full-paper tasarımı:

- 2 latent-state condition
- 2 measurement-function condition
- 4 provenance-salience level
- 3 domain
- ≥50 instance / hücre
- ≥6 model
- ≥3 model ailesi
- direct + code-agent harness

Ana yapı:

`2 × 2 × 4 × 3 = 48 hücre`

Her domain için synthetic + real-world validation kolu düşünülebilir.

---

## 17. Paper Framing

Paper'ın ana mesajı:

> **We distinguish sensitivity to true world changes from invariance to instrument-induced observational changes.**

Arkasından:

> **To our knowledge, we provide the first controlled evaluation in LLM agents where the latent state is held fixed by construction while the measurement function is intervened upon.**

Abstract'ta kullanılabilecek matematiksel özet:

> **Δy ≠ 0 ⇏ Δx ≠ 0**

Yani:

> Gözlenen değer değişti diye dünya değişmiş olmak zorunda değildir.

---

## 18. Mevcut Durum

**Proje:** #108b — Measurement-Function Invariance in LLM Agents

**Novelty durumu:** 🟢 Provisional survivor

**Literatür arama durumu:** Yeni fikir araması şimdilik kapatıldı.

**Sonraki aşama:** Pilot benchmark implementation.

**Kill condition:** Önceden sabitlenen pilot kriterleri sağlanmazsa proje bırakılacak; sağlanırsa full benchmark'a geçilecek.
