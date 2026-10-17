# measurement-invariance Deposu: İnceleme Raporu

Oct 17, 2026 · @Barış Şavk

## Özet

Depo iyi kurulmuş bir pilot benchmark içeriyor, fakat vardığı "projeye devam" kararı güçlü bir kanıta dayanmıyor. [umitaltinozzz/measurement-invariance](https://github.com/umitaltinozzz/measurement-invariance), LLM ajanlarının bir sensör veya kalibrasyon değişimini gerçek bir değişim sanıp sanmadığını ölçüyor. Bunun için 2×2 bir tasarım, deterministik veri üreteci, katı yanıt ayrıştırma ve önceden kaydedilmiş bir devam/bırak kuralı kullanıyor.

Bu incelemede kodun tamamı okundu ve 583 test temiz bir klonda çalıştırıldı. Run 2 sonuçları da ham JSONL dosyalarından bağımsız olarak yeniden hesaplandı. Yazarın tabloları birebir yeniden üretiliyor ve 581 test geçiyor. Buna karşın kararı taşıyan üç dayanak zayıf:

- **Doğrudan tetikleyici özgül değil.** Tetikleyiciyi sağlayan tek model gpt-oss:20b. Bu model saf gürültü kontrolünde de neredeyse aynı oranda sahte trend bildiriyor (%68'e karşı %76, Fisher p≈0,75).
- **Kod tetikleyicisi neredeyse kendiliğinden ateşleniyor.** Sağlayıcı (provenance) bilgisinin sabit olduğu hücre 3 hesaba katıldığı için oran %50 civarında tavan yapıyor. Yalnız hücre 2 alınıp kaçış hatası giderilince, deponun kendi ölçütüyle gpt-oss'un oranı %79 çıkıyor ve eşiği geçmiyor.
- **Koşu sırası ön kayıttan sapıyor.** Run 1 kuralı karşılamadı (`project_continue = False`). 27 dakika sonra, önceden sabitlenmemiş yeni bir model listesiyle run 2 kuruldu. Plandaki "frontier model" şartı da uygulamada düşürüldü.

Raporlama ayrıca iki örüntüyü gizliyor. Qwen3 denemelerin %91'inde "uncertain" diyor. Hücre 4'te ham verinin yanılttığı 8 vakada Gemma, gpt-oss ve Mistral 5–7 kez ters yön bildiriyor. Temiz klonda referans analiz betiği SHA uyuşmazlığıyla duruyor.

**Anahtar kelimeler:** ölçüm fonksiyonu değişmezliği, LLM değerlendirmesi, benchmark geçerliliği, ön kayıt, tekrarlanabilirlik, kod ajanları

## 1. Giriş

Depo, psikometrideki "ölçme değişmezliği" (CFA tabanlı grup karşılaştırması) ile ilgili değil. Konusu, bir LLM'in **ölçüm fonksiyonundaki** değişimi dünyadaki değişimden ayırt edip edemediği. Temel model şudur:

```latex
y_t = f_t(x_t), \qquad \Delta y \neq 0 \;\not\Rightarrow\; \Delta x \neq 0
```

Burada x gizli (latent) sıcaklık, f sensörün ölçüm fonksiyonu, y okunan değerdir. Bir sensör değiştiğinde okunan değer kayabilir, ama süreç değişmemiştir. Deponun sorusu, LLM'in bu durumda "sıcaklık arttı" deyip demediğidir.

Tasarım 2×2'dir ve dört hücreden oluşur:

| Hücre | Gizli durum x | Ölçüm fonksiyonu f | Doğru cevap | Rolü |
| --- | --- | --- | --- | --- |
| 1 | aynı | aynı | değişim yok | gürültü kontrolü |
| 2 | aynı | değişti | değişim yok | ana koşul |
| 3 | değişti | aynı | değişimi bul | aşırı çekimserlik kontrolü |
| 4 | değişti | değişti | f'yi tersine çevirip x'i bul | gizli durumu geri kazanma |

Pilot, kurgusal bir endüstriyel sıcaklık sensörüyle sınırlıdır. Hücre başına 25 vaka, vaka başına 30 zaman adımı ve yaklaşık %5 gürültü vardır. Sağlayıcı bilgisi iki düzeyde verilir: L1'de ayrı sütunlarda, L2'de bir cihaz kimliğinin içine gömülü. İki istem türü kullanılır: ana kol "natural" ve ipucu veren "guided". Bunlara ek olarak bir kod-ajanı kolu ve bir yeterlilik ön testi bulunur.

Bu rapor üç şeyi değerlendiriyor: yazılımın doğruluğu ve kalitesi, deneysel tasarımın geçerliliği ve yayımlanan sonuçların veriyle ne kadar desteklendiği. İnceleme, 17 Ekim 2026 tarihli `7d5c7fc` commit'i üzerinde yapıldı. Depo 21 commit ve tek yazardan oluşuyor; tüm geçmiş 16–17 Ekim 2026'ya sığıyor.

## 2. Yöntem

İnceleme dört adımda yapıldı. Hiçbir modele yeniden istek gönderilmedi; tüm sayılar depodaki ham çıktılardan geliyor.

1. **Okuma.** README, `proje.md`, `PILOT_RESULTS.md`, run 2 teşhis ve A100 raporları okundu. 36 kaynak dosyasının (8.464 satır) tasarımı taşıyan kısımları satır satır incelendi: şema, üreteç, istemler, ayrıştırıcı, notlayıcı, metrikler, devam kuralı, koşucu, Ollama sağlayıcısı ve kod-ajanı notlayıcısı.
2. **Testler.** 17 test dosyası (5.430 satır) temiz bir klonda çalıştırıldı. Ortam: Windows 11, Python 3.12.6, pydantic 2.14.1, PyYAML 6.0.3, pytest 9.1.1.
3. **Yeniden analiz.** Run 2 satırları, A100'de yeniden koşturulan 59 satırla aynı vaka kimliklerinden değiştirilerek birleştirildi. Notlama, deponun kendi `grade_response` ve `grade_code_agent_response` fonksiyonlarıyla yapıldı.
4. **Ek analizler (bu incelemeye ait).** Yazarın raporlamadığı şu hesaplar eklendi:
   - Wilson %95 güven aralıkları,
   - hücre 1 ile hücre 2 arasında Fisher kesin testi,
   - eşli veri setinde doğrudan kol ile kod kolu arasında kesin McNemar testi,
   - hücre 4 vakalarının ham yönün doğru/yanlış olduğu alt kümelere ayrılması,
   - kod-ajanı AST sınıflandırıcısına elle yazılmış kısa kod örnekleriyle yapılan yoklamalar,
   - git geçmişinden koşu ve commit zaman çizelgesi.

Sınırlar: GPU olmadığı için model çağrıları tekrarlanmadı. Üretilen kod çalıştırılmadı (depo da çalıştırmıyor). Colab defterleri yalnızca okundu. Aşağıda "bu inceleme" diye işaretlenmeyen her sayı yazarın tablolarından alınmıştır ve ham veriden yeniden üretilmiştir.

## 3. Deponun Yapısı ve Mimarisi

Depo, tek bir YAML dosyasından başlayıp devam kararına kadar giden doğrusal bir boru hattıdır. Her aşama bir öncekinin çıktısını SHA-256 ile doğrular.

&#91;embedded content: Boru hattı · 8 adım, commit 7d5c7fc\]

Veri soldan sağa üretilip notlanıyor; karar, sağ alttaki ham çıktılardan sol alttaki `continuation.py` dosyasına kadar dört adımda oluşuyor.

Bağımlılıklar az: `pydantic`, `PyYAML` ve `pytest`. Figür betiği `matplotlib` kullanıyor ama bu paket `requirements.txt` içinde yok. Sürümler alt ve üst sınırla veriliyor, kilit dosyası yok.

| Klasör | İçerik | Boyut |
| --- | --- | --- |
| `src/` | üreteçler, istemler, notlayıcılar, metrikler, koşucu, sağlayıcı katmanı | 36 dosya, 8.464 satır |
| `tests/` | birim ve uçtan uca testler, sentetik yanıtlarla metrik doğrulaması | 17 dosya, 583 test |
| `configs/` | pilot, Colab run 1, run 2 ve yeniden koşu yapılandırmaları | 5 dosya |
| `data/generated/` | ana veri seti (100 vaka), eşli veri seti (40), yeterlilik seti (30) | 3 JSONL |
| `data/results/colab_t4/` | run 1, run 2 ve A100 ham çıktıları, manifestler, analizler | 1.240 satır (run 2) |
| `colab/` | Colab defterleri ve onları üreten betikler | 4 dosya |

Sağlayıcı katmanı modelden bağımsız tasarlanmış. Yeniden deneme, kayıt ve devam ettirme mantığı yalnızca `src/runner.py` içinde. Ollama dışında OpenRouter, Vercel ve çevrim dışı test için Replay/Scripted sağlayıcıları var.

## 4. Bulgular

Yazılım sağlam, fakat devam kararını taşıyan üç dayanak da zayıf. Aşağıdaki tablo 13 bulguyu önem sırasına göre veriyor; alt bölümler kanıtı açıyor. "h1–h4" hücre 1–4 demektir.

| # | Bulgu | Önem | Kanıt | Konum |
| --- | --- | --- | --- | --- |
| 1 | Doğrudan tetikleyici gürültü kontrolünden ayrışmıyor | Yüksek | gpt-oss: h1 %68, h2 %76, p≈0,75 | [`continuation.py`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/src/continuation.py) |
| 2 | Kod tetikleyicisi h3 ile havuzlandığı için neredeyse hep ateşleniyor | Yüksek | h3'te PRR dört modelde 0/20; gpt-oss h2 düzeltilmiş %79 | [`code_agent_metrics.py`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/src/code_agent_metrics.py) |
| 3 | Koşu sırası ve model seçimi ön kayıttan sapıyor | Yüksek | run 1 False, 27 dk sonra yeni modellerle run 2 | [`proje.md`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/proje.md) §10 |
| 4 | Çekimserlik başlık metriklerinde görünmüyor | Yüksek | Qwen3 %91 "uncertain"; h3'te 16/25 | [`metrics.py`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/src/metrics.py) |
| 5 | Hücre 4'ün tanısal alt kümesinde modeller başarısız | Orta | ham yönün yanılttığı 8 vakanın 5–7'sinde ters yön | [`PILOT_RESULTS.md`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/PILOT_RESULTS.md) |
| 6 | Natural istem uç nokta karşılaştırmasına davet ediyor | Orta | gpt-oss h1 sahte yönü 17/17 son−ilk işaretiyle aynı | [`prompts.py`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/src/prompts.py) |
| 7 | Doğrudan–kod karşılaştırması eşli kolu kullanmıyor | Orta | eşli kolda Gemma'da kod daha iyi, p≈0,008 | [`make_readme_figures.py`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/scripts/make_readme_figures.py) |
| 8 | AST sınıflandırıcısı "kullanım"ı fazla sayıyor | Orta | ham ortalamaları karşılaştıran `groupby` düzey 2 alıyor | [`code_agent_grader.py`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/src/code_agent_grader.py) |
| 9 | Akıl yürütme koşulları modeller arasında eşit değil | Orta | `think: false`; gpt-oss 1.024 tokende boş yanıt | [`ollama.py`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/src/models/ollama.py) |
| 10 | Örneklem küçük, belirsizlik raporlanmıyor | Orta | tetik dilimi n=15: %67 \[%42, %85\] | [`PILOT_RESULTS.md`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/PILOT_RESULTS.md) |
| 11 | Temiz klonda referans analiz çalışmıyor | Orta | 2 test başarısız; CRLF/LF SHA farkı | [`test_case_selection.py`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/tests/test_case_selection.py) |
| 12 | Başarısız sağlayıcı satırları yeniden sorulmuyor | Düşük | 28 Gemma HTTP 404 satırı kalıcı oldu | [`runner.py`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/src/runner.py) |
| 13 | Dokümantasyon kodun gerisinde kalmış | Düşük | README'de `per_skill_min` 0,80; config'te 0,85 | [`README.md`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/README.md) |

### 4.1 Güçlü yönler

Mühendislik, tek kişilik ve iki günlük bir proje için alışılmadık derecede titiz.

- **Etiketler yapısal olarak korunuyor.** `BenchmarkCase` doğrulayıcıları, etiketi gizli durumla çelişen bir vakanın oluşturulmasını reddediyor.
- **Ölçüm ızgarası dengeli.** Yön, kazanç ve ofset tam faktoriyel atanıyor. Her noktanın eşiği aştığı yapılandırma yüklenirken ispatlanıyor.
- **Ayrıştırma katı ve dürüst.** Geçersiz yanıt ayrı bir sınıf. Her oran pay ve paydasını taşıyor; boş payda `0` değil `None` dönüyor.
- **Kayıtlar izlenebilir.** Her koşu veri seti ve yapılandırma SHA'larını içeren bir manifest yazıyor.
- **Sorunlar açıkça belgelenmiş.** Teşhis raporu, değiştirilen satır denetimi ve Colab Gemini asistanının meta veriyi değiştirmesi olayı saklanmadan yazılmış.
- **Tablolar ham veriden üretiliyor (bu inceleme).** Run 2 tablolarındaki her oran ham JSONL'dan birebir yeniden hesaplandı.

### 4.2 Devam kararı neye dayanıyor?

Kodlanmış kural `True` döndürüyor, ama tetikleyicilerin hiçbiri ölçüm değişimine özgü bir etki göstermiyor.

**Doğrudan tetikleyici (bulgu 1).** Kural yalnızca h2'deki oranı eşikle karşılaştırıyor; h1 kontrolüyle fark almamıyor. Tetikleyen tek model gpt-oss:20b. Bu modelin natural istemde h1 oranı 17/25, h2 oranı 19/25 (Fisher p≈0,75; bu inceleme). Kuralın tam dilimi olan L1'de tablo tersine dönüyor: h2 10/15, h1 11/15. Yani gürültü kontrolü ana koşuldan daha yüksek.

&#91;embedded content: Kaynak: run 2 ham JSONL (A100 değişiklikleriyle), natural istem · aralıklar ve testler bu incelemeye ait\]

Gemma ve Mistral'de iki aralık birbirinden tamamen ayrı; gpt-oss'ta büyük ölçüde örtüşüyor.

Yazar bu zayıflığı `PILOT_RESULTS.md` içinde kabul ediyor. Buradaki tespit, sorunun bir yorum notu değil kuralın tanımı olduğu. Özgül etki gösteren Gemma 4 (%16 → %100) ve Mistral Small 3.2 (%8 → %84) yeterlilik testini geçemediği için kural dışı kalıyor.

**Kod tetikleyicisi (bulgu 2).** Ayrıntısı 4.4'te. Özetle, eşik h3 ile havuzlama yüzünden akıllıca davranan bir modelde bile aşılıyor.

**Ön kayıttan sapma (bulgu 3).** `proje.md` §10, kuralı "en az bir frontier modelde" koşuluyla tanımlıyor ve "ikisi de sağlanmazsa bırakılır" diyor. Uygulama bu şartı düşürüp 12–24B boyutunda nicelleştirilmiş açık modellere uyguluyor. Git geçmişi şu sırayı gösteriyor (saatler UTC+3, 16–17 Ekim 2026):

| Saat | Commit | Olay |
| --- | --- | --- |
| 17 Ekim 05:55 | `67d7a18` | run 2 ve A100 yeniden koşusu referans sonuç ilan edildi |
| 17 Ekim 05:00 | `2c11de0` | sonuçlar görüldükten sonra 59 satırlık A100 yeniden koşusu eklendi (31'i daha yüksek token sınırıyla) |
| 16 Ekim 18:55 | `5c04eab` | yeni model listesiyle run 2 kuruldu; ayrıştırıcı da değişti |
| 16 Ekim 18:28 | `29b2150` | run 1 yayımlandı: hiçbir model yeterli değil, `project_continue = False` |
| 16 Ekim 14:31 | `a91ba46` | eşikler ilk commit'te sabitlendi (run 1'den yaklaşık 1 saat önce) |

Yeniden koşu, gpt-oss kod kolunun geçersiz oranını %12,5'ten %2,5'e indirdi ve kod tetikleyicisini engelli durumdan ateşlenir duruma geçirdi. İki okuma mümkün. Yazarın okumasında run 1 bilgi vermez, çünkü hiçbir model yeterliliği geçmedi. Kuralın sözüne göre ise run 1 bir "bırak" sonucudur ve run 2 ikinci bir çekiliştir. Her iki okumada da run 2 keşif amaçlı (exploratory) diye etiketlenmelidir.

### 4.3 Metrikler ve raporlama

Başlık tabloları yalnızca Spurious Trend Rate ve Miss Rate'i veriyor; bu ikisi çekimserliği ve ters yönü görmüyor.

**Çekimserlik görünmüyor (bulgu 4).** Miss Rate yalnızca "stable" cevapları sayıyor ([`metrics.py`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/src/metrics.py) satır 230). Hücre 3'teki "uncertain" aşırı çekimserlik olarak hesaplanıyor ama üretilen `analysis.md` ve `analysis.json` dahil hiçbir raporda yer almıyor. README, bu ölçümün "her şeye karar verilemez diyen" politikayı cezalandırdığını söylüyor; mevcut raporlama bunu yapmıyor. Natural istemde (bu inceleme):

| Model | "uncertain" (100 deneme) | h1 toptan şüphecilik | h3 aşırı çekimserlik | h4-L1 gizli durum geri kazanımı |
| --- | --- | --- | --- | --- |
| qwen3:14b | 91 | 25/25 | 16/25 | 0/15 |
| mistral-small3.2:24b | 17 | 10/25 | 0/25 | 12/15 |
| gpt-oss:20b | 12 | 1/25 | 0/25 | 10/15 |
| gemma4:12b | 0 | 0/25 | 0/25 | 11/15 |

README, Qwen3 için "not fooled" ve Türkçe özette "doğrudan sorulunca kanmıyor" diyor. Veri, Qwen3'ün neredeyse hiç karar vermediğini gösteriyor. Benzer bir durum gpt-oss'un guided sonucunda var: h2'deki %8'lik düşük oran 19/25 "uncertain" ile geliyor. H4-L1 geri kazanımı da natural istemdeki 10/15'ten guided istemde 3/15'e düşüyor.

**Hücre 4'ün tanısal kısmı (bulgu 5).** Hücre 4'teki 25 vakanın 17'sinde ham seri zaten doğru yönü gösteriyor. Sağlayıcı bilgisini yok sayan bir okuyucu bunları da doğru bilir. Tanısal olan, ham yönün yanılttığı 8 vaka (4'ü L1, 4'ü L2). Natural istemde bu 8 vakada Gemma 7'sinde, gpt-oss 6'sında, Mistral 5'inde ters yön bildiriyor (bu inceleme). README'deki "gerçek değişimi neredeyse hiç kaçırmıyorlar" cümlesi bu yüzden eksik. Modeller değişimi kaçırmıyor ama ölçüm değişimi yönü tersine çevirdiğinde yanlış yönü söylüyor.

**Natural istemin işlemselleştirmesi (bulgu 6).** Natural istem trendi "pencerenin başından sonuna" sıcaklığın nasıl davrandığı diye tanımlıyor. Anlamlı bir değişimin ölçüsünü vermiyor. Gürültü çarpımsal ve %5. Bu yüzden h1'deki 25 vakanın 9'unda ilk ve son okuma %5'ten fazla farklı. Doğrusal eğilim ise yalnızca 1 vakada anlamlı (|t| > 2,05). gpt-oss'un h1'deki 17 sahte trendinin 17'si de son−ilk farkının işaretiyle aynı yönde (bu inceleme). Yani h1, "gürültüyü sinyal sanma" kadar istemin harfiyen okunmasını da ölçüyor.

### 4.4 Kod-ajanı kolu

Kod kolunun üç bileşeni de, yani tetikleyici metriği, sınıflandırıcı ve karşılaştırma, sonuçları yorumlanması güç hale getiriyor.

**Havuzlama (bulgu 2).** Provenance Retention Rate (PRR), kodun sağlayıcı sütunlarına en az bir kez dokunma oranı. Oran h2 ve h3 birlikte hesaplanıyor. H3'te sağlayıcı sütunları sabit olduğu için onları yok saymak doğru analizdir. Dört modelin dördü de h3'te bu sütunlara hiç dokunmuyor. Qwen3 dışındaki üç model h3'ü 19–20/20 doğru cevaplıyor. Sonuçta sütunları yalnızca değiştiğinde kullanan ideal bir analist bile yaklaşık %50 PRR alır ve %60 eşiğini otomatik olarak sağlar.

gpt-oss'ta durum kaçış hatasıyla ağırlaşıyor. 21 yanıtın kodunda gerçek satır sonu yerine harf harf `\n` var; bunlar sözdizimi hatası sayılıp düzey 0 alıyor. İstem kodu "kaçışlı satır sonlarıyla tek bir JSON dizesi" olarak istediği için bu hata istemden kaynaklanıyor olabilir. Harf harf `\n` satır sonuna çevrildiğinde 21 yanıtın 20'si ayrıştırılıyor. Bu durumda gpt-oss'un h2 PRR'si 15/19 = %79 \[%57, %91\] oluyor ve eşiği geçmiyor (bu inceleme). Bu, tetikleyicinin kendi ölçütüyle yapılmış bir kontrol; bulgu 8'deki sınıflandırıcı sorunu bu oranı da etkiliyor. Havuzlanmış oranın %38'de kalmasının tek nedeni h3.

**Sınıflandırıcı (bulgu 8).** AST sınıflandırıcısı, bir sütun adının üstünde herhangi bir karşılaştırma, işlem, `if` bloğu veya sabit olmayan indeks görürse onu "kullanıldı" (düzey 2) sayıyor. Bu incelemede yazılan üç kısa örnek de düzey 2 aldı:

- `df[['reading', 'sensor_version']]` ile sütun seçip ham uç noktaları karşılaştıran kod,
- ilgisiz bir `if len(df) > 0:` bloğunun içinde sütunu yalnızca bir değişkene atayan kod,
- `groupby('sensor_version')` ile sensör başına ham ortalama alıp bunları karşılaştıran kod. Bu tam da projenin aramaya çalıştığı hatadır, ama hiçbir anti-örüntü işaretlenmiyor.

Veri de bunu doğruluyor: düzey 2 oranı yüksek olan modeller h2'de yine de büyük ölçüde sahte trend bildiriyor (aşağıdaki tablo). Yazar bu etiketleri "sezgisel" diye işaretliyor ve insan incelemesi öneriyor. Ancak kod tetikleyicisi bu etiketleri insan incelemesi olmadan kullanıyor.

**Doğrudan–kod karşılaştırması (bulgu 7).** README figüründe doğrudan çubuk ana veri setinden (h2, n=25, L1+L2), kod çubuğu eşli veri setinden geliyor; README bunu parantez içinde belirtiyor. Oysa depo tam bu karşılaştırma için aynı 40 vakayı doğrudan kolda da çalıştırmış. Eşli veride h2'nin 20 vakası (bu inceleme, kesin McNemar):

| Model | h2 PRR | h3 PRR | Doğrudan kol sahte trend | Kod kolu sahte trend | McNemar p |
| --- | --- | --- | --- | --- | --- |
| gemma4:12b | 18/20 | 0/20 | 20/20 | 12/20 | 0,008 |
| gpt-oss:20b | 8/19 (düzeltilmiş 15/19) | 0/20 | 13/20 | 6/20 | 0,065 |
| mistral-small3.2:24b | 20/20 | 0/20 | 17/20 | 15/20 | 0,69 |
| qwen3:14b | 19/20 | 0/20 | 0/20 | 20/20 | <0,001 |

Projenin "kod yazınca bilgi kaybolur" hipotezi yalnızca Qwen3'te tutuyor. Orada da doğrudan kolun 20 cevabının 20'si "uncertain". Yani değişim sağlayıcı farkındalığından körlüğe değil, çekimserlikten karar vermeye geçiş. Gemma ve gpt-oss'ta kod kolu daha iyi.

Ayrıca kod istemi, docstring'inin aksine natural çerçeveyi kullanmıyor. "Underlying physical quantity" diye soruyor, "cannot be determined" seçeneği sunuyor ve farklı bir sistem rolü veriyor. Bu ifade guided isteme daha yakın. Kod çalıştırılmadığı için kod kolunun "cevabı" programın çıktısı değil, modelin beyan ettiği sonuç. Bu nedenle karşılaştırmada kip (kod/düz yazı) ile istem ifadesi birbirine karışıyor.

### 4.5 Deney koşulları ve istatistik

Modeller aynı koşullarda çalışmadı ve örneklem, raporlanan kesinliği taşıyacak büyüklükte değil.

**Akıl yürütme (bulgu 9).** Ollama sağlayıcısı tüm modellere `think: false` gönderiyor. Qwen3'te düşünme kapanıyor. gpt-oss ise akıl yürütmeye devam ediyor gibi görünüyor. Bilgimize göre Ollama'da gpt-oss için yalnızca düşünme düzeyi seçilebiliyor. Nitekim gpt-oss'un 26 doğrudan ve 5 kod yanıtı tam 1.024 / 4.000 tokende boş veya yarım dönüyor (`finish_reason=length`). Yalnızca `message.content` kaydediliyor; akıl yürütme metni atılıyor, bu yüzden incelenemiyor. Bu 31 satır sonradan 4.096 / 16.000 token sınırıyla ve farklı bir GPU'da yeniden koşturuldu. Referans analiz böylece iki farklı koşulu karıştırıyor.

**Örneklem (bulgu 10).** Hücre başına 25, tetik diliminde 15 vaka var. Her vaka sıcaklık 0 ile bir kez soruluyor ve raporlarda güven aralığı yok. Wilson %95 aralıkları geniş: gpt-oss'un tetik dilimi %67 \[%42, %85\] (bu inceleme). Yeterlilik kapısı da birkaç soruya bağlı. Qwen3 30 sorunun 26'sını bildi; geçmek için iki afin sorusunu daha bilmesi yeterdi. Bulgular dört nicelleştirilmiş açık modele ve tek bir sentetik alana dayanıyor; yazar da bunu belirtiyor.

### 4.6 Tekrarlanabilirlik ve bakım

Ham veri depoda ve tablolar ondan üretilebiliyor, fakat deponun kendi analiz betiği temiz bir klonda çalışmıyor.

**SHA uyuşmazlığı (bulgu 11).** Temiz klonda 581 test geçiyor, 2 test kalıyor: `test_run2_config_sha_matches_the_environment_record` ve `test_run2_plus_reruns_analysis_replaces_rows_and_writes_a_separate_report`. Run 2 manifestleri yapılandırma SHA'sını `cb39ccdd…` olarak kaydediyor. Bu değer, yazarın Windows çalışma kopyasındaki CRLF satır sonlu dosyaya ait. Depoya giren dosya LF ve `02bd25d8…` üretiyor. Bu yüzden `analyze_with_reruns` her temiz klonda "config SHA mismatch" hatasıyla duruyor. Ayrıca run 2 manifestlerinde `git_commit: null`; sonuçları üreten kod sürümü sabitlenmemiş. Ayrıştırıcı da run 1 ile run 2 arasında değişti (`5c04eab`).

**Devam ettirme (bulgu 12).** Koşucu, sağlayıcı ve taşıma hatalarını satır olarak yazıyor ve devam ettirmede bu satırları atlıyor. Böylece geçici hatalar kalıcılaşıyor: 28 Gemma HTTP 404 satırı ve 1 Mistral zaman aşımı. Analiz bunları boş metin olarak notlayıp geçersiz yanıt sayıyor. Düzeltmek için elle bir A100 koşusu gerekti.

**Dokümantasyon (bulgu 13).** README'de birkaç eski ifade kalmış:

- `per_skill_min` 0,80 ve "bağlayıcı değil" deniyor; yapılandırmada değer 0,85 ve bağlayıcı.
- Kapsam bölümü "henüz LLM çağrısı ve kod kolu yok" diyor; düzen bölümü `data/results` için "şimdilik boş" diyor.
- "Karar fonksiyonu yazılmadı" deniyor, oysa `src/continuation.py` kararı uyguluyor. Bu dosyanın docstring'i "en az bir frontier model" diyor, kod ise her modeli değerlendiriyor.
- `.gitignore` run 2 klasörünü dışladığı halde run 2 dosyaları depoda. Bazı dosyalar BOM ile başlıyor. `matplotlib` gereksinimlerde yok.

## 5. Tartışma

Depodaki sorunların çoğu kod hatası değil, ölçüm tasarımı ve karar kuralı sorunu. Bu, projenin kendi tezine ironik biçimde uyuyor. Proje bir LLM'in ölçüm fonksiyonunu gerçek dünyayla karıştırıp karıştırmadığını soruyor. Benchmark da kendi metriklerinde benzer bir karışıklığa düşüyor: PRR "sütuna dokunmayı" ölçüyor, "doğru düzeltmeyi" değil. Miss Rate de "karar vermemeyi" başarı gibi gösteriyor.

**Güçlü yönler gerçek.** Yapısal etiket doğrulaması, dengeli faktoriyel ızgara, katı ayrıştırma ve iki paydalı oranlar, benchmark yazımında iyi uygulamadır. LLM yargıcı kullanılmaması da ölçümü denetlenebilir kılıyor. Sorunların saklanmadan belgelenmesi, örneğin Colab asistanının meta veriyi değiştirmesi, güven verici. Bu incelemedeki bulguların neredeyse tamamı depodaki ham veri sayesinde yapılabildi.

**Ön kayıt uygulaması eksik.** İyi ön kayıt pratikleri, örneğin OSF kâlıpları, eşiklerle birlikte model listesini, koşu sayısını ve yeniden koşu politikasını da veriden önce sabitler. Depoda yalnız eşikler sabitlenmiş. Model listesi, ikinci koşu kararı ve hangi satırların hangi sınırla yeniden koşturulacağı sonuçlar görüldükten sonra belirlenmiş. Bu, kural kodda dokunulmadan kalsa bile bir "çatallanan yollar" sorunu doğuruyor. Bir de şu var: eşikler ilk commit'te veriden önce girilmiş olsa da yalnızca yazarın kendi git geçmişiyle belgeleniyor.

**Kural bir fark kuralı olmalıydı.** Tasarım, h1 ve h3'ü zorunlu kontrol olarak tanımlıyor. Karar kuralı ise kontrolleri kullanmıyor. Bulgu 1 ve 2'nin kök nedeni bu. Kural h2 ile h1 arasındaki farkı ve PRR'yi yalnızca h2 üzerinden ölçseydi, gpt-oss için doğrudan tetikleyici ateşlenmezdi; kod tetikleyicisi de kaçış hatası düzeltildiğinde ateşlenmezdi.

**Bulguların ilginç kısmı başka yerde.** Gemma 4 ve Mistral Small 3.2'nin h2'deki güçlü ve özgül etkisi her iki istemde de sürüyor (dört karşılaştırmada da p < 10⁻⁵). Hücre 4'ün tanısal alt kümesindeki ters yön hataları da projenin tezini destekleyen en net veri. Bu iki gözlem yeterlilik kapısı yüzünden karar dışı kalıyor. Oysa bu modeller referans aralığı testini %100 geçti ve yalnızca afin tersine çevirmede zayıf kaldı. Kapının h2 için gerekli olup olmadığı ayrıca tartışılmalı. H2'de doğru cevap için tersine çevirme gerekmiyor; değişimi fark edip "stable" veya "uncertain" demek yetiyor.

## 6. Öneriler

Öneriler etki sırasına göre dizildi. İlk üçü, tam benchmark'a geçmeden önce yapılmalı.

1. **Karar kuralını kontrollere bağla.** Doğrudan tetikleyiciyi aynı istem ve sağlayıcı düzeyindeki h2−h1 farkına bağla; farkın güven aralığının alt sınırı sıfırın üstünde olmalı. PRR'yi yalnızca sağlayıcı bilgisinin değiştiği h2'de hesapla.
2. **Keşif ve doğrulama koşularını ayır.** Run 1'i kuralın verdiği sonuçla, run 2'yi keşif amaçlı olarak raporla. Sonraki koşudan önce model listesini, koşu sayısını ve yeniden koşu politikasını OSF gibi dış bir kayıtta sabitle. "Frontier" şartını ya uygula ya da plandan açıkça çıkar.
3. **Başlık metriklerini tamamla.** Her tabloda aşırı çekimserlik, toptan şüphecilik, ters yön oranı ve gizli durum geri kazanımı da olsun. Hücre 4'ü ham yönün doğru ve yanlış olduğu alt kümelere ayırarak raporla. Tam benchmark'ta bu iki alt kümeyi tasarımda dengele.
4. **Kod kolunu yeniden tasarla.** Kod istemini natural istemle kelimesi kelimesine hizala. Kodu bir konteynerde çalıştırıp cevabı programın çıktısından al. Kodu JSON dizesi içinde istemek yerine ayrı bir alan veya kod bloğu kullan. Sınıflandırıcıyı "dokunma" yerine "düzeltme" (ölçeğe göre normalleştirme, tersine çevirme) üzerinden tanımla. Elle etiketlenmiş bir örneklemle doğrula.
5. **Doğrudan–kod karşılaştırmasını eşli veriden yap.** Figürü eşli doğrudan kolla yeniden çiz ve McNemar testini raporla.
6. **Natural istemi netleştir.** "Baştan sona" yerine "normal dalgalanmanın ötesinde kalıcı bir seviye değişimi" gibi bir tanım kullan. Ya da gürültü düzeyini eşleştirilmiş bir taban çizgisiyle raporla.
7. **Akıl yürütmeyi eşitle ve kaydet.** gpt-oss için düşünme düzeyini açıkça seç, `message.thinking` alanını sakla. Token sınırlarını koşudan önce tüm modeller için yeterli belirle. Yeniden koşu gerekirse yalnız başarısız satırları değil, tüm dilimi koştur.
8. **İstatistiği güçlendir.** Her orana Wilson aralığı ekle. Planlanan hücre başına 50 ve üzeri vakaya çık. Vaka başına birden fazla örnek veya farklı tohumlarla tekrar düşün. Model ve vaka etkileri için karışık etkili bir model kullan.
9. **Tekrarlanabilirliği onar.** `configs/*.yaml` için `.gitattributes` kuralı ekle veya manifestleri depodaki baytlarla yeniden eşle. Manifestlere git commit'ini yaz. Bağımlılıkları kilit dosyasıyla sabitle ve `matplotlib`'i ekle. Temiz klonda testleri çalıştıran bir GitHub Actions işi kur.
10. **Koşucuyu düzelt.** Devam ettirmede sağlayıcı hatası olan satırları yeniden sor. Bu satırları "geçersiz yanıt" yerine "erişilemedi" olarak ayır.
11. **Dokümantasyonu güncelle.** README'deki eski ifadeleri düzelt ve Qwen3 için "kanmıyor" ifadesini çekimserlik oranıyla birlikte yeniden yaz.

## 7. Sonuç

Depo, iyi bir benchmark altyapısı ama pilot henüz projeye devam etmeyi destekleyen güçlü bir kanıt sunmuyor. Yazılım doğru çalışıyor ve yayımlanan tablolar ham veriden yeniden üretilebiliyor. Kodlanmış kapı da gerçekten `True` döndürüyor.

Ancak bu kararın üç dayanağı da zayıf. Doğrudan tetikleyici ölçüm değişimine özgü değil. Kod tetikleyicisi havuzlama yüzünden neredeyse kendiliğinden ateşleniyor. Koşu sırası da ön kayıttan sapıyor. Projenin tezini destekleyen en net veriler ise karar dışı kalan iki modelden ve hücre 4'ün raporlanmayan tanısal alt kümesinden geliyor.

Bu yüzden önerilen yol şu: run 2'yi keşif amaçlı say, karar kuralını 6. bölümdeki ilk üç maddeye göre yeniden yaz ve yeni kuralı bir sonraki koşudan önce dış bir kayıtta sabitle. Bu değişikliklerden sonra mevcut altyapı tam benchmark için sağlam bir temel olur.

## Kaynaklar

- [Depo ana sayfası](https://github.com/umitaltinozzz/measurement-invariance), incelenen commit [`7d5c7fc`](https://github.com/umitaltinozzz/measurement-invariance/tree/7d5c7fce63a48c076931f8b5052a91cfc6bf26ec)
- [`README.md`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/README.md), [`proje.md`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/proje.md), [`PILOT_RESULTS.md`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/PILOT_RESULTS.md)
- [`DIAGNOSIS.md`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/data/results/colab_t4/run2/DIAGNOSIS.md), [`RERUN_A100.md`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/data/results/colab_t4/run2/RERUN_A100.md)
- [`configs/pilot.yaml`](https://github.com/umitaltinozzz/measurement-invariance/blob/7d5c7fc/configs/pilot.yaml) (önceden kaydedilmiş eşikler)
- [Run 2 ham sonuçları](https://github.com/umitaltinozzz/measurement-invariance/tree/7d5c7fc/data/results/colab_t4/run2) ve [A100 yeniden koşu ham çıktıları](https://github.com/umitaltinozzz/measurement-invariance/tree/7d5c7fc/data/results/colab_t4/run2_rerun_a100_raw)
