# 🔬 Project Post-Mortem: BTCUSDT HFT Arbitraj Botu
**Statü:** *Retired (Emekliye Ayrıldı)* | **Tarih:** 26 Mart 2026  
**Disiplinler:** Quantitative Research, Algorithmic Trading, Reinforcement Learning, Market Microstructure  

---

## 1. Yönetici Özeti (Executive Summary)

Bu nicel araştırma (Quantitative Research) ve algoritmik ticaret projesi, Binance Futures üzerinde **L1 (bookTicker) verisi** kullanarak yüksek frekanslı (HFT) piyasa-nötr (market-neutral) bir arbitraj botu geliştirmek amacıyla başlatılmıştır. 

Titizlikle yürütülen 10 farklı mimari iterasyon (V1-V10) ve 1.000.000+ satırlık gerçek borsa verisi üzerindeki ampirik geriye dönük testler (backtest) sonucunda elde edilen en temel bulgu şudur: Salt L1 orderbook verisine dayalı yüksek frekanslı stratejiler; **Taker (Alıcı) komisyonları** ve pasif likidite sağlarken karşılaşılan **Ters Seçicilik (Adverse Selection)** asimetrisi nedeniyle matematiksel olarak sürdürülemez bir yapıya sahiptir. Proje, veri bilimi ve nicel finans prensiplerinden sapmamak adına şeffaf bir analiz ile sonlandırılmıştır.

---

## 2. Model Evrimi: Takviyeli Öğrenme (PPO) Çıkmazı

Projenin ilk aşamasında, mikro-yapı sinyallerine otonom reaksiyon gösterebilmesi için `stable-baselines3` kullanılarak bir **Proximal Policy Optimization (PPO)** Takviyeli Öğrenme (RL) ajanı tasarlanmıştır. Ancak 7 farklı versiyonda (V1-V7) çözülemeyen temel mikroyapı engelleriyle karşılaşılmıştır:

*   **Ceza Kaynaklı Ölüm Sarmalı (Death Spiral):** Gürültülü (noisy) HFT verisinde aşırı işlem yapmayı (overtrading) engellemek adına ödül (reward) fonksiyonuna eklenen cezalar (penalty), ajanın rasyonel karar alma mekanizmasını bozmuştur. Ceza yememek için art arda mantıksız işlemler yapan ajan, portföyü hızla eriterek matematiksel bir sarmala girmiştir.
*   **Maker (Limit Emir) Açmazı ve Düşen Bıçak:** Taker maliyetlerinden kaçmak için sistem Limit Emir simülasyonuna (Maker Pivot) geçirildiğinde, ajanın ciddi bir *"Catching a Falling Knife"* eğilimi sergilediği görülmüştür. Botun girdiği Limit Alış (Bid) emirleri yalnızca büyük satıcıların piyasayı aşağı kırdığı, yani fiyatın botun aleyhine çöktüğü saniyelerde (`Adverse Selection`) gerçekleşmiştir.

---

## 3. Stratejik Pivot: XGBoost ve Bulunan "Alfa"

RL ajanının "Markov Decision Process" doğasındaki stabilite eksikliği tespit edildikten sonra, doğrudan tahmine (Directional Prediction) dayalı Gözetimli Öğrenme (Supervised Learning) problemine geçiş yapılmıştır.

*   **Zaman Serisi Mühendisliği (Feature Engineering):** Veri setine 100-Tick Rolling **OFI (Order Flow Imbalance) Z-Skorları** ve **OBI (Order Book Imbalance)** gibi likidite akış indikatörleri entegre edilmiştir. Bu yolla model, fiyatın ne yöne gideceğinden ziyade "Emir defterindeki basıncın kırılımını" ölçmeye odaklanmıştır.
*   **İstatistiki Başarı:** Geliştirilen **XGBoost (Hist Gradient Boosting)** modeli, Zaman Serisi Sızıntısı (Data Leakage) barındırmayan %80 Eğitim / %20 OOS (Out-of-Sample) test kurgusunda eğitilmiştir. Model, anlık fiyattan **1.0 USDT'lik kopuşları (Threshold)** **%86 ila %90 Duyarlılık (Recall)** oranlarıyla saniyeler öncesinden tespit etmeyi başararak gerçek bir "Alfa" (Predictive Edge) bulduğunu kanıtlamıştır.

---

## 4. Acı Gerçek: Borsa Matematiği ve Komisyon Bariyeri

XGBoost modelinin yakaladığı devasa zeka ve alfa'ya rağmen sistem canlı (Live Testnet) teste çıkarıldığında ampirik bir gerçekle yüzleşilmiştir:

Algoritma bir fiyat kırılımını (örneğin 1.5 USDT'lik bir mum) önceden görüp pozisyona dahi girse, Binance Futures üzerindeki mevcut **%0.04 Taker komisyonu** yuvarlamalarla birlikte (Giriş + Çıkış) asgari **~54 USDT**'lik bir engeli dayatmaktadır. Borsa matematiğindeki bu komisyon spread'i, modelin yakaladığı mikroskobik alfa marjından kümülatif olarak büyüktür.

Projenin asıl DNA'sı olan **"Scalping/Arbitraj"** mantığından saparak; stop-loss riskini genişleten, saatlerce bekleyen tehlikeli bir yönsel (directional/swing) bota dönüşmek, kantitatif araştırma ahlakıyla örtüşmemektedir. Bu sebeple projenin orijinal felsefesine ve matematiksel sınırlarına sadık kalınarak proje olgunlukla emekliye ayrılmıştır.

---

## 5. Çıkarımlar ve Gelecek Vizyonu

Bu araştırma laboratuvarı, bir finansal problemin teknolojik fantezilerin (Sadece AI kullanmak) ötesine geçip derin piyasa gerçekleri ile (Limit Emir Defteri mekanikleri) harmanlanması gerektiğini muazzam bir netlikle ortaya koymuştur. 

1.  **L1 Verisinin Yetersizliği:** Yüksek frekanslı piyasa-yapıcı (Market Maker) bir bot geliştirebilmek için yalnızca `bookTicker` (En iyi Alış/Satış, L1) verisi yetersiz kalmaktadır. Ajanın sırasını bilebilmesi (Queue Position) için Orderbook'un ilk 10-20 kademesi ile Trade-Stream akışını birleştiren L2/L3 derinlik verisi elzemdir.
2.  **Mühendislik Kazanımları:** Asenkron (`asyncio`, `websockets`) canlı fiyat akışı dizaynı, saniyenin altında çalışan `collections.deque` destekli rolling feature vektörizasyonu ve sızdırmaz backtest altyapısı gibi konularda dünya standartlarında bir mimari bilgi havuzu ve mikro-yapı risk vizyonu kazanılmıştır.

**Son Söz:** Başarılı bir nicel süreç, yalnızca kârlı algoritmalar bulan değil; nerede durması gerektiğini bilerek, zarara giden bir yapıyı kanıtlara dayandırarak sonlandırabilen süreçtir.
