# Methodology

## 1. Spoofing detection

**Pattern:** A trader places a large order to create a false impression of buy/sell pressure, intending to influence other participants' behavior, then cancels before it executes.

**Implementation:** For each symbol, compute the mean and standard deviation of order quantity across all orders. Flag any order that is both (a) cancelled without execution, and (b) has a quantity z-score above a configurable threshold (default 2.5).

**Why z-score by symbol:** Order sizes vary enormously by instrument (liquidity, typical lot size), so a global threshold would either miss spoofing in illiquid symbols or over-flag in highly liquid ones. Per-symbol normalization keeps the threshold meaningful across different instruments.

**Known limitation:** This is a single-order-level detector. Real spoofing surveillance typically looks at *layering* — multiple orders across price levels — and at the order book impact, not just individual order size. A production system would also weight by how close the order sat to the best bid/offer, since a large order far from the touch is less market-moving than one near it.

## 2. Wash trading detection

**Pattern:** Two related (or the same) parties trade with each other, generating volume without real economic risk transfer, often to inflate apparent liquidity or manipulate a reference price.

**Implementation:** For each symbol, for every SELL order, search for BUY orders from a *different* account within a configurable time window (default 60 seconds) at a near-identical price (default within 0.5%) and quantity. A match on both dimensions, within the time window, is flagged as a likely wash trade pair.

**Why price AND quantity AND time:** Any single dimension alone produces too many false positives — two unrelated trades can easily share a similar price in a liquid market. Requiring all three to align tightly is what makes the match suspicious.

**Known limitation:** This pairwise approach doesn't scale well (O(n²) worst case per symbol) and misses *multi-hop* wash trading (A→B→C→A), which is common in more sophisticated schemes. A production system would typically build a transaction graph and look for near-closed cycles among a small set of accounts, rather than pure pairwise matching.

## 3. Insider trading detection

**Pattern:** A small number of accounts accumulate an abnormally large position ahead of material non-public information becoming public, then the price moves sharply once the news is released.

**Implementation is two-stage:**

1. **Event detection** (`detect_price_volume_events`): aggregate trades by symbol/day, compute rolling z-scores on both price change and volume, and flag days that look like a material event.
2. **Pre-event accumulation** (`detect_pre_event_accumulation`): given an event date (detected or externally known, e.g. an earnings date), look backward a configurable window (default 48 hours) and flag accounts whose total buy quantity in that window is a statistical outlier relative to other accounts trading the same symbol in the same window.

**Why two stages:** In practice, you often don't know the event date in advance — stage 1 exists to surface *candidate* event dates from the data itself, which stage 2 can then be pointed at. In this demo, the event date is known (since it's synthetically injected), but the pipeline is built to support the unsupervised case.

**Known limitation:** This only looks at *volume* anomalies, not the many other signals real insider trading surveillance uses — unusual options activity, trading in economically related securities, employee/insider relationship mapping (e.g., known associates of company insiders), and NLP on news timing. It also assumes a single clean "event" — real markets have overlapping news flow that makes isolating one causal event much harder.

## On thresholds and the precision/recall tradeoff

All three detectors use z-score thresholds that trade off precision against recall. In this demo, thresholds were tuned to prioritize **precision** (few false positives) over recall, on the reasoning that in a real compliance workflow, every flagged case typically requires costly manual review — so a high false positive rate erodes analyst trust in the system and wastes investigative capacity.

In a production setting, these thresholds would ideally be calibrated against a labeled dataset of confirmed historical cases (e.g., past enforcement actions or internally confirmed manipulation), rather than fixed constants — and would likely be re-tuned per-symbol or per-asset-class given very different volatility/liquidity profiles across instruments.
