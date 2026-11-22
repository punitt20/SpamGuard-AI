# SpamGuard AI — Exploratory Data Analysis

Generated: 2026-11-18T10:07:17+00:00

> Descriptive only. No classifier fitting or test-set-driven decisions were made in this analysis.

## Dataset summary

| Statistic | SMS Spam Collection | Phishing email dataset |
|---|---:|---:|
| Raw rows | 5,572 | 18,650 |
| Missing messages | 0 | 16 |
| Duplicate messages (raw) | 403 | 1,112 |
| Clean rows | 5,158 | 17,521 |
| Training rows (80%) | 4,126 | 14,016 |
| Test rows (20%) | 1,032 | 3,505 |
| Imbalance ratio | 7.03:1 | 1.68:1 |
| Mean length (chars) | 79.2 | 1,771.4 |
| Median length (chars) | 61.0 | 915.0 |

## Class counts

- SMS: HAM 4,516; SPAM 642.
- Phishing: LEGITIMATE 10,978; PHISHING 6,543.
- The phishing corpus contains 87 emails longer than 20,000 characters; those were truncated for modelling only.

## SMS: top spam terms

call (303), free (191), txt (134), stop (113), text (107), mobile (106), reply (96), claim (95), prize (79), cash (64), send (59), urgent (57), won (56), nokia (53), msg (50)

## SMS: top legitimate terms

i'm (370), ok (254), go (247), good (217), come (214), call (192), time (187), love (185), day (185), going (164), lor (159), home (156), need (154), da (142), i'll (137)

## SMS: top spam bigrams

"please call" (40), "po box" (21), "call landline" (21), "guaranteed call" (20), "prize guaranteed" (19), "send stop" (18), "urgent mobile" (17), "national rate" (17), "selected receive" (16), "reply call" (16)

## Phishing: top phishing terms

email (5013), com (4511), free (4141), mail (3713), company (3693), money (3674), please (3641), information (3603), business (3374), time (3083), http (3006), report (2989), click (2791), order (2505), people (2298)

## Phishing: top legitimate terms

enron (14041), ect (10899), com (6012), university (6005), language (5867), please (5866), hou (5218), information (4976), subject (4667), list (4535), mail (4299), time (4113), pm (3673), re (3566), message (3315)

## Figures

All figures are available in `reports/figures/`:

### Sms
- Class Distribution: `reports/figures/sms_class_distribution.png`
- Message Lengths: `reports/figures/sms_message_length_distribution.png`
- Length By Class: `reports/figures/sms_length_by_class.png`
- Top Spam Words: `reports/figures/sms_top_spam_words.png`
- Top Ham Words: `reports/figures/sms_top_ham_words.png`
- Top Spam Bigrams: `reports/figures/sms_top_spam_bigrams.png`

### Phishing
- Class Distribution: `reports/figures/phishing_class_distribution.png`
- Message Lengths: `reports/figures/phishing_message_length_distribution.png`
- Length By Class: `reports/figures/phishing_length_by_class.png`
- Top Phishing Words: `reports/figures/phishing_top_phishing_words.png`
- Top Legitimate Words: `reports/figures/phishing_top_legitimate_words.png`
