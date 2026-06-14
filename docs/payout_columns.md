# Payout Columns

This check uses `PRAGMA table_info` and limited samples only. It does not run full-table counts or distinct scans.

## Payout-related columns

| table | column                   | type    | pk | inferred_role                       |
| ----- | ------------------------ | ------- | -- | ----------------------------------- |
| NL_HR | TanUmaban                | TEXT    | 0  | win payout horse-number candidate   |
| NL_HR | TanPay                   | BIGINT  | 0  | win payout amount candidate         |
| NL_HR | TanNinki                 | INTEGER | 0  | popularity/rank                     |
| NL_HR | FukuUmaban               | TEXT    | 0  | place payout horse-number candidate |
| NL_HR | FukuPay                  | BIGINT  | 0  | place payout amount candidate       |
| NL_HR | FukuNinki                | INTEGER | 0  | popularity/rank                     |
| NL_HR | WakuKumi                 | TEXT    | 0  | combination/horse-number candidate  |
| NL_HR | WakuPay                  | BIGINT  | 0  | payout/refund candidate             |
| NL_HR | WakuNinki                | INTEGER | 0  | popularity/rank                     |
| NL_HR | UmarenKumi               | TEXT    | 0  | combination/horse-number candidate  |
| NL_HR | UmarenPay                | BIGINT  | 0  | payout/refund candidate             |
| NL_HR | UmarenNinki              | INTEGER | 0  | popularity/rank                     |
| NL_HR | WideKumi                 | TEXT    | 0  | combination/horse-number candidate  |
| NL_HR | WidePay                  | BIGINT  | 0  | payout/refund candidate             |
| NL_HR | WideNinki                | INTEGER | 0  | popularity/rank                     |
| NL_HR | UmatanKumi               | TEXT    | 0  | combination/horse-number candidate  |
| NL_HR | UmatanPay                | BIGINT  | 0  | win payout amount candidate         |
| NL_HR | UmatanNinki              | INTEGER | 0  | popularity/rank                     |
| NL_HR | SanrenfukuKumi           | TEXT    | 0  | combination/horse-number candidate  |
| NL_HR | SanrenfukuPay            | BIGINT  | 0  | place payout amount candidate       |
| NL_HR | SanrenfukuNinki          | INTEGER | 0  | popularity/rank                     |
| NL_HR | SanrentanKumi            | TEXT    | 0  | combination/horse-number candidate  |
| NL_HR | SanrentanPay             | BIGINT  | 0  | win payout amount candidate         |
| NL_HR | SanrentanNinki           | INTEGER | 0  | popularity/rank                     |
| NL_H1 | FukuChakuBaraiKey        | TEXT    | 0  | context                             |
| NL_H1 | BetType                  | TEXT    | 7  | bet type discriminator              |
| NL_H1 | Kumi                     | TEXT    | 8  | combination or horse-number key     |
| NL_H1 | Ninki                    | INTEGER | 0  | popularity/rank                     |
| NL_H1 | TanHyoTotal              | BIGINT  | 0  | context                             |
| NL_H1 | FukuHyoTotal             | BIGINT  | 0  | context                             |
| NL_H1 | UmatanHyoTotal           | BIGINT  | 0  | context                             |
| NL_H1 | SanrenfukuHyoTotal       | BIGINT  | 0  | context                             |
| NL_H1 | TanHenkanHyoTotal        | BIGINT  | 0  | context                             |
| NL_H1 | FukuHenkanHyoTotal       | BIGINT  | 0  | context                             |
| NL_H1 | UmatanHenkanHyoTotal     | BIGINT  | 0  | context                             |
| NL_H1 | SanrenfukuHenkanHyoTotal | BIGINT  | 0  | context                             |

## NL_H1 columns

| column                   | type    | pk |
| ------------------------ | ------- | -- |
| RecordSpec               | TEXT    | 0  |
| DataKubun                | TEXT    | 0  |
| MakeDate                 | TEXT    | 0  |
| Year                     | INTEGER | 1  |
| MonthDay                 | INTEGER | 2  |
| JyoCD                    | TEXT    | 3  |
| Kaiji                    | INTEGER | 4  |
| Nichiji                  | INTEGER | 5  |
| RaceNum                  | INTEGER | 6  |
| TorokuTosu               | INTEGER | 0  |
| SyussoTosu               | INTEGER | 0  |
| HatubaiFlag1             | TEXT    | 0  |
| HatubaiFlag2             | TEXT    | 0  |
| HatubaiFlag3             | TEXT    | 0  |
| HatubaiFlag4             | TEXT    | 0  |
| HatubaiFlag5             | TEXT    | 0  |
| HatubaiFlag6             | TEXT    | 0  |
| HatubaiFlag7             | TEXT    | 0  |
| FukuChakuBaraiKey        | TEXT    | 0  |
| HenkanUma                | TEXT    | 0  |
| HenkanWaku               | TEXT    | 0  |
| HenkanDoWaku             | TEXT    | 0  |
| BetType                  | TEXT    | 7  |
| Kumi                     | TEXT    | 8  |
| Hyo                      | BIGINT  | 0  |
| Ninki                    | INTEGER | 0  |
| TanHyoTotal              | BIGINT  | 0  |
| FukuHyoTotal             | BIGINT  | 0  |
| WakuHyoTotal             | BIGINT  | 0  |
| UmarenHyoTotal           | BIGINT  | 0  |
| WideHyoTotal             | BIGINT  | 0  |
| UmatanHyoTotal           | BIGINT  | 0  |
| SanrenfukuHyoTotal       | BIGINT  | 0  |
| TanHenkanHyoTotal        | BIGINT  | 0  |
| FukuHenkanHyoTotal       | BIGINT  | 0  |
| WakuHenkanHyoTotal       | BIGINT  | 0  |
| UmarenHenkanHyoTotal     | BIGINT  | 0  |
| WideHenkanHyoTotal       | BIGINT  | 0  |
| UmatanHenkanHyoTotal     | BIGINT  | 0  |
| SanrenfukuHenkanHyoTotal | BIGINT  | 0  |

## NL_H1 limited sample

First 50 `NL_H1` rows were inspected. The payout-relevant fields visible in this table are `BetType`, `Kumi`, `Hyo`, and `Ninki`.

| race             | BetType | Kumi | Hyo   | Ninki |
| ---------------- | ------- | ---- | ----- | ----- |
| 2016010506010101 | Tansyo  | 01   | 856   | 14    |
| 2016010506010101 | Tansyo  | 02   | 1935  | 11    |
| 2016010506010101 | Tansyo  | 03   | 11077 | 6     |
| 2016010506010101 | Tansyo  | 04   | 557   | 16    |
| 2016010506010101 | Tansyo  | 05   | 30115 | 3     |
| 2016010506010101 | Tansyo  | 06   | 1418  | 13    |
| 2016010506010101 | Tansyo  | 07   | 10902 | 7     |
| 2016010506010101 | Tansyo  | 08   | 6952  | 8     |
| 2016010506010101 | Tansyo  | 09   | 2483  | 10    |
| 2016010506010101 | Tansyo  | 10   | 23520 | 4     |
| 2016010506010101 | Tansyo  | 11   | 20460 | 5     |
| 2016010506010101 | Tansyo  | 12   | 45755 | 1     |
| 2016010506010101 | Tansyo  | 13   | 42515 | 2     |
| 2016010506010101 | Tansyo  | 14   | 716   | 15    |
| 2016010506010101 | Tansyo  | 15   | 1456  | 12    |
| 2016010506010101 | Tansyo  | 16   | 3128  | 9     |
| 2016010506010101 | Fukusyo | 01   | 1671  | 15    |
| 2016010506010101 | Fukusyo | 02   | 3249  | 12    |
| 2016010506010101 | Fukusyo | 03   | 23099 | 6     |
| 2016010506010101 | Fukusyo | 04   | 1242  | 16    |

## NL_H1 BetType inference

For the first sampled race, observed `BetType` values in a `LIMIT 500` sample were: `Fukusyo`, `Sanrenpuku`.

`Fukusyo` rows use `Kumi` as the horse number and `Hyo` as vote count. No payout amount column was found in `NL_H1`; `Hyo` is not a refund amount.

## Other schema candidates

`NL_HA` has `PayKumi1-3`, `PayAmount1-3`, `TotalPay`, and `PayoutCount`, which look like normalized payout slots. However, `SELECT * FROM NL_HA LIMIT 5` returned no rows in this DB, so it cannot currently be used as the place payout source.

`NL_WF` has `Kumi` and `PayJyushosiki`, but sampled rows look like WIN5-style aggregate data rather than horse-level place payout rows.
