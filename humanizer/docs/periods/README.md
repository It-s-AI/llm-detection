# Period commitments (D-57)

Each period's seed is `H(secret || blockhash(B))`. The commitment `c = H(secret)` is published first;
B is a finney block produced after it. The secret is revealed with the period.

From the second period on, the rotation (`scripts/rotate_period.py`, `period/lifecycle.py`) does this by
itself. When period t starts, it publishes the next commitment together with the seed block B, about
half a day before t ends: `GET <evaluator>/v1/periods` serves `schedule.json`, and a copy goes to the
public HF repo when one is configured. After B is final, the next period is built, and its Merkle root
is added to the schedule. When t closes, it is revealed under `GET <evaluator>/v1/periods/<id>/`.
Check a reveal with `python scripts/verify_period.py --url <evaluator>/v1/periods/<id> --notary <address>`.

## Notary remarks on chain

The API and the HF copy only show what we say. The chain shows *when*. Twice per period a dedicated
notary wallet (it holds only fees, not the owner's key) posts `System.remark_with_event` on finney:

    sn32-humanizer/1 {"commitment":"<H(secret)>","kind":"commitment","netuid":32,"period":"p2","seed_block":B}
    sn32-humanizer/1 {"block_hash":"<hash of B>","kind":"merkle_root","merkle_root":"<root>","n_items":484,"netuid":32,"period":"p2"}

The schedule records the block of each remark. `verify_period.py` finds both in those blocks' bodies
on a public archive node, checks that they are signed by the notary address, that the commitment's
block is before B, and that the Merkle root's block is before the first evaluation of the period. A
commitment whose remark misses B is abandoned openly (it stays in the schedule under `abandoned`, with
the reason) and a new one is published.

Notary address: `5D76hDgEK6VT3y9XChyWTQUVqSmJEftiCGtxfGBrnnYTjkwH` (the same key on finney and testnet).

The first period, built by hand, has no remarks: its commitment is the row below.

Hand-built periods:

| Period | n | c = H(secret) | Published | Block B |
|---|---|---|---|---|
| p1 | 500 (484 built) | `b0c54ac31365568505794c7325785b930c278e035a248e52a54290bd0c1238d1` | 2026-11-17 | 9526464 |
