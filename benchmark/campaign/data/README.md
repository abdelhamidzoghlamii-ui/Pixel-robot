# Reusing campaign data

Rebuild offline from the unchanged JSON archive with `python3 -B benchmark/campaign/campaign_data.py` or `python3 -B benchmark/campaign/summarize_campaign.py`. Verify with `python3 -B benchmark/campaign/self_check_p23d.py`. No model, root, camera or hardware is involved.

The [schema](schema.md) defines units, clocks and missing values. [Run cards](../runs/cards/) retain model hashes, matching code revision and evidence hashes. [Results](../CAMPAIGN_RESULTS.md) preserve validity labels. Older archives are not converted; see [inventory](../../UNARCHIVED_INVENTORY.md).

BIG-cluster M2 median ms versus policy6 max frequency, VALID runs only:

```python
import statistics
import sys
sys.path.insert(0, 'benchmark/campaign')
from campaign_data import load_card, iter_calls

by_frequency = {}
for call in iter_calls():
    if call['label'] != 'VALID' or call['kind'] != 'M2':
        continue
    card = load_card(call['run'])
    cluster = card['config']['m2_cluster']
    if cluster is None:
        cluster = next((p['m2_cluster'] for p in card['config']['by_phase']
                        if p['phase_id'] == call['phase_id']), None)
    if not cluster or cluster['cpus'] != [6, 7]:
        continue
    if call['policy6_max_mhz'] and call['duration_ms']:
        by_frequency.setdefault(float(call['policy6_max_mhz']), []).append(float(call['duration_ms']))
for mhz, values in sorted(by_frequency.items()):
    print(mhz, len(values), statistics.median(values))
```

The loader streams CSV dictionaries: numeric and boolean values are strings; missing values are empty strings. Convert only fields needed by a query. A frequency is the last recorded policy maximum at or before call start, not the CPU's instantaneous frequency. Its age is retained. Association does not establish causality.
