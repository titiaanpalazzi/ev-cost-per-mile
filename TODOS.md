# TODOS

## P2: Vectorize run_simulation() loop
The row-by-row `iterrows()` loop in `run_simulation()` is the performance ceiling.
With 50K rides it takes ~5-10 seconds. Vectorizing with numpy would bring it under
1 second and remove the need for the 50K sampling cap.

**Why:** Enables larger datasets (200K+ rides from customer data) without artificial limits.
**Effort:** M (human: ~1 day / CC: ~30 min)
**Depends on:** ev_model.py extraction
**Added:** 2026-03-29 via /plan-ceo-review

## P3: Customer ride data integration
Once CSV upload proves valuable, build a structured data pipeline: persisted ride
profiles per customer, automatic scenario generation from customer fleet data.

**Why:** CSV upload is manual and one-shot. Repeat customer analysis needs persisted profiles.
**Effort:** L (human: ~1 week / CC: ~2-3 hours)
**Depends on:** CSV upload validation + real usage data proving the concept
**Added:** 2026-03-29 via /plan-ceo-review

## P3: SOC stranding detection
The simulation assumes the vehicle can always reach the nearest depot for charging.
When `energy_to_depot > current_soc`, the model sets `soc = max(0, soc - energy)` and
charges anyway — the vehicle "arrives" with negative effective energy. This silently
produces optimistic numbers when battery is small or depots are far apart.

**Why:** An analyst with a 30 kWh battery and a single distant depot gets a cost model
that ignores stranding events. The fix: detect when soc < energy_to_depot, flag it as
a stranding event, and either penalize the cost model or surface a warning.
**Effort:** S (human: ~2h / CC: ~5 min)
**Depends on:** ev_model.py extraction
**Added:** 2026-03-30 via /plan-eng-review (outside voice finding)
