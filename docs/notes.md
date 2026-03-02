## Thoughts
expander - Three target gene queries might be heavy. PRKAA1, PRKAA2, and STK11 are all legit AMPK 
subunits/regulators, but that's 3 of your 9 queries on the same axis. If you're capping at 
8–10 queries, you might want the prompt to say "max 2 target gene queries" so it leaves room 
for other axes. 
Or just let it ride and see if the deduplication in fetch_and_cache handles the overlap.