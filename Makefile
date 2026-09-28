.PHONY: data analysis quick clean

## build the precomputed tables from the raw MCS files (one-off, ~15 hours)
data:
	python scripts/prepare_data.py

## run the analysis on the precomputed tables
analysis:
	python scripts/run_pipeline.py

## smoke test: one CV repeat, small bootstrap
quick:
	python scripts/run_pipeline.py --quick

## remove saved intermediates (keeps the precomputed tables)
clean:
	rm -rf data/interim/*
