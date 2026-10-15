.PHONY: up down install test tune-threshold tune-rate-limits tricky-tests smoke-test dashboard producer consumer prepare-data bakeoff latency-bench export-feedback

up:
	docker compose up -d

down:
	docker compose down

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v || (python tests/test_simhash_cluster.py && python tests/test_feedback.py)

tune-threshold:
	python scripts/tune_simhash_threshold.py

tune-rate-limits:
	python scripts/tune_rate_limits.py

tricky-tests:
	python scripts/run_tricky_tests.py

smoke-test:
	python scripts/smoke_test.py

prepare-data:
	python scripts/prepare_data.py --raw_csv $(RAW_CSV) --output_dir $(OUTPUT_DIR)

bakeoff:
	python scripts/model_bakeoff.py --data_dir $(DATA_DIR) --mode $(MODE)

latency-bench:
	python scripts/latency_benchmark.py --mode placeholder --iterations 200

export-feedback:
	python scripts/export_feedback_queue.py --output $(OUTPUT)

dashboard:
	streamlit run dashboard/app.py

producer:
	python producer/producer.py

consumer:
	python consumer/consumer.py
