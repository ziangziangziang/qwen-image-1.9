PYTHON ?= python3
PYTHONPATH := src

.PHONY: test install-editable preflight-dry-run merge-dry-run abliterate-dry-run quantize-dry-run report-dry-run

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

install-editable:
	$(PYTHON) -m pip install -e .

preflight-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli preflight --dry-run

merge-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli merge --dry-run --run-id local-merge-dry-run

abliterate-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli abliterate --dry-run --run-id local-merge-dry-run --input-checkpoint s3://artifacts/example-merged-model

quantize-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli quantize --dry-run --run-id local-merge-dry-run --input-checkpoint s3://artifacts/example-abliterated-model

report-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli report
