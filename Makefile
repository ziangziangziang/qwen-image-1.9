PYTHON ?= python3
PYTHONPATH := src

.PHONY: test install-editable merge-dry-run post-merge-train-dry-run abliterate-dry-run post-abliterate-train-dry-run quantize-dry-run eval-dry-run report-dry-run serve

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

install-editable:
	$(PYTHON) -m pip install -e .

merge-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli merge --dry-run --run-id demo-run

post-merge-train-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli post-merge-train --dry-run --run-id demo-run

abliterate-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli abliterate --dry-run --run-id demo-run

post-abliterate-train-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli post-abliterate-train --dry-run --run-id demo-run

quantize-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli quantize --dry-run --run-id demo-run

eval-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli eval --dry-run --run-id demo-run

report-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli report

serve:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli report --serve
