PYTHON ?= python3
PYTHONPATH := src

.PHONY: test stage1-dry-run stage2-dry-run stage3-dry-run stage4-dry-run stage5-dry-run

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

stage1-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli stage1 analyze --dry-run

stage2-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli stage2 fuse --dry-run

stage3-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli stage3 eval --dry-run

stage4-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli stage4 quantize --dry-run

stage5-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli stage5 deploy --dry-run

