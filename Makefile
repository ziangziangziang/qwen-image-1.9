PYTHON ?= python3
PYTHONPATH := src
# GPU env overrides — set in .env or shell for your hardware.
# AMD MI300X default: HSA_OVERRIDE_GFX_VERSION=9.4.2
HF_GPU_ENV ?= $(shell cat .env 2>/dev/null | grep -E '^HSA_OVERRIDE' | tr '\n' ' ')

.PHONY: test install-editable preflight-dry-run preflight-benchmark \
        merge-dry-run post-merge-train-dry-run \
        abliterate-dry-run post-abliterate-train-dry-run quantize-dry-run \
        eval-dry-run report-dry-run publish-dry-run serve \
        measure-directions pipeline-execute pipeline-smoke gpu-check

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

install-editable:
	$(PYTHON) -m pip install -e .

gpu-check:
	sg render -c "$(HF_GPU_ENV) $(PYTHON) -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))\""

preflight-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli preflight --dry-run --run-id demo-run

preflight-benchmark:
	sg render -c "$(HF_GPU_ENV) PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli preflight \
		--execute --run-id $(RUN_ID) --stress-test --stress-test-seconds $(BENCHMARK_SECONDS)"

BENCHMARK_SECONDS ?= 300

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

publish-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli publish --dry-run --run-id demo-run

serve:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.cli report --serve

RUN_ID ?= prod-$(shell date +%Y%m%d)

measure-directions:
	sg render -c "$(HF_GPU_ENV) PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m qwen_image_19.measure_directions \
		--checkpoint reports/runs/$(RUN_ID)/merge/merged-checkpoint \
		--output reports/abliterate/measurements.pt --layers 18 --device cuda"

pipeline-smoke:
	sg render -c "$(HF_GPU_ENV) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_pipeline.py \
		--run-id smoke-$(shell date +%Y%m%d-%H%M%S) --execute --smoke"

pipeline-execute:
	sg render -c "$(HF_GPU_ENV) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_pipeline.py \
		--run-id $(RUN_ID) --execute --resume"

download-datasets:
	HF_DATASETS_CACHE=$${HF_DATASETS_CACHE:-/scratch/hf-cache/huggingface/datasets} \
		$(PYTHON) scripts/download_datasets.py
