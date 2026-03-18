# Quantization Notes

## Goal
Get to smaller artifacts without erasing the parts that make the model useful for image work.

## Policy
- GGUF path must declare imatrix provenance.
- EXL2/GPTQ path must document the serving target and hardware assumptions.
- All quantization results get compared back to the BF16 reference artifact.

