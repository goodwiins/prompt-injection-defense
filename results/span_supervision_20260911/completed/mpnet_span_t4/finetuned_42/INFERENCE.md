# MPNet research inference package

Keep checkpoint.pt, inference_config.json, mpnet_inference.py and inference_assets/ together. The encoder, tokenizer, pooling, head and threshold load offline. No pretrained weights are downloaded.

Reference environment: torch==2.9.1, transformers==4.57.3, sentence-transformers==5.1.2, numpy==2.3.5.

Run `python mpnet_inference.py predict --package . --input page.html --device cuda` on a T4. Use `--device cpu` for float32 inference; small numerical differences from the CUDA reference are possible. Long inputs are processed completely and may take longer. Empty input raises an error.

The output is a detector score and decision, not a security authorization.
