.PHONY: test demo live clean

test:
	python3 tests/test_basic.py

demo:
	python3 -m jammate --demo

demo-jazz:
	python3 -m jammate --demo --jazz

demo-blues:
	python3 -m jammate --demo --blues

demo-rock:
	python3 -m jammate --demo --rock

live:
	python3 -m jammate

config:
	python3 -m jammate --config

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f demo_output.wav test_output.wav
