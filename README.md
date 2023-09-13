# Metrics for Code Generation 

A library for running 

## Installation

To start using the library, install it via pip or use your favorite package manager:
```
pip install codegen-metrics
```

## Usage

To start using the library, import a metric and pass to it the reference code and the generated one.

**Please, notice!** You should call it as `metric(reference_code, generated_code)`, not the other way around! 

```python
from codegen_metrics import bleu, chrf, codebleu, meteor, rougel, ruby

metrics = (bleu, chrf, codebleu, meteor, rougel, ruby)
generated_code = "def hello():\n\tprint('Hello, world!')"
reference_code = "def bye(name: str):\n\tprint(f'bye, {name}!')"

for metric in metrics:
    print(metric(reference_code, generated_code))
```

## Remarks

1. Computation of RUBY involves comparison of graphs and takes a lot of time.
2. On the initial run, the library will setup `tree-sitter` and `nltk` which may take some time. 
