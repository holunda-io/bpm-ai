# bpm-ai
_AI task automation for BPM engines._

See [camunda-8-connector-gpt](https://github.com/holunda-io/camunda-8-connector-gpt) for actual integration with the Camunda Platform 8 BPM engine using custom BPMN elements (Connectors).

## Installation
Requires Python 3.11.
### Default
```bash
$ pip install bpm-ai
```

### For Local Inference
Install PyTorch (remove index url for CUDA GPU support) and spaCy:
```bash
$ pip install torch --index-url https://download.pytorch.org/whl/cpu
$ pip install spacy
```

For Apple Silicon:
```bash
$ pip install torch spacy[apple]
```

Install bpm-ai with inference extra:
```bash
$ pip install bpm-ai[inference]
```

---

## License

This project is developed under

[![Apache 2.0 License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](/LICENSE)

## Sponsors and Customers

[![sponsored](https://img.shields.io/badge/sponsoredBy-Holisticon-red.svg)](https://holisticon.de/)