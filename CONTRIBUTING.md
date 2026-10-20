# Contributing to SGE-Goal

Thanks for your interest in SGE-Goal. This repository accompanies a research paper, so its main
job is to let others reproduce and build on the published results. Contributions that make that
easier are very welcome.

## Reporting a problem

Open a [GitHub issue](https://github.com/Riddhi-Chatterjee/SGE-Goal/issues) and include:

- what you ran (the exact command and the directory you ran it from),
- the full error or the unexpected output,
- your environment: OS, Python version, GPU, CUDA version, and whether you ran locally or on Modal,
- which checkpoint and dataset you used.

Check the [troubleshooting guide](docs/guides/troubleshooting.md) first; it covers the most common
setup issues.

## Questions

Questions about the paper or the method are best asked in an issue too, so the answer helps
others.

## Pull requests

Good candidates for a pull request:

- bug fixes, with a short description of how you verified them,
- documentation fixes and clarifications,
- making a script easier to run outside our infrastructure (for example, fewer hard-coded paths).

Before starting a larger change, such as a refactor or a new feature, please open an issue to
discuss it, so that it fits the release and does not change published behaviour.

When you open a pull request:

1. Keep it focused on one change.
2. Match the style of the surrounding code.
3. Do not change default hyperparameters, prompts or evaluation settings. These reproduce the
   paper's numbers; if you think one is wrong, raise it in an issue.
4. Never commit secrets (API keys, Hugging Face or Modal tokens), model weights or datasets.
   Credentials are read from environment variables; see the [installation guide](docs/guides/installation.md).

## Licence

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the licence of this repository.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you agree
to uphold it.
