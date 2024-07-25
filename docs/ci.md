# CI self-hosted workflow on CPU/Gaudi

LLM-on-Ray introduces CI workflows with or without self-hosted runner, allowing developers to easily run for a specific PR including CPU and Gaudi. It can also be deployed to run automatically in self-hosted nodes.

## Description
The workflow starts from loading state file which stores the states of handled PRs. Then it will get all opened PRs from Github, to check if there is any PR new or has new commits. After getting the list of PRs which need to be checked, it will run with CPU or Gaudi(depends on setting `is_gaudi`) and post a running state to PR. When the tests are done, the result success or failed will be posted again to PR page. Meanwhile, local state file will be update.

## Sandbox project repo
[intel-sandbox/self-hosted-workflow](https://github.com/intel-sandbox/self-hosted-workflow)

## Setup Environment
Use `pip list` or `pip3 list` to make sure urllib3>=1.26 and requests>=2.25.

## How to use
Main script can be found at `llm-on-ray-ci/llm-on-ray-ci.py`.
There are two args, the first one is `is_gaudi`, which must be set to true or false, and it stands for running Guadi tests or CPU tests. The second one is 'pr_number', which is optional and it stands for whether specific a PR to run. 


## Examples

### 1. Run PR#268 on Gaudi
```bash
python llm-on-ray-ci/llm-on-ray-ci.py true 268
```

If the result is good:

![ci_gaudi_success](./assets/ci_gaudi_success.png)


### 2. Run all PRs on CPU which need update
```bash
python llm-on-ray-ci/llm-on-ray-ci.py false
```

If the result is good:

![ci_cpu_success](./assets/ci_cpu_success.png)


## Whitelist for developer
Whitelist is located on `llm-on-ray-ci/author_whitelist.json`. Only on this list will the authors' PR be checked.
