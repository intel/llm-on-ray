Dockerfiles for users to convenient build containers. 
1.Dockerfile.user for user to build llm-on-ray with docker on Intel CPU.
2.Dockerfile.habana for user to build llm-on-ray with docker on Intel GPU .

Dockerfiles for CI tests in 'ci/*'. 
In CI, the environment required by different models is separated, and the dockerfiles with different functions are distinguished by different suffixes.

There could be one Dockerfile with ARG declared to distinguish different pip extras. However, ARG will bust cache of 'pip install', which usually takes long time, when build docker image. Instead, we have two almost identical Dockerfiles here to improve CI efficiency. 
