# Versioning of the repository

## Overview

The versioning follows an **X.Y.Z** pattern with the following rules for their incrementation:

- **X**: API changes
  - code will not compile (or will fail at runtime after loading the TVM libraries dynamically)
  - e.g. updated descriptor interface
  - e.g. updated TVM version

- **Y**: behavior changes
  - changes in either performance or results (or failure to run altogether) when doing the inference
  - e.g. updated model artifacts
    - models modified (changed model files, added new models, deleted models, ...)
    - scripts modified (change in compilation options, change in targeted backends, ...)

- **Z**: convenience changes
  - e.g. updated docker image

The maintainers have to assess whether changes to the repository fall into any of the previous categories and tag accordingly.
Reason for not automating the process: allows for intelligent control of the versioning which in turn minimizes the amount of artifacts uploaded.

## Artifact management

Compiled models are uploaded as archives to the `autoware-modelzoo` S3 bucket.
The path to the files are of the form "${version}/${model_name}-${model_arch}-${model_backend}-${version}.tar.gz".
For each "${version}" directory, the list of available models is listed in "${version}/index-${model_arch}".

### Latest

The `:latest` tag of the docker image and the `latest` version of the models artifacts are updated on merges, when relevant files were modified in the merged commits.

### Releases

A release corresponds to a new tag, containing the version number, being pushed to the repository.

An `:X.Y.Z` tag of the docker image is created on all increments of the version number.

An `X.Y.0-date` version of the models artifacts is created on increments of the X and Y version numbers.
With `date` being formatted as `YYYYMMDD`.
