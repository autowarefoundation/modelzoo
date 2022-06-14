# Versioning of the repository

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
