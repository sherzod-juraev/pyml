# Security Policy

## Supported versions

Only the latest release of pyml gets security fixes.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Use the private reporting of GitHub: open the **Security and quality** tab of this repository
and choose **Report a vulnerability**. Describe what you found, the version it affects and, if
you can, how to reproduce it.

This is a personal project run by one maintainer, so I may be slow, but I will read every
report and answer it as soon as I can.

## What is and is not a security issue

A report is welcome if the build, CI or release process of pyml, or a dependency as pyml uses
it, can be used to put harmful content in front of users.

Estimators are ordinary Python objects and can be pickled. Unpickling data from an untrusted
source can run arbitrary code. This is a property of Python's `pickle`, not a vulnerability in
pyml, so load only pickles you trust.

Like NumPy, this library is not meant to be exposed directly to untrusted users: somebody who
can already run Python code in your process does not need a vulnerability here. A crash or a
wrong result on malformed input is a regular bug, so report it as an ordinary issue.
