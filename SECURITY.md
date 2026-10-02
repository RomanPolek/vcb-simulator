# Security

Security fixes are made on the current `main` branch. Older snapshots do not
have a separate support commitment.

Please report vulnerabilities through
[GitHub private vulnerability reporting](https://github.com/RomanPolek/vcb-simulator/security/advisories/new).
Include reproduction steps, affected versions, and the potential impact.
Avoid posting exploit details or sensitive circuit data in a public issue.

Only load projects and external assembly files from sources you trust. The
simulator parses project data and compiles a native library locally using GCC.
