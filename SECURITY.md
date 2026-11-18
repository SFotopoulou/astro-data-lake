# Security policy

## Supported versions

Only the latest released minor version receives security fixes. The current
release line is 0.10.x.

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately with GitHub's private vulnerability reporting:
<https://github.com/SFotopoulou/astro-data-lake/security/advisories/new>
(Security tab, then "Report a vulnerability").

Include the affected version, the steps to reproduce, and the expected impact.
We aim to acknowledge reports within one week.

## Scope and trust boundaries

The lake treats some inputs as trusted operator input, for example `--where`
predicates, view DDL built from configured paths, and the ingest token. These
are documented in [docs/security.md](docs/security.md). A report that depends
on untrusted text reaching one of those inputs is still welcome, and it will be
handled as a hardening issue.
