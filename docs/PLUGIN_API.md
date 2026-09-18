# JARVIS 2.5.0 Plugin API

Plugins are trusted application extensions, not an arbitrary-code marketplace. Only IDs and module paths in the application-controlled trusted plugin map may load. Manifests cannot choose import paths, register callable names, or authorize extra tools.

Enabling a plugin imports trusted Python code. Disabled plugins are not imported. Plugin tools use the same validators, registry, policy, and executor as built-in tools. Plugins cannot replace built-ins, register confirmation commands, lower risk, or approve plans.

There is no directory scanning, downloading, dependency installation, or model-controlled plugin loading.
