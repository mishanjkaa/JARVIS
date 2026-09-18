from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import SplitResult, urlsplit, urlunsplit

from app.brain.browser.errors import BrowserPolicyError

_FORBIDDEN_SCHEMES = {
    "file",
    "javascript",
    "data",
    "blob",
    "chrome",
    "browser-extension",
    "about",
}
_DEFAULT_ALLOWED_SCHEMES = {"https"}
_METADATA_HOSTS = {
    "169.254.169.254",
    "metadata.google.internal",
    "metadata",
}


def _default_resolver(hostname: str) -> list[str]:
    infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    addresses = sorted({item[4][0] for item in infos if item[4]})
    return addresses


@dataclass
class BrowserUrlPolicy:
    allow_http: bool = False
    allowed_test_hosts: set[str] = field(default_factory=set)
    resolver: Callable[[str], list[str]] = _default_resolver

    def validate_url(self, raw_url: str) -> str:
        normalized = self._normalize(raw_url)
        split = urlsplit(normalized)
        self._validate_split(split)
        self._validate_host(split.hostname or "")
        return urlunsplit(split)

    def validate_redirect_chain(self, urls: list[str]) -> list[str]:
        validated: list[str] = []
        for value in urls:
            validated.append(self.validate_url(value))
        return validated

    def _normalize(self, raw_url: str) -> str:
        if not isinstance(raw_url, str) or not raw_url.strip():
            raise BrowserPolicyError("URL is required.")
        value = raw_url.strip()
        split = urlsplit(value)
        if not split.scheme:
            raise BrowserPolicyError("URL must include http:// or https://.")
        return value

    def _validate_split(self, split: SplitResult) -> None:
        scheme = split.scheme.lower()
        allowed_schemes = set(_DEFAULT_ALLOWED_SCHEMES)
        if self.allow_http:
            allowed_schemes.add("http")
        if scheme in _FORBIDDEN_SCHEMES:
            raise BrowserPolicyError("That URL scheme is not allowed.")
        if scheme not in allowed_schemes:
            raise BrowserPolicyError("That URL scheme is not allowed.")
        if not split.hostname:
            raise BrowserPolicyError("URL host is invalid.")
        if split.username or split.password or "@" in split.netloc:
            raise BrowserPolicyError("URLs with embedded credentials are not allowed.")

    def _validate_host(self, host: str) -> None:
        normalized_host = host.strip().strip("[]").lower()
        if not normalized_host:
            raise BrowserPolicyError("URL host is invalid.")
        if normalized_host in _METADATA_HOSTS:
            raise BrowserPolicyError("That URL is not allowed.")
        if normalized_host in {item.lower() for item in self.allowed_test_hosts}:
            return
        direct_ip = _parse_ip_host(normalized_host)
        if direct_ip is not None:
            _validate_public_ip(direct_ip)
            return
        if normalized_host in {"localhost", "localhost.localdomain"}:
            raise BrowserPolicyError("Local addresses are not allowed.")
        try:
            resolved = self.resolver(normalized_host)
        except OSError as error:
            raise BrowserPolicyError("URL host could not be resolved safely.") from error
        if not resolved:
            raise BrowserPolicyError("URL host could not be resolved safely.")
        for item in resolved:
            ip_value = _parse_ip_host(item)
            if ip_value is None:
                raise BrowserPolicyError("URL host could not be resolved safely.")
            _validate_public_ip(ip_value)


def _parse_ip_host(value: str) -> ipaddress._BaseAddress | None:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        pass
    try:
        if value.lower().startswith("0x"):
            numeric = int(value, 16)
            return ipaddress.ip_address(numeric)
        if value.isdigit():
            return ipaddress.ip_address(int(value))
    except ValueError:
        return None
    return None


def _validate_public_ip(value: ipaddress._BaseAddress) -> None:
    if value.is_loopback or value.is_private or value.is_link_local or value.is_multicast or value.is_reserved or value.is_unspecified:
        raise BrowserPolicyError("Local or private network addresses are not allowed.")
