from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin


_WHITESPACE_RE = re.compile(r"\s+")
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_TEXT_INPUT_TYPES = {"", "text", "search", "email", "url", "tel"}


def normalize_visible_text(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return _WHITESPACE_RE.sub(" ", value).strip()


class _VisibleHtmlParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._hidden_stack: list[bool] = []
        self._open_elements: list[dict[str, Any]] = []
        self._text_parts: list[str] = []
        self._elements: list[dict[str, Any]] = []
        self._controls: list[dict[str, Any]] = []
        self._label_stack: list[dict[str, Any]] = []
        self._label_by_id: dict[str, str] = {}
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name.lower(): (value or "") for name, value in attrs}
        hidden = self._is_hidden(tag, attr_map)
        inherited_hidden = self._hidden_stack[-1] if self._hidden_stack else False
        self._hidden_stack.append(hidden or inherited_hidden)
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() == "label":
            self._label_stack.append({"for": attr_map.get("for", "").strip(), "text": "", "control_indexes": []})
        element = self._element_candidate(tag.lower(), attr_map, hidden or inherited_hidden)
        if element is not None:
            self._open_elements.append(element)
        control = self._control_candidate(tag.lower(), attr_map, hidden or inherited_hidden)
        if control is not None:
            control["index"] = len(self._controls)
            self._controls.append(control)
            if self._label_stack:
                self._label_stack[-1]["control_indexes"].append(control["index"])

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = False
        if self._hidden_stack:
            self._hidden_stack.pop()
        for index in range(len(self._open_elements) - 1, -1, -1):
            if self._open_elements[index]["tag"] != lowered:
                continue
            element = self._open_elements.pop(index)
            element["visible_text"] = normalize_visible_text(element["visible_text"])
            element["accessible_name"] = normalize_visible_text(element["accessible_name"] or element["visible_text"])
            if element["visible"] and (
                element["visible_text"]
                or element["accessible_name"]
                or element["href"]
                or element["tag"] in {"img", "input", "form"}
            ):
                self._elements.append(element)
            break
        if lowered == "label" and self._label_stack:
            label = self._label_stack.pop()
            label_text = normalize_visible_text(label["text"])
            label_for = str(label.get("for") or "").strip()
            if label_for and label_text:
                self._label_by_id[label_for] = label_text
            if label_text:
                for index in label.get("control_indexes", []):
                    if 0 <= index < len(self._controls):
                        self._controls[index]["label"] = normalize_visible_text(
                            f"{self._controls[index].get('label', '')} {label_text}"
                        )

    def handle_data(self, data: str) -> None:
        if not isinstance(data, str) or not data.strip():
            return
        if self._in_title:
            self.title = normalize_visible_text(f"{self.title} {data}")
        hidden = self._hidden_stack[-1] if self._hidden_stack else False
        if hidden:
            return
        cleaned = normalize_visible_text(data)
        if cleaned:
            self._text_parts.append(cleaned)
            for element in self._open_elements:
                element["visible_text"] = normalize_visible_text(f"{element['visible_text']} {cleaned}")
                element["accessible_name"] = normalize_visible_text(f"{element['accessible_name']} {cleaned}")
            for label in self._label_stack:
                label["text"] = normalize_visible_text(f"{label['text']} {cleaned}")

    def visible_text(self) -> str:
        return normalize_visible_text(" ".join(self._text_parts))

    def elements(self) -> list[dict[str, Any]]:
        return list(self._elements)

    def controls(self) -> list[dict[str, Any]]:
        controls: list[dict[str, Any]] = []
        for control in self._controls:
            label = normalize_visible_text(control.get("label", "") or self._label_by_id.get(control.get("id", ""), ""))
            placeholder = normalize_visible_text(control.get("placeholder", ""))
            name = normalize_visible_text(control.get("name", ""))
            submit_text = normalize_visible_text(control.get("submit_text", ""))
            form_text = normalize_visible_text(control.get("form_text", ""))
            visible = bool(control.get("visible")) and not bool(control.get("disabled"))
            if control.get("control_type") not in {"text_input", "textarea"}:
                continue
            if not visible:
                continue
            controls.append(
                {
                    "control_type": control.get("control_type", ""),
                    "tag": control.get("tag", ""),
                    "input_type": control.get("input_type", ""),
                    "label": label,
                    "placeholder": placeholder,
                    "name": name,
                    "disabled": bool(control.get("disabled")),
                    "visible": bool(control.get("visible")),
                    "form_action": control.get("form_action", ""),
                    "form_method": control.get("form_method", "get"),
                    "form_text": form_text,
                    "submit_text": submit_text,
                }
            )
        return controls

    def _is_hidden(self, tag: str, attrs: dict[str, str]) -> bool:
        if tag in {"script", "style", "noscript"}:
            return True
        style = attrs.get("style", "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            return True
        if "hidden" in attrs:
            return True
        if tag == "input" and attrs.get("type", "").lower() == "hidden":
            return True
        return False

    def _element_candidate(self, tag: str, attrs: dict[str, str], hidden: bool) -> dict[str, Any] | None:
        role = attrs.get("role", "")
        element_type = ""
        if tag == "a":
            element_type = "links"
        elif tag == "button":
            element_type = "buttons"
        elif tag == "input":
            element_type = "inputs"
        elif tag == "form":
            element_type = "forms"
        elif tag in _HEADING_TAGS:
            element_type = "headings"
        elif tag == "img":
            element_type = "images"
        if not element_type:
            return None
        href = attrs.get("href", "")
        normalized_href = urljoin(self.base_url, href) if href else ""
        return {
            "element_type": element_type,
            "tag": tag,
            "role": role,
            "visible_text": "",
            "accessible_name": attrs.get("aria-label", "") or attrs.get("alt", "") or attrs.get("value", ""),
            "input_type": attrs.get("type", ""),
            "href": normalized_href,
            "disabled": "disabled" in attrs,
            "visible": not hidden,
            "action": attrs.get("action", ""),
            "method": attrs.get("method", ""),
        }

    def _control_candidate(self, tag: str, attrs: dict[str, str], hidden: bool) -> dict[str, Any] | None:
        if tag == "textarea":
            return {
                "control_type": "textarea",
                "tag": tag,
                "id": attrs.get("id", ""),
                "input_type": "",
                "label": attrs.get("aria-label", ""),
                "placeholder": attrs.get("placeholder", ""),
                "name": attrs.get("name", ""),
                "disabled": "disabled" in attrs,
                "visible": not hidden,
                "form_action": self._nearest_form_attr("action"),
                "form_method": self._nearest_form_attr("method") or "get",
                "form_text": self._nearest_form_text(),
                "submit_text": "",
            }
        if tag != "input":
            return None
        input_type = attrs.get("type", "").strip().lower()
        if input_type == "password":
            return None
        if input_type not in _TEXT_INPUT_TYPES:
            return None
        return {
            "control_type": "text_input",
            "tag": tag,
            "id": attrs.get("id", ""),
            "input_type": input_type or "text",
            "label": attrs.get("aria-label", ""),
            "placeholder": attrs.get("placeholder", ""),
            "name": attrs.get("name", ""),
            "disabled": "disabled" in attrs,
            "visible": not hidden,
            "form_action": self._nearest_form_attr("action"),
            "form_method": self._nearest_form_attr("method") or "get",
            "form_text": self._nearest_form_text(),
            "submit_text": "",
        }

    def _nearest_form_attr(self, name: str) -> str:
        for element in reversed(self._open_elements):
            if element.get("tag") != "form":
                continue
            return str(element.get(name, "") or "")
        return ""

    def _nearest_form_text(self) -> str:
        for element in reversed(self._open_elements):
            if element.get("tag") != "form":
                continue
            return str(element.get("visible_text", "") or "")
        return ""


def extract_visible_text_from_html(html: str, *, base_url: str) -> tuple[str, str]:
    parser = _VisibleHtmlParser(base_url)
    parser.feed(html or "")
    parser.close()
    return parser.visible_text(), parser.title


def inspect_elements_from_html(
    html: str,
    *,
    base_url: str,
    element_types: list[str],
    max_elements: int,
) -> tuple[list[dict[str, Any]], str]:
    parser = _VisibleHtmlParser(base_url)
    parser.feed(html or "")
    parser.close()
    allowed = {item for item in element_types if isinstance(item, str)}
    elements: list[dict[str, Any]] = []
    for element in parser.elements():
        if element["element_type"] not in allowed:
            continue
        elements.append(element)
        if len(elements) >= max_elements:
            break
    return elements, parser.title


def inspect_form_controls_from_html(
    html: str,
    *,
    base_url: str,
    max_controls: int,
) -> tuple[list[dict[str, Any]], str]:
    parser = _VisibleHtmlParser(base_url)
    parser.feed(html or "")
    parser.close()
    controls = parser.controls()[:max_controls]
    for control in controls:
        action = str(control.get("form_action") or "").strip()
        if action:
            control["form_action"] = urljoin(base_url, action)
        method = str(control.get("form_method") or "get").strip().lower()
        control["form_method"] = method or "get"
    return controls, parser.title
