import logging
import os
from typing import Optional, Literal, Annotated, Any
from urllib.parse import urlparse

import html2text
import html_text
import requests
from langchain_core.tools import ToolException, InjectedToolArg, BaseTool
from langchain_core.tools.base import create_schema_from_function
from lxml import html
from lxml.html import defs
from lxml_html_clean import Cleaner
from pydantic import BaseModel
from requests import RequestException

from llm_workers.api import ExtendedBaseTool
from llm_workers.config import Json

logger = logging.getLogger(__name__)


def _handle_error(url: str, e: IOError, on_error: Literal['raise_exception', 'return_error', 'return_empty'], empty: Json) -> Json:
    logger.debug(f"Error fetching %s", url, e)
    if on_error == "raise_exception":
        raise RequestException(f"Error fetching {url}: {e}", e)
    elif on_error == "return_error":
        raise ToolException(f"Error fetching {url}: {e}")
    else:
        return empty

def _handle_no_content(url: str, xpath: str, on_no_content: Literal['raise_exception', 'return_error', 'return_empty'], empty: Json) -> Json:
    logger.debug(f"Got empty content for URL %s and xpath %s", url, xpath)
    if on_no_content == "raise_exception":
        raise ValueError(f"No content matching '{xpath}' found at url {url}")
    elif on_no_content == "return_error":
        raise ToolException(f"No content matching '{xpath}' found at url {url}")
    else:
        return empty


def _fetch_content(
        url: str,
        headers: Annotated[dict[str, str], InjectedToolArg] = None,
        on_no_content: Annotated[Literal['raise_exception', 'return_error', 'return_empty'], InjectedToolArg] = 'return_empty',
        on_error: Annotated[Literal['raise_exception', 'return_error', 'return_empty'], InjectedToolArg] = 'raise_exception'
) -> str:
    """Fetches (textual) content from given URL.

    Args:
        url: The URL to fetch from.
        headers: Extra headers to use for the request.
        on_no_content: What to do if no matching content is found (404 or content not text).
        on_error: What to do if an error occurs.
    """
    if headers is None:
        headers = {}
    if not 'User-Agent' in headers:
        # read the user-agent from USER_AGENT environment variable
        useragent = os.getenv('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
        headers['User-Agent'] = useragent

    logger.debug("Fetching content from URL %s", url)
    try:
        response = requests.get(url, headers=headers)

        # Check for a valid response (HTTP Status Code 200)
        if response.status_code != 200:
            if response.status_code == 404:
                return _handle_no_content(url, '', on_no_content, empty="")
            else:
                raise RequestException(f"Got HTTP status code {response.status_code} for {url}")
        # TODO handle non-text content
        return response.content.decode('utf-8', errors='ignore')
    except IOError as e:
        return _handle_error(url, e, on_error, empty = "")

class FetchContentTool(ExtendedBaseTool, BaseTool):
    name: str = "fetch_content"
    description: str = "Fetches text content from a URL and returns it as a string."
    args_schema: type[BaseModel] = create_schema_from_function(
        name,
        _fetch_content,
        parse_docstring=True,
        error_on_invalid_docstring=True,
    )

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return _fetch_content(**kwargs)

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Fetching {input['url']}"

_cleaner = Cleaner(
    scripts=True,
    javascript=True,
    comments=True,
    style=True,
    links=True,
    meta=True,
    page_structure=False,  # <title> may be nice to have
    processing_instructions=True,
    embedded=True,
    frames=True,
    forms=False,  # keep forms
    annoying_tags=True,
    remove_unknown_tags=True,
    safe_attrs_only=True,
    safe_attrs=defs.safe_attrs.copy() - {'class', 'style'}
)


def _fetch_page_markdown(
        url: str,
        xpath: Annotated[str, InjectedToolArg] = None,
        headers: Annotated[dict[str, str], InjectedToolArg] = None,
        on_no_content: Annotated[Literal['raise_exception', 'return_error', 'return_empty'], InjectedToolArg] = 'return_empty',
        on_error: Annotated[Literal['raise_exception', 'return_error', 'return_empty'], InjectedToolArg] = 'raise_exception'
) -> str:
    """Fetches web page or web page element and converts it to markdown.

    Args:
        url: The URL of the page.
        xpath: The xpath to the element containing the text; if not specified the entire page will be returned.
        headers: Extra headers to use for the request.
        on_no_content: What to do if no matching content is found.
        on_error: What to do if an error occurs.
    """
    content = _fetch_content(url, headers, on_no_content, on_error)
    tree = html_text.parse_html(content)
    cleaned_tree = _cleaner.clean_html(tree)

    if xpath is None:
        return html2text.html2text(
            html.tostring(cleaned_tree, pretty_print=False, encoding='unicode'),
            baseurl=url,
            bodywidth=0
        )
    else:
        texts = []
        for element in cleaned_tree.xpath(xpath):
            texts.append(html2text.html2text(
                html.tostring(element, pretty_print=False, encoding='unicode'),
                baseurl=url,
                bodywidth=0))
        if len(texts) == 0:
            return _handle_no_content(url, xpath, on_no_content, empty="")
        return '\n'.join(texts)


class FetchPageMarkdownTool(ExtendedBaseTool, BaseTool):
    name: str = "fetch_page_markdown"
    description: str = "Fetches web page or web page element and converts it to markdown."
    args_schema: type[BaseModel] = create_schema_from_function(
        name,
        _fetch_page_markdown,
        parse_docstring=True,
        error_on_invalid_docstring=True,
    )

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return _fetch_page_markdown(**kwargs)

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Fetching {input['url']}"

def _fetch_page_text(
    url: str,
    xpath: Annotated[str, InjectedToolArg] = None,
    headers: Annotated[dict[str, str], InjectedToolArg] = None,
    on_no_content: Annotated[Literal['raise_exception', 'return_error', 'return_empty'], InjectedToolArg] = 'return_empty',
    on_error: Annotated[Literal['raise_exception', 'return_error', 'return_empty'], InjectedToolArg] = 'raise_exception'
) -> str:
    """Fetches the text from web page or web page element.

    Args:
        url: The URL of the page.
        xpath: The xpath to the element containing the text; if not specified the entire page will be returned.
        headers: Extra headers to use for the request.
        on_no_content: What to do if no matching content is found.
        on_error: What to do if an error occurs.
    """
    content = _fetch_content(url, headers, on_no_content, on_error)

    tree = html_text.parse_html(content)
    cleaned_tree = html_text.cleaner.clean_html(tree)

    if xpath is None:
        return html_text.etree_to_text(cleaned_tree)
    else:
        texts = []
        for element in cleaned_tree.xpath(xpath):
            texts.append(html_text.etree_to_text(element))
        if len(texts) == 0:
            return _handle_no_content(url, xpath, on_no_content, empty="")
        return '\n'.join(texts)

class FetchPageTextTool(ExtendedBaseTool, BaseTool):
    name: str = "fetch_page_text"
    description: str = "Fetches the text from web page or web page element."
    args_schema: type[BaseModel] = create_schema_from_function(
        name,
        _fetch_page_text,
        parse_docstring=True,
        error_on_invalid_docstring=True,
    )

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return _fetch_page_text(**kwargs)

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Fetching {input['url']}"

class LinkTextPair(BaseModel):
    link: str
    text: Optional[str] = None

def _fetch_page_links(
        url: str,
        xpath: Annotated[str, InjectedToolArg] = None,
        headers: Annotated[dict[str, str], InjectedToolArg] = None,
        on_no_content: Annotated[Literal['raise_exception', 'return_error', 'return_empty'], InjectedToolArg] = 'return_empty',
        on_error: Annotated[Literal['raise_exception', 'return_error', 'return_empty'], InjectedToolArg] = 'raise_exception'
) -> list[LinkTextPair] | Json:
    """Fetches the links from web page or web page element.

    Args:
        url: The URL of the page.
        xpath: The xpath to the element containing the links; if not specified the entire page will be searched.
        headers: Extra headers to use for the request.
        on_no_content: What to do if no matching content is found.
        on_error: What to do if an error occurs.
    """
    content = _fetch_content(url, headers, on_no_content, on_error)
    tree = html_text.parse_html(content)
    cleaned_tree = html_text.cleaner.clean_html(tree)

    links = []
    found = False
    if xpath is None:
        docs = [cleaned_tree]
    else:
        docs = cleaned_tree.xpath(xpath)
    for doc in docs:
        found = True
        for link in doc.xpath('.//a[@href]'):
            href = link.get('href')
            parsed = urlparse(href)
            if parsed.scheme in ['http', 'https'] and parsed.netloc:
                text = link.text_content().strip()
                if len(text) == 0:
                    text = None
                links.append(LinkTextPair(link=href, text=text))
    if not found:
        return _handle_no_content(url, xpath, on_no_content, empty = [])
    return links

class FetchPageLinksTool(ExtendedBaseTool, BaseTool):
    name: str = "fetch_page_links"
    description: str = "Fetches the links from web page or web page element."
    args_schema: type[BaseModel] = create_schema_from_function(
        name,
        _fetch_page_links,
        parse_docstring=True,
        error_on_invalid_docstring=True,
    )

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return _fetch_page_links(**kwargs)

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Fetching links from {input['url']}"
