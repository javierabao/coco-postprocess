#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Object-oriented HTML report generation for COCO post-processing.

This module provides a cleaner, more maintainable way to manage HTML report
generation with support for:
- Creating index files immediately
- Incrementally updating HTML pages as they're generated
- Opening results in browser for real-time monitoring
- Centralized management of all report files
"""

from __future__ import absolute_import, print_function

import os
import time
import webbrowser
import shutil
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from . import toolsdivers


class PageType:
    """Enum for different page types in the report."""
    SINGLE_ALGORITHM = 'single_algorithm'
    MULTI_ALGORITHM = 'multi_algorithm'
    DIMENSION_COMPARISON = 'dimension_comparison'
    SCATTER_PLOT = 'scatter_plot'
    TABLE_COMPARISON = 'table_comparison'
    ECDF_PLOT = 'ecdf_plot'
    CUSTOM = 'custom'


class HtmlBuilder:
    """Utility class for building HTML content."""
    
    HTML_HEADER_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 25px;
            border-left: 4px solid #007bff;
            padding-left: 10px;
        }}
        h3 {{
            color: #666;
        }}
        a {{
            color: #007bff;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .link-list {{
            list-style-type: none;
            padding-left: 0;
        }}
        .link-list li {{
            padding: 5px 0;
        }}
        .link-list li:before {{
            content: "▸ ";
            color: #007bff;
            margin-right: 8px;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.9em;
            font-style: italic;
        }}
        .status {{
            padding: 10px;
            border-radius: 3px;
            margin: 10px 0;
        }}
        .status.success {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .status.in-progress {{
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
        }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .image-item {{
            border: 1px solid #ddd;
            border-radius: 3px;
            padding: 10px;
            text-align: center;
        }}
        .image-item img {{
            max-width: 100%;
            height: auto;
        }}
        .toc {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 3px;
            margin: 15px 0;
        }}
        .toc ul {{
            margin: 0;
            padding-left: 20px;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>{title}</h1>
    <p class="timestamp">Generated on {timestamp}</p>
    {toc}
    {content}
</div>
</body>
</html>"""

    @staticmethod
    def create_header(title: str, timestamp: Optional[str] = None, toc_html: str = "") -> str:
        """Create HTML header."""
        if timestamp is None:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        return HtmlBuilder.HTML_HEADER_TEMPLATE.format(
            title=title,
            timestamp=timestamp,
            toc=toc_html,
            content="<!--CONTENT_PLACEHOLDER-->"
        )
    
    @staticmethod
    def create_section(section_title: str, content: str = "") -> str:
        """Create HTML section."""
        return f"<h2>{section_title}</h2>\n{content}\n"
    
    @staticmethod
    def create_link_list(links: List[Tuple[str, str]]) -> str:
        """Create an HTML list of links. Args: list of (label, url) tuples."""
        html = '<ul class="link-list">\n'
        for label, url in links:
            html += f'    <li><a href="{url}">{label}</a></li>\n'
        html += '</ul>\n'
        return html
    
    @staticmethod
    def create_status_message(message: str, status: str = "success") -> str:
        """Create a status message (success, in-progress, error)."""
        return f'<div class="status {status}">{message}</div>\n'


class PageBuilder:
    """Builder for individual HTML pages with incremental updates."""
    
    def __init__(self, name: str, output_dir: str, page_type: str = PageType.CUSTOM,
                 title: str = "", parent_link: Optional[Tuple[str, str]] = None):
        """Initialize page builder.
        
        Args:
            name: Page name (used for filename)
            output_dir: Directory where HTML will be saved
            page_type: Type of page (for customizing appearance)
            title: Page title
            parent_link: Tuple of (label, url) for parent link
        """
        self.name = name
        self.output_dir = output_dir
        self.page_type = page_type
        self.title = title if title else name
        self.parent_link = parent_link
        self.file_path = os.path.join(output_dir, f"{name}.html")
        self.sections: OrderedDict[str, List[str]] = OrderedDict()
        self.table_of_contents: List[str] = []
        self._create_initial_file()
    
    def _create_initial_file(self):
        """Create initial skeleton HTML file."""
        header = HtmlBuilder.create_header(self.title)
        
        # Add parent link if provided
        parent_link_html = ""
        if self.parent_link:
            parent_link_html = f'<p><a href="{self.parent_link[1]}">← {self.parent_link[0]}</a></p>\n'
        
        html = header.replace(
            "<!--CONTENT_PLACEHOLDER-->",
            parent_link_html + '<div id="content"></div>'
        )
        
        with open(self.file_path, 'w') as f:
            f.write(html)
    
    def add_section(self, section_title: str):
        """Add a new section to the page."""
        if section_title not in self.sections:
            self.sections[section_title] = []
            self.table_of_contents.append(section_title)
    
    def add_content_to_section(self, section_title: str, content: str):
        """Add HTML content to a section."""
        if section_title not in self.sections:
            self.add_section(section_title)
        self.sections[section_title].append(content)
        self.save()
    
    def add_image(self, image_path: str, caption: str = "", height: int = 160):
        """Add an image to the current section."""
        img_html = '<div class="image-item">'
        if caption:
            img_html += '<p><strong>{}</strong></p>'.format(caption)
        img_html += '<img src="{}" height="{}px">'.format(image_path, height)
        img_html += '</div>\n'
        
        # Add to first section or create default section
        if not self.sections:
            self.add_section("Content")
        
        first_section = next(iter(self.sections))
        self.sections[first_section].append(img_html)
        self.save()
    
    def add_gallery(self, images: List[Tuple[str, str]], section_title: str = "Gallery"):
        """Add a gallery of images."""
        self.add_section(section_title)
        gallery_html = '<div class="image-gallery">\n'
        for img_path, caption in images:
            gallery_html += '<div class="image-item">'
            if caption:
                gallery_html += '<p>{}</p>'.format(caption)
            gallery_html += '<a href="{}"><img src="{}" style="max-width: 100%;"></a>'.format(img_path, img_path)
            gallery_html += '</div>\n'
        gallery_html += '</div>\n'
        self.add_content_to_section(section_title, gallery_html)
    
    def add_table(self, table_html: str, section_title: str = "Tables"):
        """Add a table to a section."""
        self.add_section(section_title)
        self.add_content_to_section(section_title, table_html)
    
    def add_link_list(self, links: List[Tuple[str, str]], section_title: str = "Links"):
        """Add a list of links."""
        self.add_section(section_title)
        link_html = HtmlBuilder.create_link_list(links)
        self.add_content_to_section(section_title, link_html)
    
    def add_status_message(self, message: str, status: str = "success"):
        """Add a status message."""
        status_html = HtmlBuilder.create_status_message(message, status)
        if not self.sections:
            self.add_section("Status")
        first_section = next(iter(self.sections))
        self.sections[first_section].append(status_html)
        self.save()
    
    def save(self):
        """Save page to disk with all accumulated content."""
        # Generate table of contents
        toc_html = ""
        if len(self.sections) > 3:  # Only show TOC if there are multiple sections
            toc_html = '<div class="toc"><strong>Table of Contents:</strong>\n<ul>\n'
            for section_title in self.table_of_contents:
                # Create anchor-friendly title
                anchor = section_title.lower().replace(" ", "-")
                toc_html += f'  <li><a href="#{anchor}">{section_title}</a></li>\n'
            toc_html += '</ul></div>\n'
        
        # Generate content sections
        content_html = ""
        for section_title in self.table_of_contents:
            if section_title in self.sections and self.sections[section_title]:
                anchor = section_title.lower().replace(" ", "-")
                content_html += f'<h2 id="{anchor}">{section_title}</h2>\n'
                for item in self.sections[section_title]:
                    content_html += item + '\n'
        
        # Get header and insert content
        header = HtmlBuilder.create_header(self.title, toc_html=toc_html)
        html = header.replace("<!--CONTENT_PLACEHOLDER-->", content_html)
        
        with open(self.file_path, 'w') as f:
            f.write(html)


class IndexPage(PageBuilder):
    """Special page for main index."""
    
    def __init__(self, output_dir: str, title: str = "COCO Post-Processing Results"):
        """Initialize index page."""
        super().__init__('index', output_dir, PageType.CUSTOM, title)


class HtmlReportManager:
    """Central manager for HTML report generation with incremental updates."""
    
    def __init__(self, output_dir: str, title: str = "COCO Post-Processing Results",
                 auto_open_browser: bool = True, verbose: bool = True):
        """Initialize report manager.
        
        Args:
            output_dir: Directory where all report files will be saved
            title: Title for the main index page
            auto_open_browser: Whether to open index in browser immediately
            verbose: Whether to print status messages
        """
        self.output_dir = output_dir
        self.title = title
        self.verbose = verbose
        self.pages: Dict[str, PageBuilder] = {}
        self.auto_open_browser = auto_open_browser
        
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create index immediately
        self.index_page = IndexPage(output_dir, title)
        if self.verbose:
            print(f"✓ Index file created at: {self.index_page.file_path}")
        
        # Open in browser immediately
        if auto_open_browser:
            self.open_in_browser()
    
    def create_page(self, page_name: str, page_type: str = PageType.CUSTOM,
                    page_title: str = "") -> PageBuilder:
        """Create a new page.
        
        Args:
            page_name: Name of the page (used for filename)
            page_type: Type of page (for styling)
            page_title: Display title of the page
            
        Returns:
            PageBuilder instance for the new page
        """
        if not page_title:
            page_title = page_name
        
        parent_link = (self.title, "index.html")
        page = PageBuilder(page_name, self.output_dir, page_type, page_title, parent_link)
        self.pages[page_name] = page
        
        if self.verbose:
            print(f"✓ Created page: {page_name}")
        
        return page
    
    def add_section(self, section_name: str):
        """Add a new section to the index.
        
        Args:
            section_name: Name of the section
        """
        self.index_page.add_section(section_name)
    
    def add_link_to_index(self, section_name: str, link_text: str, link_url: str):
        """Add a link to a section in the index (incremental update).
        
        This is the key method for progressive feedback - it allows adding links
        to the index as they become available without waiting for all processing.
        
        Args:
            section_name: Name of section in index
            link_text: Text to display for the link
            link_url: URL the link points to
        """
        if section_name not in self.index_page.sections:
            self.index_page.add_section(section_name)
        
        link_html = f'<a href="{link_url}">{link_text}</a>'
        self.index_page.add_content_to_section(section_name, link_html)
        
        if self.verbose:
            print(f"✓ Added link: {link_text} → {link_url}")
    
    def add_status_to_index(self, message: str, status: str = "success"):
        """Add a status message to the index."""
        status_section = "Status"
        if status_section not in self.index_page.sections:
            self.index_page.add_section(status_section)
        
        status_html = HtmlBuilder.create_status_message(message, status)
        self.index_page.add_content_to_section(status_section, status_html)
    
    def get_page(self, page_name: str) -> Optional[PageBuilder]:
        """Get an existing page.
        
        Args:
            page_name: Name of the page
            
        Returns:
            PageBuilder instance or None if not found
        """
        return self.pages.get(page_name)
    
    def open_in_browser(self):
        """Open index file in default browser."""
        try:
            file_url = 'file://' + os.path.abspath(self.index_page.file_path)
            webbrowser.open(file_url)
            if self.verbose:
                print(f"✓ Opened index in browser: {file_url}")
        except Exception as e:
            if self.verbose:
                print(f"⚠ Could not open browser: {e}")
    
    def save_all(self):
        """Save all pages to disk."""
        self.index_page.save()
        for page in self.pages.values():
            page.save()
        if self.verbose:
            print("✓ All pages saved")
    
    def get_index_path(self) -> str:
        """Get the absolute path to the index file."""
        return self.index_page.file_path

    def copy_static_files(self):
        """Copy static assets from the package into the report output directory."""
        folder = os.path.join(toolsdivers.path_in_package(), "static")
        if not os.path.isdir(folder):
            if self.verbose:
                print(f"⚠ Static folder not found: {folder}")
            return
        for file_in_folder in os.listdir(folder):
            src = os.path.join(folder, file_in_folder)
            dst = os.path.join(self.output_dir, file_in_folder)
            try:
                shutil.copy(src, dst)
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Could not copy static file {src} -> {dst}: {e}")
        if self.verbose:
            print(f"✓ Copied static files to: {self.output_dir}")


# Backward compatibility: function wrappers to ease migration
def create_report_manager(output_dir: str, title: str = "COCO Post-Processing Results",
                          auto_open_browser: bool = True) -> HtmlReportManager:
    """Convenience function to create a report manager."""
    return HtmlReportManager(output_dir, title, auto_open_browser)
