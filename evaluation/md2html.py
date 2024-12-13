# #!/usr/bin/python
# # -*- coding: UTF-8 -*-

# import os
# import sys

# try:
#     from markdown import markdown
# except ModuleNotFoundError as e:
#     os.system("pip install markdown")
#     os.system("pip install python-markdown-math")
#     os.system("pip install markdown_checklist")
#     from markdown import markdown


# def MD2HTML(mdstr=str):
#     exts = ['markdown.extensions.extra', 'markdown.extensions.codehilite', 'markdown.extensions.tables',
#             'markdown.extensions.toc']
#     html = '''
# <head>
# <meta content="text/html; charset=utf-8" http-equiv="content-type" />
# </head>
# <body> 
# %s
# </body>
# </html>
# '''
#     ret = markdown(mdstr, extensions=exts)
#     return html % (ret)


# def TransMD2H5(files=[], outPath=str):
#     for i in files:
#         file = os.path.basename(i)  # 获取文件名称
#         with open(i, 'r+', encoding='utf-8') as f:
#             mdStr = f.read()
#         h5Str = MD2HTML(mdStr, file.split('.')[0])
#         with open(i + '.html', 'w+', encoding='utf-8') as f:
#             err = f.write(h5Str)
#         print('write %s to html finished' % (file))


# def CheckInputArgs(input=[]):
#     for i in input:
#         file = os.path.basename(i)  # 获取文件名称
#         sufix = file.split('.')[-1]
#         if not sufix.lower() == 'md':
#             return False
#     return True


# if __name__ == '__main__':
#     if CheckInputArgs(sys.argv[1:]) is True:
#         TransMD2H5(sys.argv[1:], './')
#     else:
#         print('input file(%s) is not md file, program will do nothing & exit.' % (sys.argv[1:]))
import os
import argparse
import markdown
from bs4 import BeautifulSoup
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

def convert_md_to_html(md_text, light_mode=True):
    html = markdown.markdown(md_text,
                             extensions=['fenced_code', 'tables', 'toc', 'footnotes', 'attr_list', 'md_in_html'])
    soup = BeautifulSoup(html, 'lxml')

    for code in soup.find_all('code'):
        parent = code.parent
        if parent.name == 'pre':
            language = code.get('class', [''])[0].replace('language-', '') or 'text'
            lexer = get_lexer_by_name(language, stripall=True)
            formatter = HtmlFormatter(style='default' if light_mode else 'monokai')
            try:
                highlighted_code = highlight(code.string, lexer, formatter)
                code.replace_with(BeautifulSoup(highlighted_code, 'lxml'))
            except:
                continue

            copy_button_html = f'''
            <div class="code-header">
                <span class="language-label">{language}</span>
                <button class="copy-button" onclick="copyCode(this)">
                    <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
                        <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path>
                        <path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
                    </svg>
                </button>
            </div>
            '''
            parent.insert_before(BeautifulSoup(copy_button_html, 'lxml'))

    return soup.prettify()

def md2html(string):
    
    input = string
    html = convert_md_to_html(input, light_mode="dark")
    #pred = html
    #with open('output\\output.html', 'w', encoding='utf-8') as html_file:
        #html_file.write(html)
    #print(f"Markdown converted to HTML successfully! Output saved to {html_file}")
    return html
    