import re

with open('Paper/sections/04_results.tex') as f:
    text = f.read()

# Replace **text** with \textbf{text}
def repl_bold(match):
    return r'\textbf{' + match.group(1) + '}'

text = re.sub(r'\*\*(.*?)\*\*', repl_bold, text)

# Replace markdown lists with itemize / enumerate where needed
lines = text.split('\n')
new_lines = []
in_list = False
for line in lines:
    if line.strip().startswith('* '):
        item_text = line.strip()[2:]
        if not in_list:
            new_lines.append(r'\begin{itemize}')
            in_list = True
        new_lines.append(r'    \item ' + item_text)
    elif line.strip().startswith('1. ') or line.strip().startswith('2. ') or line.strip().startswith('3. '):
        # Check if inside an enumerate
        new_lines.append(line)
    else:
        if in_list and (line.strip() == '' or line.strip().startswith('\\') or line.strip().startswith('---')):
            new_lines.append(r'\end{itemize}')
            in_list = False
        new_lines.append(line)

if in_list:
    new_lines.append(r'\end{itemize}')

text = '\n'.join(new_lines)

# Remove markdown horizontal rules '---'
text = re.sub(r'\n---\n', '\n\n', text)

with open('Paper/sections/04_results.tex', 'w') as f:
    f.write(text)

print("Cleaned markdown artifacts from 04_results.tex!")
