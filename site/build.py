"""Build the static site: inject app_data/*.json into site/template.html.

Outputs
  docs/index.html          full HTML document for GitHub Pages
  docs/sim_sample.json     data sample for the in-browser simulator
  showcase/index.html      fragment version (only if showcase/ exists locally)
"""
import json, os, shutil
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
j = lambda n: json.load(open(os.path.join(root, 'app_data', n), encoding='utf-8'))
data = {'profile': j('profile.json'), 'fl': j('fl_results.json'), 'base': j('baselines.json'), 'v2': j('v2_results.json'), 'runDate': 'Nov 2026'}
data['profile'].pop('feature_names', None)
tpl = open(os.path.join(root, 'site', 'template.html'), encoding='utf-8').read()
body = tpl.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))

# GitHub Pages: wrap in a full document (claude.ai adds this skeleton itself)
head_end = body.index('</style>') + len('</style>')
full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="Privacy-preserving federated learning for hospitals: animated protocol, measured results, live simulator.">\n'
        '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n'
        + body[:head_end] + '\n</head>\n<body>\n' + body[head_end:] + '\n</body>\n</html>\n')
os.makedirs(os.path.join(root, 'docs'), exist_ok=True)
open(os.path.join(root, 'docs', 'index.html'), 'w', encoding='utf-8').write(full)
shutil.copyfile(os.path.join(root, 'app_data', 'sim_sample.json'), os.path.join(root, 'docs', 'sim_sample.json'))
open(os.path.join(root, 'docs', '.nojekyll'), 'w').close()
print('docs/index.html', len(full))
sc = os.path.join(root, 'showcase')
if os.path.isdir(sc):
    open(os.path.join(sc, 'index.html'), 'w', encoding='utf-8').write(body)
    print('showcase/index.html', len(body))
