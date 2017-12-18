from distutils.core import setup

setup(
    name='wikt_morphodecomp',
    version='0.1',
    packages=['wikt_morphodecomp', 'wikt_morphodecomp.ml', 'wikt_morphodecomp.test', 'wikt_morphodecomp.config',
              'wikt_morphodecomp.data_access'],
    url='',
    license='',
    author='Danilo S. Carvalho',
    author_email='danilo@jaist.ac.jp',
    description='Morphological decomposition seq2seq RNN trained on Wiktionary data',
    install_requires=[
        'saf',
        'numpy',
        'keras',
        'gensim',
        'web.py'
    ]
)
