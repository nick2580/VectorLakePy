from distutils.core import setup
setup(
    name = 'VectorLakePy',         
    packages = ['VectorLakePy'],   
    version = '0.1',      
    license='MIT',        
    description = '',  
    author = 'Nikhil',                   
    author_email = 'nikhil.bhamere@gmail.com',      
    url = 'https://github.com/nick2580/VecStore',   
    download_url = 'https://github.com/nick2580/VecStore/archive/refs/tags/0.1.tar.gz',
    keywords = ['vectors', 'embedding', 'polars'],   
    install_requires=[            # I get to this in a second
            'openai',
            'polars',
        ],
    classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',     
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    ],
)