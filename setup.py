import setuptools

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pyRaTS',                     
    version='0.21',                        
    author='Arvid Trapp',
    author_email='arvid.trapp@hm.edu',
    url='https://github.com/ArvidTrapp/pyRaTS',
    maintainer='Arvid Trapp, Peter Wolfsteiner',
    maintainer_email='arvid.trapp@hm.edu',                    
    description='processing of (RAndom) TimeSeries for vibration fatigue',
    long_description=long_description,      
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),    
    keywords=['vibration fatigue, non-stationarity matrix, structural dynamics', 'Fatigue Damage Spectrum'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],                                      
    python_requires='>=3.6',                
    py_modules=['pyRaTS'],             
    package_dir={'':'pyRaTS/src'},     
)
    
    
