from setuptools import setup
  
setup(
    name='acpsr',
    version='0.1',
    description='acupoint speech recognition',
    author='JinYu Long',
    author_email='sicnu.long@gmail.com',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'pandas',
        'torchaudio',
        'librosa',
        'scipy',
        'matplotlib',
        'timm==0.4.5',
        'wget',
    ],
)
