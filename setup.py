from setuptools import setup

with open("README", "r") as fh:
    long_description = fh.read()


setup(
    name='placeandroute',
    version='0.0.0',
    packages=['placeandroute','placeandroute.routing',
              'placeandroute.tilebased'],
    install_requires=['networkx>=2.4,<3.0',
                      'dwave-networkx>=0.8.2',
                      'matplotlib',
                      ],
    extras_requires= {"cpp": "placeandroutecpp"},
    url='https://bitbucket.org/StefanoVt/placeandroute/',
    license='MIT',
    author='Stefano Varotti',
    author_email='stefano.varotti@unitn.it',
    description='graph placement and routing library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers = [
                  "Programming Language :: Python :: 3",
                  "License :: OSI Approved :: MIT License",
                  "Operating System :: OS Independent",
              ],

)
