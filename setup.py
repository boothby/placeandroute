from setuptools import setup

setup(
    name='placeandroute',
    version='0.0.0',
    packages=['placeandroute','placeandroute.routing',
              'placeandroute.tilebased'],
    install_requires=['networkx', 'dwave-networkx', 'six'],
    url='https://bitbucket.org/StefanoVt/placeandroute/src/master/',
    license='MIT',
    author='Stefano Varotti',
    author_email='stefano.varotti@unitn.it',
    description=''
)
