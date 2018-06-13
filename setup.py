from setuptools import setup

setup(
    name='placeandroute',
    version='0.0.0',
    packages=['placeandroute','placeandroute.routing',
              'placeandroute.tilebased'],
    install_requires=['networkx>=2.0,<3.0',
                      'dwave-networkx>=0.6.3,<0.7.0',
                      'six>=1.11.0,<2.0.0'
                      ],
    url='https://bitbucket.org/StefanoVt/placeandroute/src/master/',
    license='MIT',
    author='Stefano Varotti',
    author_email='stefano.varotti@unitn.it',
    description=''
)
