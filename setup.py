from setuptools import find_packages, setup

requirements = [
    "mlx==0.30.0; platform_system == 'Darwin'",
    "tokenizerz==0.0.3a4",
    "tqdm",
    "datasets",
    # "lm-eval==0.4.9.1", "matplotlib"
]

extras_require = {
    'cuda': ['mlx[cuda]'],
    'cpu':  ['mlx[cpu]'],
    'no_mlx': [],
}

setup(
    name='rcrlm',
    url='https://github.com/JosefAlbers/rcrlm',
    packages=find_packages(),
    version='0.0.2',
    readme="README.md",
    author_email="albersj66@gmail.com",
    description="rcr-lm: Collapse layers into a recurrent block",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="J Joe",
    license="Apache-2.0",
    # python_requires=">=3.12.8",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "rlm=rcrlm.main:cli",
        ],
    },
)
