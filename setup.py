import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="human_detection",
    version="0.51",
    author="Lior Israeli",
    author_email="israelilior@gmail.com",
    description="set usb camera as security camera, trigger or human detection and send detection video over mail",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lisrael1/human_detection",
    project_urls={
        "Bug Tracker": "https://github.com/lisrael1/human_detection/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={"": ["*.xlsx"]},
    install_requires=[['dynaconf', 'yagmail', 'opencv-python', 'matplotlib']],
    entry_points={
        'console_scripts': ['human_motion_detection=human_detection.command_line:main'],
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude=['*_tests', '*_examples'], ),
    python_requires=">=3.6",
)
