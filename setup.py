import setuptools

setuptools.setup(
    name="pyspark-streamlit",
    version="0.0.1",
    author="Shruti Agrawal",
    author_email="shrutiagr2806@gmail.com",
    description="Liver disease prediction",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[],
    python_requires=">=3.8",
    install_requires=[
        "streamlit >= 0.66",
        "pyspark >= 3.0.0"
    ],
)