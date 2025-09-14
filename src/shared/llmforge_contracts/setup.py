from setuptools import setup, find_packages

setup(
    name="llmforge-contracts",
    version="1.0.0",
    description="Shared contracts for LLMForge microservices communication",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.4.2",
    ],
    python_requires=">=3.11",
)