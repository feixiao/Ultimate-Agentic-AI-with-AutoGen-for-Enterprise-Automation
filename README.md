# Ultimate Agentic AI for Enterprise Agentic Solutions

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This project utilizes various Python libraries for natural language processing, machine learning, and vector search capabilities. It combines transformer-based models with efficient similarity search algorithms to process and analyze text data.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Running the Code](#running-the-code)
- [Dependencies](#dependencies)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Contact](#contact)

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git
- Virtual environment tool (recommended)
- 8GB+ RAM recommended for running larger models



## Setup Instructions

### Manual Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your_username>/enterprise-agentic-ai.git
   cd enterprise-agentic-ai
   ```

2. **Create and activate a virtual environment**:

   For macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

   <!-- uv python install  3.12 3.14   
   uv venv
   source .venv/bin/activate -->
   ```

   For Windows:
   ```bash
   python3 -m venv .venv
   .venv\Scripts\activate
   ```

   > Note: If you need to recreate your virtual environment, remove the existing one first with `rm -rf .venv` (macOS/Linux) or `rmdir /s /q .venv` (Windows).

3. **Update pip and install dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `src/chapters/` - Contains individual chapter code implementations

## Running the Code

### Running Individual Chapters

Navigate to a chapter folder and execute:

```bash
cd chap01
python ch1_first_agent.py
```

**NOTE:** Three chapters (ch02, ch09 and ch10) are theoretical and don't have any code.

## Best Practices

1. **Security**
   - Never commit API keys to version control
   - Use environment variables or secure vaults

2. **Performance**
   - Batch process data when possible
   - Cache embeddings for frequently used data
   - Use quantized models for deployment

3. **Updates**
   - Regularly check for security updates:
     ```bash
     pip list --outdated
     ```
