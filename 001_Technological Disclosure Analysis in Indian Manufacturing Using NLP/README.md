# Technological Disclosure Analysis in Indian Manufacturing using NLP

## Overview

This project analyzes how Indian manufacturing companies communicate their **technology adoption and digital transformation strategies** in their annual reports.

Using **Natural Language Processing (NLP)**, the study extracts technology-related terms from **103 annual reports (2013–2025)** across major Indian companies to identify trends in **AI, automation, cybersecurity, IoT, and digital transformation**.

The goal is to quantify how firms signal their technological capabilities through corporate communication.

---

## Business Problem

Manufacturing companies are rapidly adopting technologies such as:

- Artificial Intelligence
- Automation
- Industry 4.0
- IoT
- Cybersecurity
- Data Analytics

However, measuring **digital maturity** across firms is difficult.

Annual reports contain rich information but are **unstructured text documents**.  
This project converts these documents into **structured insights using NLP**.

---

## Dataset

The dataset includes **103 annual reports** from the following companies:

- Adani Power
- Asian Paints
- Britannia Industries
- Dabur India
- Hindustan Unilever
- Tata Steel
- UltraTech Cement
- Varun Beverages
- United Spirits
- Vedanta

Reports were collected from **BSE/NSE filings and company investor relations websites**.

---

## Technologies Used

- Python
- Natural Language Processing (NLP)
- Pandas
- pdfplumber
- PyMuPDF
- Regex
- WordCloud
- Matplotlib

---

## Methodology

### 1. Data Collection
Annual reports were downloaded and text was extracted using:

- `pdfplumber`
- `PyMuPDF`

---

### 2. Technology Vocabulary

A dictionary of **80+ technology terms** was created across categories:

- AI & Machine Learning
- Automation
- Data Analytics
- Cloud & IT Infrastructure
- ERP Systems
- IoT
- Industry 4.0
- Cybersecurity
- Digital Transformation
- Sustainability Technologies

---

### 3. NLP Processing

Steps performed:

1. Convert text to lowercase
2. Apply regex-based matching
3. Count technology term frequencies
4. Create company-level and year-level datasets

---

### 4. Visualization

The results were visualized using:

- Word Clouds
- Trend charts
- Company comparison charts
- Technology category analysis

---

## Key Insights

**AI Adoption Surge**

Companies like Tata Steel and Adani Power show strong growth in AI mentions after 2022.

**Cybersecurity Growth**

Cybersecurity discussions increased significantly after 2021.

**Automation Dominance**

Automation appears consistently across manufacturing companies.

**Digital Transformation Focus**

Hindustan Unilever shows a strong focus on digital transformation strategy.

---

## Example Insights

| Company | Key Technology Focus |
|------|------|
| Tata Steel | AI & Industry 4.0 |
| Asian Paints | Robotic Process Automation |
| Dabur | Automation & IoT |
| HUL | Digital Transformation |
| Adani Power | Cybersecurity |

---

## Future Improvements

- Sentiment analysis of technology discussions
- Topic modeling (LDA)
- Machine learning-based classification
- Cross-country industry comparison

---

## Author

**Chitranjan Kumar**

