#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import gc
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud


# In[2]:


COMPANY_NAME = "Vedanta"

DATA_FOLDER = r"C:\Users\chitr\OneDrive - Intsitute of Management Technology-Hyderabad\IMT HYDERABAD\Management Project\Companies\vedanta"


# In[3]:


pdf_files = [
    os.path.join(DATA_FOLDER, f)
    for f in os.listdir(DATA_FOLDER)
    if f.lower().endswith(".pdf")
]

print("Total PDFs found:", len(pdf_files))


# In[4]:


TECH_VOCAB = {

    # AI / ML
    "artificial intelligence", "ai", "machine learning", "ml",
    "deep learning", "neural networks", "computer vision",
    "natural language processing", "nlp", "predictive modeling",
    "decision intelligence", "intelligent systems",

    # Data & Analytics
    "data analytics", "advanced analytics", "big data",
    "data science", "data-driven", "business analytics",
    "real-time analytics", "predictive analytics",
    "prescriptive analytics", "data platforms", "analytics platform",

    # Automation
    "automation", "process automation", "robotic process automation",
    "rpa", "automated systems", "automated operations",
    "robotics", "industrial automation", "autonomous systems",

    # Cloud & IT
    "cloud computing", "cloud infrastructure", "cloud platform",
    "hybrid cloud", "edge computing", "it infrastructure",
    "digital infrastructure", "software platforms",
    "enterprise systems",

    # ERP & Enterprise Tech
    "erp", "enterprise resource planning", "sap", "oracle",
    "it systems", "digital core", "it modernization",
    "legacy systems", "system integration",
    "application modernization",

    # IoT & Smart
    "internet of things", "iot", "smart systems",
    "smart infrastructure", "connected devices",
    "remote monitoring", "sensor-based monitoring",
    "digital sensors",

    # Industry 4.0
    "industry 4.0", "smart manufacturing",
    "digital manufacturing", "advanced manufacturing",
    "predictive maintenance", "digital twin",
    "process optimization", "operational analytics",
    "condition monitoring", "asset performance management",
    "advanced process control", "plant automation",
    "smart operations",

    # Cybersecurity
    "cybersecurity", "information security",
    "data security", "cyber risk", "cyber resilience",
    "blockchain", "distributed ledger technology",

    # Digital Transformation
    "digital transformation", "digital initiatives",
    "digital technologies", "technology-driven",
    "digital platforms", "technology adoption",
    "technology-enabled", "digital roadmap",
    "digital strategy",

    # Sustainability + Tech
    "energy analytics", "emissions monitoring",
    "carbon tracking", "digital sustainability",
    "environmental monitoring", "smart energy management",
    "renewable integration", "climate analytics"
}


# In[5]:


TECH_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in TECH_VOCAB) + r")\b"
)


# In[6]:


yearwise_freq = {}
all_years_terms = []

for pdf_path in sorted(pdf_files):

    filename = os.path.basename(pdf_path)
    year = re.search(r"(20\d{2})", filename)
    year = year.group(1) if year else None

    if not year:
        continue

    print("\n===================================")
    print(f"{COMPANY_NAME} – {year}")
    print("===================================")

    freq = Counter()

    # 🔥 USE PyMuPDF INSTEAD OF pdfplumber
    doc = fitz.open(pdf_path)

    for page in doc:
        text = page.get_text("text")  # lightweight extraction
        if text:
            text = text.lower()
            matches = TECH_PATTERN.findall(text)
            freq.update(matches)

    doc.close()

    yearwise_freq[year] = freq
    all_years_terms.extend(freq.elements())

    if freq:
        df = (
            pd.DataFrame(freq.items(), columns=["Tech_Term", "Frequency"])
            .sort_values("Frequency", ascending=False)
        )

        display(df.head(20))

        wc = WordCloud(
            width=900,
            height=450,
            background_color="white",
            max_words=200
        ).generate_from_frequencies(freq)

        plt.figure(figsize=(12,5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{COMPANY_NAME} – Technology WordCloud ({year})")
        plt.show()

        plt.close()

    del freq
    gc.collect()


# In[7]:


print("\n===================================")
print(f"{COMPANY_NAME} – CONSOLIDATED (ALL YEARS)")
print("===================================")

consolidated_freq = Counter(all_years_terms)

df_all = (
    pd.DataFrame(consolidated_freq.items(), columns=["Tech_Term", "Frequency"])
    .sort_values("Frequency", ascending=False)
)

display(df_all.head(30))

wc_all = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    max_words=300
).generate_from_frequencies(consolidated_freq)

plt.figure(figsize=(14,6))
plt.imshow(wc_all, interpolation="bilinear")
plt.axis("off")
plt.title(f"{COMPANY_NAME} – Consolidated Technology WordCloud")
plt.show()

plt.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




